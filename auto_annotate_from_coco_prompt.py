import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.nn import DataParallel
from torchvision.ops import nms, roi_align
from torchvision.transforms import functional as TF
from tqdm import tqdm

from utils.arg_parser import get_argparser
from utils.box_ops import boxes_with_scores
from utils.data import resize_and_pad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-annotate an image folder using one prompt image + its COCO annotations. "
            "Outputs a CVAT-style COCO instances JSON."
        ),
        parents=[get_argparser()],
    )
    parser.add_argument("--prompt-image", type=str, required=True, help="Path to prompt image used in CVAT.")
    parser.add_argument(
        "--prompt-coco-json",
        type=str,
        required=True,
        help="COCO JSON containing annotations for the prompt image.",
    )
    parser.add_argument("--target-images-dir", type=str, required=True, help="Folder with images to annotate.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="./instances_default.json",
        help="Output COCO JSON path (instances_default.json style).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Model checkpoint path. If omitted, script tries "
            "./CNTQG_multitrain_ca44.pth and <model_path>/<model_name>.pth."
        ),
    )
    parser.add_argument(
        "--prompt-image-id",
        type=int,
        default=None,
        help="Optional COCO image id for prompt image. If omitted, resolved by filename.",
    )
    parser.add_argument(
        "--use-all-categories",
        action="store_true",
        help="Run per category from prompt image. By default, only first prompt category is used.",
    )
    parser.add_argument(
        "--max-exemplars-per-category",
        type=int,
        default=20,
        help="Maximum prompt boxes to use per category.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.33,
        help="Selection threshold in GECO2 style (higher keeps fewer boxes).",
    )
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Optional minimum confidence filter using model score.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff",
        help="Comma-separated target image extensions.",
    )
    return parser.parse_args()


def _to_xyxy(box_xywh: List[float]) -> List[float]:
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]


def _load_prompt_boxes_by_category(
    prompt_coco_path: str,
    prompt_image_path: str,
    prompt_image_id: int = None,
    use_all_categories: bool = False,
    max_exemplars: int = 20,
) -> Tuple[Dict[int, torch.Tensor], List[Dict]]:
    with open(prompt_coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    if not categories:
        raise ValueError("No categories found in prompt COCO JSON.")

    if prompt_image_id is None:
        prompt_name = Path(prompt_image_path).name
        matched = [img for img in images if Path(img.get("file_name", "")).name == prompt_name]
        if len(matched) != 1:
            raise ValueError(
                "Could not uniquely resolve prompt image in COCO JSON by filename. "
                "Use --prompt-image-id explicitly."
            )
        prompt_image_id = int(matched[0]["id"])

    prompt_anns = [ann for ann in annotations if int(ann.get("image_id", -1)) == prompt_image_id]
    if not prompt_anns:
        raise ValueError("No annotations found for prompt image in COCO JSON.")

    boxes_by_cat: Dict[int, List[List[float]]] = {}
    for ann in prompt_anns:
        cat_id = int(ann["category_id"])
        if "bbox" not in ann or len(ann["bbox"]) != 4:
            continue
        boxes_by_cat.setdefault(cat_id, []).append(_to_xyxy(ann["bbox"]))

    if not boxes_by_cat:
        raise ValueError("Prompt image has no valid bbox annotations.")

    selected_cats = sorted(boxes_by_cat.keys())
    if not use_all_categories:
        selected_cats = [selected_cats[0]]

    tensor_by_cat: Dict[int, torch.Tensor] = {}
    for cat_id in selected_cats:
        boxes = boxes_by_cat[cat_id][:max_exemplars]
        tensor_by_cat[cat_id] = torch.tensor(boxes, dtype=torch.float32)

    selected_categories = [c for c in categories if int(c["id"]) in selected_cats]
    return tensor_by_cat, selected_categories


def _list_target_images(folder: str, extensions_csv: str) -> List[Path]:
    allowed = {ext.strip().lower() for ext in extensions_csv.split(",") if ext.strip()}
    out = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in allowed:
            out.append(p)
    return out


def _load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img).to(device)
    return TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def _run_cross_image_inference(
    model_module,
    prompt_img_norm: torch.Tensor,
    prompt_boxes_xyxy: torch.Tensor,
    target_img_norm: torch.Tensor,
    score_threshold: float,
    nms_iou: float,
    min_confidence: float,
    device: torch.device,
) -> Tuple[List[List[float]], List[float]]:
    prompt_img_1024, prompt_boxes_1024, _ = resize_and_pad(
        prompt_img_norm,
        prompt_boxes_xyxy.to(device),
        size=1024.0,
        zero_shot=True,
    )
    prompt_img_1024 = prompt_img_1024.unsqueeze(0)

    dummy = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    target_img_1024, _, target_scale = resize_and_pad(
        target_img_norm,
        dummy,
        size=1024.0,
        zero_shot=True,
    )
    target_img_1024 = target_img_1024.unsqueeze(0)

    with torch.no_grad():
        prompt_feats = model_module.backbone(prompt_img_1024)
        target_feats = model_module.backbone(target_img_1024)

    src_prompt = prompt_feats["vision_features"]
    src_target = target_feats["vision_features"]

    num_objects = int(prompt_boxes_1024.shape[0])
    reduction = 1024.0 / src_prompt.shape[-1]
    kernel_dim = 1

    bboxes_roi = torch.cat(
        [
            torch.zeros((num_objects, 1), dtype=torch.float32, device=device),
            prompt_boxes_1024,
        ],
        dim=1,
    )

    exemplars = roi_align(
        src_prompt,
        boxes=bboxes_roi,
        output_size=kernel_dim,
        spatial_scale=1.0 / reduction,
        aligned=True,
    ).permute(0, 2, 3, 1).reshape(1, num_objects * kernel_dim**2, model_module.emb_dim)

    l1 = prompt_feats["backbone_fpn"][0]
    l2 = prompt_feats["backbone_fpn"][1]
    exemplars_l1 = roi_align(
        l1,
        boxes=bboxes_roi,
        output_size=kernel_dim,
        spatial_scale=1.0 / reduction * 2 * 2,
        aligned=True,
    ).permute(0, 2, 3, 1).reshape(1, num_objects * kernel_dim**2, model_module.emb_dim)
    exemplars_l2 = roi_align(
        l2,
        boxes=bboxes_roi,
        output_size=kernel_dim,
        spatial_scale=1.0 / reduction * 2,
        aligned=True,
    ).permute(0, 2, 3, 1).reshape(1, num_objects * kernel_dim**2, model_module.emb_dim)

    box_hw = torch.zeros((1, num_objects, 2), dtype=torch.float32, device=device)
    box_hw[:, :, 0] = prompt_boxes_1024[:, 2] - prompt_boxes_1024[:, 0]
    box_hw[:, :, 1] = prompt_boxes_1024[:, 3] - prompt_boxes_1024[:, 1]
    shape = model_module.shape_or_objectness(box_hw).reshape(1, -1, model_module.emb_dim)

    prototype_embeddings = torch.cat([exemplars, shape], dim=1)
    prototype_embeddings_l1 = torch.cat([exemplars_l1, shape], dim=1)
    prototype_embeddings_l2 = torch.cat([exemplars_l2, shape], dim=1)

    adapted_f, _ = model_module.adapt_features(
        image_embeddings=src_target,
        image_pe=model_module.sam_prompt_encoder.get_dense_pe(),
        prototype_embeddings=prototype_embeddings,
        hq_features=target_feats["backbone_fpn"],
        hq_prototypes=[prototype_embeddings_l1, prototype_embeddings_l2],
        hq_pos=target_feats["vision_pos_enc"],
    )

    bs, c, w, h = adapted_f.shape
    adapted_f = adapted_f.view(bs, model_module.emb_dim, -1).permute(0, 2, 1)
    centerness = model_module.class_embed(adapted_f).view(bs, w, h, 1).permute(0, 3, 1, 2)
    outputs_coord = model_module.bbox_embed(adapted_f).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
    outputs, _ = boxes_with_scores(centerness, outputs_coord, sort=False, validate=True)

    masks, ious, corrected_bboxes = model_module.sam_mask(target_feats, outputs)
    del masks  # not needed for JSON export
    outputs[0]["scores"] = ious[0]
    outputs[0]["pred_boxes"] = corrected_bboxes[0].unsqueeze(0) / target_img_1024.shape[-1]

    # Keep all post-processing tensors on CPU to avoid mixed-device indexing errors.
    pred_boxes = outputs[0]["pred_boxes"][0].detach().cpu()
    box_v = outputs[0]["box_v"][0].detach().cpu()
    scores = outputs[0]["scores"][0].detach().cpu()

    if pred_boxes.numel() == 0:
        return [], []

    thr_inv = 1.0 / max(score_threshold, 1e-6)
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum().item() == 0:
        return [], []

    pred_boxes = pred_boxes[sel]
    scores = scores[sel]
    keep = nms(pred_boxes, box_v[sel], nms_iou)
    pred_boxes = pred_boxes[keep]
    scores = scores[keep]

    if min_confidence > 0:
        conf_mask = scores >= min_confidence
        pred_boxes = pred_boxes[conf_mask]
        scores = scores[conf_mask]

    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    pred_boxes = pred_boxes / target_scale * target_img_1024.shape[-1]
    return pred_boxes.cpu().tolist(), scores.cpu().tolist()


def _to_coco_annotation(
    ann_id: int,
    image_id: int,
    category_id: int,
    xyxy: List[float],
) -> Dict:
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(category_id),
        "segmentation": [],
        "area": float(w * h),
        "bbox": [float(x1), float(y1), float(w), float(h)],
        "iscrowd": 0,
        "attributes": {"occluded": False, "rotation": 0.0},
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for GECO2 inference. "
            "Please run in a CUDA-enabled environment (same one used for demo_gradio.py)."
        )

    try:
        from models.counter_infer import build_model
    except ModuleNotFoundError as e:
        if "MultiScaleDeformableAttention" in str(e):
            raise ModuleNotFoundError(
                "Missing CUDA extension 'MultiScaleDeformableAttention'. "
                "Build it once in your active env:\n"
                "  cd /opt/workspace_zeeshan/GECO2_for_Object_detection/GECO2/models/ops\n"
                "  python -m pip install -e . --no-build-isolation\n"
                "Then rerun auto_annotate_from_coco_prompt.py."
            ) from e
        raise

    candidate_weights = []
    if args.weights:
        candidate_weights.append(args.weights)
    candidate_weights.extend(
        [
            "CNTQG_multitrain_ca44.pth",
            os.path.join(args.model_path, f"{args.model_name}.pth"),
        ]
    )
    resolved_weights = None
    for w in candidate_weights:
        if w and os.path.isfile(w):
            resolved_weights = w
            break
    if resolved_weights is None:
        tried = "\n".join([f"  - {p}" for p in candidate_weights if p])
        raise FileNotFoundError(
            "Checkpoint not found. Tried:\n"
            f"{tried}\n"
            "Provide --weights /absolute/path/to/your_checkpoint.pth"
        )
    if not os.path.isfile(args.prompt_image):
        raise FileNotFoundError(f"Prompt image not found: {args.prompt_image}")
    if not os.path.isfile(args.prompt_coco_json):
        raise FileNotFoundError(f"Prompt COCO not found: {args.prompt_coco_json}")
    if not os.path.isdir(args.target_images_dir):
        raise NotADirectoryError(f"Target image folder not found: {args.target_images_dir}")

    args.zero_shot = True
    model = DataParallel(build_model(args).to(device))
    ckpt = torch.load(resolved_weights, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model_module = model.module

    prompt_boxes_by_cat, selected_categories = _load_prompt_boxes_by_category(
        prompt_coco_path=args.prompt_coco_json,
        prompt_image_path=args.prompt_image,
        prompt_image_id=args.prompt_image_id,
        use_all_categories=args.use_all_categories,
        max_exemplars=args.max_exemplars_per_category,
    )

    target_paths = _list_target_images(args.target_images_dir, args.extensions)
    if not target_paths:
        raise ValueError("No target images found with provided extensions.")

    prompt_img_norm = _load_image_tensor(Path(args.prompt_image), device)

    coco_output = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Auto-annotated by GECO2 prompt-based script",
            "url": "",
            "version": "1.0",
            "year": str(datetime.utcnow().year),
        },
        "categories": selected_categories,
        "images": [],
        "annotations": [],
    }

    next_ann_id = 1
    for image_idx, image_path in enumerate(tqdm(target_paths, desc="Annotating images"), start=1):
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size
        coco_output["images"].append(
            {
                "id": image_idx,
                "width": w,
                "height": h,
                "file_name": image_path.name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        target_img_norm = TF.normalize(
            TF.to_tensor(pil_img).to(device),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        for cat_id, prompt_boxes in prompt_boxes_by_cat.items():
            if prompt_boxes.numel() == 0:
                continue

            xyxy_boxes, _ = _run_cross_image_inference(
                model_module=model_module,
                prompt_img_norm=prompt_img_norm,
                prompt_boxes_xyxy=prompt_boxes,
                target_img_norm=target_img_norm,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
                min_confidence=args.min_confidence,
                device=device,
            )

            for box in xyxy_boxes:
                box[0] = max(0.0, min(float(box[0]), w - 1.0))
                box[1] = max(0.0, min(float(box[1]), h - 1.0))
                box[2] = max(0.0, min(float(box[2]), w - 1.0))
                box[3] = max(0.0, min(float(box[3]), h - 1.0))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                coco_output["annotations"].append(_to_coco_annotation(next_ann_id, image_idx, cat_id, box))
                next_ann_id += 1

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, separators=(",", ":"))

    print(f"Saved COCO annotations to: {output_path}")
    print(f"Images: {len(coco_output['images'])}, Annotations: {len(coco_output['annotations'])}")
    print(f"Weights used: {resolved_weights}")
    print("Output schema matches CVAT instances_default.json style.")


if __name__ == "__main__":
    main()
