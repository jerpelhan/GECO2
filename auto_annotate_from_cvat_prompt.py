import argparse
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import torch
from PIL import Image
from torch.nn import DataParallel
from torchvision.transforms import functional as TF
from tqdm import tqdm

from auto_annotate_from_coco_prompt import _run_cross_image_inference
from utils.arg_parser import get_argparser

CVAT_ACCEPT_HEADER = "application/vnd.cvat+json, application/json;q=0.9, */*;q=0.8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-annotate a CVAT task from the first frame annotated with the target class. "
            "Only adds new boxes for the selected class and keeps other classes untouched."
        ),
        parents=[get_argparser()],
    )
    parser.add_argument("--cvat-host", type=str, required=True, help="CVAT base URL, e.g. http://localhost:8080")
    parser.add_argument("--cvat-username", type=str, required=True, help="CVAT username")
    parser.add_argument("--cvat-password", type=str, required=True, help="CVAT password")
    parser.add_argument("--task-id", type=int, required=True, help="CVAT task id")
    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help="Class name that is manually annotated on the prompt frame",
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
    parser.add_argument("--score-threshold", type=float, default=0.33, help="Selection threshold in GECO2 style")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence filter")
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=20,
        help="Maximum prompt boxes to use from the prompt frame",
    )
    parser.add_argument(
        "--annotate-even-if-target-exists",
        action="store_true",
        help="Also annotate frames that already have target-class annotation (off by default)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (not recommended)",
    )
    parser.add_argument(
        "--save-debug-dir",
        type=str,
        default=None,
        help="Optional directory to save downloaded frames for debugging",
    )
    parser.add_argument(
        "--updated-annotations-json",
        type=str,
        default=None,
        help="Optional path to save a local copy of updated CVAT annotations JSON",
    )
    return parser.parse_args()


def _normalize_host(host: str) -> str:
    return host.rstrip("/")


def _resolve_weights(args: argparse.Namespace) -> str:
    candidate_weights = []
    if args.weights:
        candidate_weights.append(args.weights)
    candidate_weights.extend(
        [
            "CNTQG_multitrain_ca44.pth",
            os.path.join(args.model_path, f"{args.model_name}.pth"),
        ]
    )
    for path in candidate_weights:
        if path and os.path.isfile(path):
            return path
    tried = "\n".join([f"  - {p}" for p in candidate_weights if p])
    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n"
        f"{tried}\n"
        "Provide --weights /absolute/path/to/your_checkpoint.pth"
    )


def _build_model(args: argparse.Namespace, device: torch.device):
    try:
        from models.counter_infer import build_model
    except ModuleNotFoundError as e:
        if "MultiScaleDeformableAttention" in str(e):
            raise ModuleNotFoundError(
                "Missing CUDA extension 'MultiScaleDeformableAttention'. "
                "Build it once in your active env:\n"
                "  cd /opt/workspace_zeeshan/GECO2_for_Object_detection/GECO2/models/ops\n"
                "  python -m pip install -e . --no-build-isolation"
            ) from e
        raise

    args.zero_shot = True
    model = DataParallel(build_model(args).to(device))
    weights_path = _resolve_weights(args)
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model.module, weights_path


def _cvat_login(host: str, username: str, password: str, verify_ssl: bool) -> requests.Session:
    s = requests.Session()
    s.verify = verify_ssl
    s.headers.update({"Referer": host})

    login_url = f"{host}/api/auth/login"

    # Try JSON login first with CVAT-specific Accept negotiation.
    attempts = [
        {
            "json": {"username": username, "password": password},
            "headers": {"Accept": CVAT_ACCEPT_HEADER, "Content-Type": "application/json"},
        },
        {
            "json": {"username": username, "password": password},
            "headers": {"Accept": "*/*", "Content-Type": "application/json"},
        },
        {
            "data": {"username": username, "password": password},
            "headers": {"Accept": CVAT_ACCEPT_HEADER},
        },
    ]

    resp = None
    for attempt in attempts:
        resp = s.post(login_url, timeout=30, **attempt)
        if resp.status_code < 400:
            break
        if resp.status_code not in (400, 401, 403, 406, 415):
            break
    if resp is None or resp.status_code >= 400:
        raise RuntimeError(f"CVAT login failed ({resp.status_code if resp else 'N/A'}): {resp.text if resp else ''}")

    csrf = s.cookies.get("csrftoken")
    if csrf:
        s.headers["X-CSRFToken"] = csrf
    return s


def _get_json(session: requests.Session, url: str, timeout: int = 60) -> Dict:
    resp = session.get(url, timeout=timeout, headers={"Accept": CVAT_ACCEPT_HEADER})
    if resp.status_code >= 400:
        raise RuntimeError(f"GET {url} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _extract_name_to_id(labels_obj) -> Dict[str, int]:
    name_to_id: Dict[str, int] = {}
    for label in labels_obj or []:
        if isinstance(label, dict):
            name = str(label.get("name", "")).strip()
            lid = label.get("id")
            if name and lid is not None:
                try:
                    name_to_id[name] = int(lid)
                except (TypeError, ValueError):
                    continue
        # Some CVAT responses can include strings only (name without id).
        # We ignore those here because GECO2 upload needs a numeric label_id.
    return name_to_id


def _get_task_labels(task_json: Dict) -> Dict[str, int]:
    return _extract_name_to_id(task_json.get("labels", []))


def _get_project_labels(session: requests.Session, host: str, project_id: int) -> Dict[str, int]:
    project = _get_json(session, f"{host}/api/projects/{project_id}")
    return _extract_name_to_id(project.get("labels", []))


def _get_labels_from_api(session: requests.Session, host: str, task_id: int) -> Dict[str, int]:
    page = 1
    page_size = 100
    out: Dict[str, int] = {}
    while True:
        url = f"{host}/api/labels?task_id={task_id}&page={page}&page_size={page_size}"
        payload = _get_json(session, url)
        if isinstance(payload, dict) and "results" in payload:
            chunk = _extract_name_to_id(payload.get("results", []))
            out.update(chunk)
            next_url = payload.get("next")
            if not next_url:
                break
            page += 1
            continue
        # Non-paginated variant
        if isinstance(payload, list):
            out.update(_extract_name_to_id(payload))
        break
    return out


def _shape_to_xyxy(shape: Dict) -> Tuple[float, float, float, float]:
    points = shape.get("points", [])
    if not points or len(points) < 4:
        raise ValueError("Shape does not contain enough points.")

    stype = shape.get("type")
    if stype == "rectangle":
        x1, y1, x2, y2 = points[:4]
        return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))

    xs = points[0::2]
    ys = points[1::2]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _collect_target_boxes_by_frame(annotations: Dict, target_label_id: int) -> Dict[int, List[List[float]]]:
    out: Dict[int, List[List[float]]] = {}

    for shape in annotations.get("shapes", []):
        if int(shape.get("label_id", -1)) != target_label_id:
            continue
        frame = int(shape.get("frame", -1))
        if frame < 0:
            continue
        try:
            x1, y1, x2, y2 = _shape_to_xyxy(shape)
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        out.setdefault(frame, []).append([x1, y1, x2, y2])

    for track in annotations.get("tracks", []):
        if int(track.get("label_id", -1)) != target_label_id:
            continue
        for tshape in track.get("shapes", []):
            if bool(tshape.get("outside", False)):
                continue
            frame = int(tshape.get("frame", -1))
            if frame < 0:
                continue
            try:
                x1, y1, x2, y2 = _shape_to_xyxy(tshape)
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            out.setdefault(frame, []).append([x1, y1, x2, y2])

    return out


def _download_frame_bytes(
    session: requests.Session,
    host: str,
    task_id: int,
    frame_number: int,
    quality: str = "original",
) -> bytes:
    url = f"{host}/api/tasks/{task_id}/data?type=frame&number={frame_number}&quality={quality}"
    resp = session.get(url, timeout=120, headers={"Accept": "image/*,*/*;q=0.8"})
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to download frame {frame_number} ({resp.status_code}): {resp.text}")
    return resp.content


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_norm_tensor(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    return TF.normalize(
        TF.to_tensor(pil_img).to(device),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def _shape_payload(frame: int, label_id: int, box_xyxy: List[float]) -> Dict:
    x1, y1, x2, y2 = box_xyxy
    return {
        "type": "rectangle",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0,
        "points": [float(x1), float(y1), float(x2), float(y2)],
        "frame": int(frame),
        "label_id": int(label_id),
        "group": 0,
        "source": "auto",
        "attributes": [],
    }


def _post_new_shapes(
    session: requests.Session,
    host: str,
    task_id: int,
    shapes: List[Dict],
) -> None:
    if not shapes:
        return
    url = f"{host}/api/tasks/{task_id}/annotations?action=create"
    payload = {"shapes": shapes, "tags": [], "tracks": []}
    resp = session.patch(
        url,
        json=payload,
        timeout=300,
        headers={"Accept": CVAT_ACCEPT_HEADER, "Content-Type": "application/json"},
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Failed to upload annotations ({resp.status_code}): {resp.text}")


def main() -> None:
    args = parse_args()
    host = _normalize_host(args.cvat_host)
    verify_ssl = not bool(args.insecure)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for GECO2 inference. "
            "Please run in a CUDA-enabled environment."
        )

    model_module, resolved_weights = _build_model(args, device)
    session = _cvat_login(host, args.cvat_username, args.cvat_password, verify_ssl)

    task = _get_json(session, f"{host}/api/tasks/{args.task_id}")
    labels_by_name = _get_task_labels(task)
    if not labels_by_name and task.get("project_id") is not None:
        labels_by_name = _get_project_labels(session, host, int(task["project_id"]))
    if not labels_by_name:
        labels_by_name = _get_labels_from_api(session, host, args.task_id)
    labels_by_name_ci = {k.lower(): v for k, v in labels_by_name.items()}
    if args.object_name.lower() not in labels_by_name_ci:
        valid = ", ".join(sorted(labels_by_name.keys()))
        raise ValueError(f"Object class '{args.object_name}' not found in task labels. Available: {valid}")
    target_label_id = labels_by_name_ci[args.object_name.lower()]

    annotations = _get_json(session, f"{host}/api/tasks/{args.task_id}/annotations")
    target_boxes_by_frame = _collect_target_boxes_by_frame(annotations, target_label_id)
    if not target_boxes_by_frame:
        raise ValueError(
            f"No existing annotation found for class '{args.object_name}'. "
            "Please annotate at least one frame first in CVAT."
        )

    prompt_frame = min(target_boxes_by_frame.keys())
    prompt_boxes = target_boxes_by_frame[prompt_frame][: max(1, args.max_exemplars)]
    prompt_boxes_tensor = torch.tensor(prompt_boxes, dtype=torch.float32, device=device)

    data_meta = _get_json(session, f"{host}/api/tasks/{args.task_id}/data/meta")
    frame_total = int(data_meta.get("size", task.get("size", 0)))
    if frame_total <= 0:
        frame_total = len(data_meta.get("frames", []))
    if frame_total <= 0:
        raise RuntimeError("Could not resolve task frame count from CVAT.")

    prompt_frame_bytes = _download_frame_bytes(session, host, args.task_id, prompt_frame, quality="original")
    prompt_pil = _bytes_to_pil(prompt_frame_bytes)
    prompt_img_norm = _pil_to_norm_tensor(prompt_pil, device)

    debug_dir = None
    if args.save_debug_dir:
        debug_dir = Path(args.save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    new_shapes: List[Dict] = []
    processed = 0
    skipped_with_target = 0
    created_boxes = 0

    frame_iter = list(range(frame_total))
    for frame_idx in tqdm(frame_iter, desc="Annotating CVAT frames"):
        if frame_idx == prompt_frame:
            continue
        if (not args.annotate_even_if_target_exists) and frame_idx in target_boxes_by_frame:
            skipped_with_target += 1
            continue

        frame_bytes = _download_frame_bytes(session, host, args.task_id, frame_idx, quality="original")
        pil_img = _bytes_to_pil(frame_bytes)

        if debug_dir is not None:
            (debug_dir / f"frame_{frame_idx:06d}.jpg").write_bytes(frame_bytes)

        w, h = pil_img.size
        target_img_norm = _pil_to_norm_tensor(pil_img, device)

        xyxy_boxes, _ = _run_cross_image_inference(
            model_module=model_module,
            prompt_img_norm=prompt_img_norm,
            prompt_boxes_xyxy=prompt_boxes_tensor,
            target_img_norm=target_img_norm,
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
            min_confidence=args.min_confidence,
            device=device,
        )

        for box in xyxy_boxes:
            x1 = max(0.0, min(float(box[0]), w - 1.0))
            y1 = max(0.0, min(float(box[1]), h - 1.0))
            x2 = max(0.0, min(float(box[2]), w - 1.0))
            y2 = max(0.0, min(float(box[3]), h - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            new_shapes.append(_shape_payload(frame_idx, target_label_id, [x1, y1, x2, y2]))
            created_boxes += 1

        processed += 1

    _post_new_shapes(session, host, args.task_id, new_shapes)

    if args.updated_annotations_json:
        updated = {
            "version": annotations.get("version"),
            "tags": annotations.get("tags", []),
            "tracks": annotations.get("tracks", []),
            "shapes": [*annotations.get("shapes", []), *new_shapes],
        }
        output_path = Path(args.updated_annotations_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(updated, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")

    print("CVAT auto-annotation finished.")
    print(f"Task id: {args.task_id}")
    print(f"Prompt frame: {prompt_frame}")
    print(f"Target class: {args.object_name} (label_id={target_label_id})")
    print(f"Frames processed: {processed}/{frame_total - 1}")
    print(f"Frames skipped (already had target class): {skipped_with_target}")
    print(f"New boxes uploaded: {created_boxes}")
    print(f"Weights used: {resolved_weights}")
    if args.updated_annotations_json:
        print(f"Saved updated annotations JSON: {args.updated_annotations_json}")


if __name__ == "__main__":
    main()
