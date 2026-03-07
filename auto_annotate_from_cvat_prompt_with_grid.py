import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm

from auto_annotate_from_coco_prompt import _run_cross_image_inference
from auto_annotate_from_cvat_prompt import (
    _build_model,
    _bytes_to_pil,
    _collect_target_boxes_by_frame,
    _cvat_login,
    _download_frame_bytes,
    _get_json,
    _get_labels_from_api,
    _get_project_labels,
    _get_task_labels,
    _normalize_host,
    _pil_to_norm_tensor,
    _post_new_shapes,
    _shape_payload,
)
from utils.arg_parser import get_argparser


@dataclass(frozen=True)
class Tile:
    tid: int
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-annotate CVAT task using prompt frame + optional image grid. "
            "If --grid is set (e.g. 2x3), prompt and target frames are tiled."
        ),
        parents=[get_argparser()],
    )
    parser.add_argument("--cvat-host", type=str, required=True, help="CVAT base URL, e.g. https://app.cvat.ai")
    parser.add_argument("--cvat-username", type=str, required=True, help="CVAT username")
    parser.add_argument("--cvat-password", type=str, required=True, help="CVAT password")
    parser.add_argument("--task-id", type=int, required=True, help="CVAT task id")
    parser.add_argument("--object-name", type=str, required=True, help="Target class name already annotated")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help=(
            "Model checkpoint path. If omitted, script tries "
            "./CNTQG_multitrain_ca44.pth and <model_path>/<model_name>.pth."
        ),
    )
    parser.add_argument("--grid", type=str, default="1x1", help="Grid as ROWSxCOLS, e.g. 2x2, 3x4")
    parser.add_argument("--score-threshold", type=float, default=0.33, help="GECO2 score threshold")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence filter")
    parser.add_argument("--max-exemplars", type=int, default=20, help="Max prompt boxes from prompt frame")
    parser.add_argument(
        "--annotate-even-if-target-exists",
        action="store_true",
        help="Also annotate frames that already have target-class annotation",
    )
    parser.add_argument(
        "--include-prompt-frame",
        action="store_true",
        help="Also run inference on the prompt frame (useful for single-image tasks)",
    )
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification (not recommended)")
    parser.add_argument("--save-debug-dir", type=str, default=None, help="Optional directory to save frame images")
    parser.add_argument(
        "--enforce-color-0-255",
        action="store_true",
        help="Clamp image RGB values strictly to uint8 range [0, 255] before inference",
    )
    parser.add_argument(
        "--prompt-debug-image-path",
        type=str,
        default=None,
        help="Optional output path for saving the full prompt image after filtering",
    )
    return parser.parse_args()


def _parse_grid(grid: str) -> Tuple[int, int]:
    text = grid.strip().lower().replace(" ", "")
    if "x" not in text:
        raise ValueError("--grid must be in ROWSxCOLS format, e.g. 2x3")
    parts = text.split("x")
    if len(parts) != 2:
        raise ValueError("--grid must be in ROWSxCOLS format, e.g. 2x3")
    rows = int(parts[0])
    cols = int(parts[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("--grid rows and cols must be > 0")
    return rows, cols


def _grid_edges(length: int, segments: int) -> List[int]:
    return [int(round(i * length / segments)) for i in range(segments + 1)]


def _build_tiles(width: int, height: int, rows: int, cols: int) -> Tuple[List[Tile], List[int], List[int]]:
    x_edges = _grid_edges(width, cols)
    y_edges = _grid_edges(height, rows)
    tiles: List[Tile] = []
    tid = 0
    for r in range(rows):
        for c in range(cols):
            x1, x2 = x_edges[c], x_edges[c + 1]
            y1, y2 = y_edges[r], y_edges[r + 1]
            tiles.append(Tile(tid=tid, row=r, col=c, x1=x1, y1=y1, x2=x2, y2=y2))
            tid += 1
    return tiles, x_edges, y_edges


def _find_segment(edges: List[int], v: float) -> int:
    if v <= edges[0]:
        return 0
    if v >= edges[-1]:
        return len(edges) - 2
    lo, hi = 0, len(edges) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if edges[mid] <= v < edges[mid + 1]:
            return mid
        if v < edges[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return len(edges) - 2


def _clip_to_tile(box: List[float], tile: Tile) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(x1 - tile.x1), float(tile.x2 - tile.x1))),
        max(0.0, min(float(y1 - tile.y1), float(tile.y2 - tile.y1))),
        max(0.0, min(float(x2 - tile.x1), float(tile.x2 - tile.x1))),
        max(0.0, min(float(y2 - tile.y1), float(tile.y2 - tile.y1))),
    ]


def _assign_prompt_boxes_to_tiles(
    prompt_boxes_xyxy: List[List[float]],
    tiles: List[Tile],
    x_edges: List[int],
    y_edges: List[int],
) -> Dict[int, List[List[float]]]:
    tile_by_rc = {(t.row, t.col): t for t in tiles}
    out: Dict[int, List[List[float]]] = {}
    for box in prompt_boxes_xyxy:
        cx = (float(box[0]) + float(box[2])) * 0.5
        cy = (float(box[1]) + float(box[3])) * 0.5
        col = _find_segment(x_edges, cx)
        row = _find_segment(y_edges, cy)
        tile = tile_by_rc[(row, col)]
        local = _clip_to_tile(box, tile)
        if local[2] <= local[0] or local[3] <= local[1]:
            continue
        out.setdefault(tile.tid, []).append(local)
    return out


def _global_nms(boxes_xyxy: List[List[float]], scores: List[float], iou: float) -> Tuple[List[List[float]], List[float]]:
    if not boxes_xyxy:
        return [], []
    t_boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
    t_scores = torch.tensor(scores if scores else [1.0] * len(boxes_xyxy), dtype=torch.float32)
    keep = nms(t_boxes, t_scores, iou)
    k = keep.tolist()
    return [boxes_xyxy[i] for i in k], [float(t_scores[i]) for i in k]


def _filter_color_0_255(pil_img):
    """
    Apply a binary threshold per RGB channel:
    - value > 220 -> 255
    - value <= 220 -> 0
    """
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    arr = np.where(arr > 220, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    args = parse_args()
    rows, cols = _parse_grid(args.grid)

    host = _normalize_host(args.cvat_host)
    verify_ssl = not bool(args.insecure)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for GECO2 inference.")

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
            f"No annotation found for class '{args.object_name}'. "
            "Annotate at least one frame first in CVAT."
        )

    prompt_frame = min(target_boxes_by_frame.keys())
    prompt_boxes_global = target_boxes_by_frame[prompt_frame][: max(1, args.max_exemplars)]

    prompt_bytes = _download_frame_bytes(session, host, args.task_id, prompt_frame, quality="original")
    prompt_pil = _bytes_to_pil(prompt_bytes)
    saved_prompt_debug_path = None
    if args.enforce_color_0_255:
        prompt_pil = _filter_color_0_255(prompt_pil)
        # Save full prompt image after filtering for debugging.
        if args.prompt_debug_image_path:
            debug_prompt_path = Path(args.prompt_debug_image_path)
        elif args.save_debug_dir:
            debug_prompt_path = Path(args.save_debug_dir) / f"prompt_frame_{prompt_frame:06d}_filtered.jpg"
        else:
            debug_prompt_path = Path.cwd() / f"prompt_frame_{prompt_frame:06d}_filtered.jpg"
        debug_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_pil.save(debug_prompt_path)
        saved_prompt_debug_path = str(debug_prompt_path)
    p_w, p_h = prompt_pil.size
    prompt_tiles, p_x_edges, p_y_edges = _build_tiles(p_w, p_h, rows, cols)
    prompt_boxes_by_tile = _assign_prompt_boxes_to_tiles(prompt_boxes_global, prompt_tiles, p_x_edges, p_y_edges)
    if not prompt_boxes_by_tile:
        raise ValueError("Could not map prompt boxes to tiles. Try --grid 1x1 or check prompt annotations.")

    # Build reusable normalized prompt tile tensors.
    prompt_tile_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for tile in prompt_tiles:
        boxes = prompt_boxes_by_tile.get(tile.tid, [])
        if not boxes:
            continue
        crop = prompt_pil.crop((tile.x1, tile.y1, tile.x2, tile.y2))
        if crop.size[0] <= 1 or crop.size[1] <= 1:
            continue
        prompt_img_norm = _pil_to_norm_tensor(crop, device)
        prompt_boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        prompt_tile_data[tile.tid] = (prompt_img_norm, prompt_boxes_tensor)

    if not prompt_tile_data:
        raise ValueError("No valid prompt tiles remained after filtering.")

    data_meta = _get_json(session, f"{host}/api/tasks/{args.task_id}/data/meta")
    frame_total = int(data_meta.get("size", task.get("size", 0)))
    if frame_total <= 0:
        frame_total = len(data_meta.get("frames", []))
    if frame_total <= 0:
        raise RuntimeError("Could not resolve task frame count from CVAT.")

    debug_dir = None
    if args.save_debug_dir:
        debug_dir = Path(args.save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    new_shapes: List[Dict] = []
    processed = 0
    skipped_with_target = 0
    created_boxes = 0

    candidate_frames: List[int] = []
    for frame_idx in range(frame_total):
        if (not args.include_prompt_frame) and frame_idx == prompt_frame:
            continue
        candidate_frames.append(frame_idx)

    if not candidate_frames:
        raise ValueError(
            "No frames selected for inference. "
            "If this is a single-image task, pass --include-prompt-frame."
        )

    for frame_idx in tqdm(candidate_frames, desc="Annotating CVAT frames (grid)"):
        if (not args.annotate_even_if_target_exists) and frame_idx in target_boxes_by_frame:
            skipped_with_target += 1
            continue

        frame_bytes = _download_frame_bytes(session, host, args.task_id, frame_idx, quality="original")
        frame_pil = _bytes_to_pil(frame_bytes)
        if args.enforce_color_0_255:
            frame_pil = _filter_color_0_255(frame_pil)
        if debug_dir is not None:
            (debug_dir / f"frame_{frame_idx:06d}.jpg").write_bytes(frame_bytes)
        f_w, f_h = frame_pil.size
        target_tiles, _, _ = _build_tiles(f_w, f_h, rows, cols)
        target_tile_by_tid = {t.tid: t for t in target_tiles}

        frame_boxes_global: List[List[float]] = []
        frame_scores: List[float] = []

        # Run each prompt tile against each target tile.
        for _, (prompt_img_norm, prompt_boxes_tensor) in prompt_tile_data.items():
            for target_tile in target_tiles:
                crop = frame_pil.crop((target_tile.x1, target_tile.y1, target_tile.x2, target_tile.y2))
                if crop.size[0] <= 1 or crop.size[1] <= 1:
                    continue
                target_img_norm = _pil_to_norm_tensor(crop, device)
                local_boxes, local_scores = _run_cross_image_inference(
                    model_module=model_module,
                    prompt_img_norm=prompt_img_norm,
                    prompt_boxes_xyxy=prompt_boxes_tensor,
                    target_img_norm=target_img_norm,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                    min_confidence=args.min_confidence,
                    device=device,
                )

                for i, box in enumerate(local_boxes):
                    tx = target_tile_by_tid[target_tile.tid].x1
                    ty = target_tile_by_tid[target_tile.tid].y1
                    x1 = float(box[0]) + tx
                    y1 = float(box[1]) + ty
                    x2 = float(box[2]) + tx
                    y2 = float(box[3]) + ty
                    x1 = max(0.0, min(x1, f_w - 1.0))
                    y1 = max(0.0, min(y1, f_h - 1.0))
                    x2 = max(0.0, min(x2, f_w - 1.0))
                    y2 = max(0.0, min(y2, f_h - 1.0))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    frame_boxes_global.append([x1, y1, x2, y2])
                    score = float(local_scores[i]) if i < len(local_scores) else 1.0
                    frame_scores.append(score)

        frame_boxes_global, frame_scores = _global_nms(frame_boxes_global, frame_scores, args.nms_iou)
        for box in frame_boxes_global:
            new_shapes.append(_shape_payload(frame_idx, target_label_id, box))
            created_boxes += 1

        processed += 1

    _post_new_shapes(session, host, args.task_id, new_shapes)

    print("CVAT grid auto-annotation finished.")
    print(f"Task id: {args.task_id}")
    print(f"Prompt frame: {prompt_frame}")
    print(f"Target class: {args.object_name} (label_id={target_label_id})")
    print(f"Grid: {rows}x{cols}")
    print(f"Prompt tiles used: {len(prompt_tile_data)}")
    print(f"Frames selected: {len(candidate_frames)}")
    print(f"Frames processed: {processed}/{len(candidate_frames)}")
    print(f"Frames skipped (already had target class): {skipped_with_target}")
    print(f"New boxes uploaded: {created_boxes}")
    print(f"Weights used: {resolved_weights}")
    if saved_prompt_debug_path:
        print(f"Saved filtered prompt image: {saved_prompt_debug_path}")


if __name__ == "__main__":
    main()
