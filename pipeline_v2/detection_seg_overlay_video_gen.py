from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class VideoFrame:
    frame_index: int
    timestamp: float
    image: Any


@dataclass
class SplitVideo:
    fps: float
    width: int
    height: int
    frames: List[VideoFrame] = field(default_factory=list)

    def add_frame(self, frame: VideoFrame) -> None:
        self.frames.append(frame)


def split_video_into_frames(video_path: str) -> SplitVideo:
    """
    Mirrors Ciclopes-API/core/VideoUtil/FrameSplit.py behavior.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at path: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    split_video = SplitVideo(fps=fps, width=width, height=height)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        split_video.add_frame(
            VideoFrame(frame_index=frame_index, timestamp=timestamp_s, image=frame)
        )
        frame_index += 1

    cap.release()
    return split_video


def resolve_existing_path(path_str: str, *, base_dir: Path) -> Path:
    """
    Resolve absolute or relative path against cwd and pipeline_v2 base.
    """
    input_path = Path(path_str).expanduser()
    candidates = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append(Path.cwd() / input_path)
        candidates.append(base_dir / input_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Path does not exist: {path_str}")


def resolve_output_path(output_path_str: str, video_path: Path, *, base_dir: Path) -> Path:
    """
    Resolve output path and ensure parent directory exists.
    If output path is a directory, create: <dir>/<video_stem>_seg_overlay.mp4
    """
    output_path = Path(output_path_str).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path)
        if not output_path.exists():
            output_path = base_dir / output_path_str

    if output_path.suffix.lower() == ".mp4":
        output_file = output_path
    else:
        output_file = output_path / f"{video_path.stem}_seg_overlay.mp4"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file.resolve()


def class_color(class_id: int) -> tuple[int, int, int]:
    # BGR colors for stable per-class overlays.
    palette = {
        0: (30, 200, 255),  # ball
        1: (40, 220, 40),   # lane
        2: (255, 100, 60),  # pins
    }
    if class_id in palette:
        return palette[class_id]
    # deterministic fallback color
    return (
        int((37 * class_id + 80) % 255),
        int((17 * class_id + 130) % 255),
        int((97 * class_id + 40) % 255),
    )


def overlay_result_on_frame(
    frame_bgr: np.ndarray,
    result: Any,
    class_names: Dict[int, str],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Overlay segmentation masks + boxes + labels for one result onto one frame.
    """
    overlay = frame_bgr.copy()
    output = frame_bgr.copy()

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if boxes is None or len(boxes) == 0:
        return output

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)

    mask_arr = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_arr = masks.data.detach().cpu().numpy()

    h, w = frame_bgr.shape[:2]
    for i, cls_id in enumerate(cls_ids):
        color = class_color(int(cls_id))

        if mask_arr is not None and i < mask_arr.shape[0]:
            mask = (mask_arr[i] > 0.5).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0] = color

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, color, 2)

        x1, y1, x2, y2 = xyxy[i].tolist()
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        cls_name = class_names.get(int(cls_id), f"class_{int(cls_id)}")
        label = f"{cls_name} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 8 if y1 - 8 > th else y1 + th + 8
        cv2.rectangle(output, (x1, ty - th - 6), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0, dst=output)
    return output


def run_overlay_generation(
    *,
    model_path: Path,
    video_path: Path,
    output_path: Path,
    imgsz: int,
    batch_size: int,
    conf: float,
    iou: float,
    device: str | None,
    alpha: float,
) -> None:
    model = YOLO(str(model_path))

    names_raw = getattr(model, "names", None) or {0: "ball", 1: "lane", 2: "pins"}
    class_names = {int(k): str(v) for k, v in names_raw.items()}

    split_video = split_video_into_frames(str(video_path))
    if not split_video.frames:
        raise RuntimeError("No frames found in video.")

    fps = split_video.fps if split_video.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (split_video.width, split_video.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        frames = split_video.frames
        total = len(frames)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_frames = frames[start:end]

            batch_bgr = [vf.image for vf in batch_frames]
            batch_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in batch_bgr]

            results = model.predict(
                source=batch_rgb,
                verbose=False,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                retina_masks=True,
            )

            for frame_bgr, result in zip(batch_bgr, results):
                overlay = overlay_result_on_frame(
                    frame_bgr=frame_bgr,
                    result=result,
                    class_names=class_names,
                    alpha=alpha,
                )
                writer.write(overlay)

            print(f"Processed frames {start}..{end - 1} / {total - 1}")
    finally:
        writer.release()

    print(f"Overlay video saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate segmentation overlay video from YOLO segmentation model."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO .pt model")
    parser.add_argument("--video_path", type=str, required=True, help="Input .mp4 video path")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .mp4 path or directory",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--batch_size", type=int, default=16, help="Predict batch size")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for Ultralytics, e.g. '0', 'cpu', 'cuda'",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = resolve_existing_path(args.model_path, base_dir=project_root)
    video_path = resolve_existing_path(args.video_path, base_dir=project_root)
    output_path = resolve_output_path(args.output_path, video_path, base_dir=project_root)

    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected .mp4 input video, got: {video_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    run_overlay_generation(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
        conf=float(args.conf),
        iou=float(args.iou),
        device=args.device,
        alpha=float(args.alpha),
    )


if __name__ == "__main__":
    main()
