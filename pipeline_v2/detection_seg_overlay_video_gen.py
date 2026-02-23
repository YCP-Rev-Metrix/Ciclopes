from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
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


@dataclass
class TrapezoidCandidate:
    polygon: np.ndarray
    coverage: float
    purity: float
    score: float
    y_top: int
    y_bottom: int


def _transcode_to_mp4(video_path: str) -> str:
    """
    Transcode a video to a temporary mp4 with pixel format that OpenCV can read.
    Returns the path to the temp file. Caller is responsible for cleanup.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to handle this video format."
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-an", tmp.name,
    ]
    print(f"Transcoding video for OpenCV compatibility: {Path(video_path).name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg transcode failed:\n{result.stderr[-500:]}")
    return tmp.name


def _read_frames(video_path: str) -> SplitVideo:
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


def split_video_into_frames(video_path: str) -> SplitVideo:
    """
    Read video frames. If OpenCV can't decode the pixel format (e.g. YUV411P
    interlaced AVI), falls back to ffmpeg transcoding to a temp mp4 first.
    """
    split_video = _read_frames(video_path)

    # Detect all-black frames from failed color conversion.
    if split_video.frames:
        check_count = min(5, len(split_video.frames))
        all_black = all(
            np.max(split_video.frames[i].image) == 0
            for i in range(check_count)
        )
        if all_black:
            print("Detected all-black frames (likely pixel format issue), transcoding...")
            tmp_path = _transcode_to_mp4(video_path)
            try:
                split_video = _read_frames(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

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

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    if output_path.suffix.lower() in video_suffixes:
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


def clean_mask(mask: np.ndarray, *, close_size: int = 15, open_size: int = 7) -> np.ndarray:
    """
    Aggressively clean a bubbly segmentation mask.
    Large close fills holes/gaps, then open removes small spurs.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)
    return cleaned


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest contour by area from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def approx_to_quad(contour: np.ndarray) -> np.ndarray | None:
    """
    Adaptive approxPolyDP: binary-search epsilon until we get exactly 4 vertices.
    Works on the convex hull of the contour to ignore concavities from bubbly masks.
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 20:
        return None

    # Binary search for the epsilon that gives 4 points.
    lo, hi = 0.005, 0.15
    best_approx: np.ndarray | None = None
    best_diff = 999

    for _ in range(40):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(hull, mid * peri, True)
        n = len(approx)
        diff = abs(n - 4)
        if diff < best_diff or (diff == best_diff and n >= 4):
            best_diff = diff
            best_approx = approx
        if n > 4:
            lo = mid
        elif n < 4:
            hi = mid
        else:
            return approx.reshape(-1, 2).astype(np.int32)

    # Accept 4-6 point results and reduce to 4 by dropping shortest edges.
    if best_approx is not None and 4 <= len(best_approx) <= 6:
        pts = best_approx.reshape(-1, 2).astype(np.float32)
        while len(pts) > 4:
            # Remove the vertex whose removal changes area least.
            min_loss = float("inf")
            drop_idx = 0
            for i in range(len(pts)):
                tri = np.array([
                    pts[(i - 1) % len(pts)],
                    pts[i],
                    pts[(i + 1) % len(pts)],
                ])
                area = 0.5 * abs(float(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                if area < min_loss:
                    min_loss = area
                    drop_idx = i
            pts = np.delete(pts, drop_idx, axis=0)
        return pts.astype(np.int32)

    return None


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    This is the standard ordering for homography source points.
    """
    pts = pts.astype(np.float32)
    # Sort by y first to split top vs bottom.
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    # Within top, left has smaller x.
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def hough_quad_fallback(
    mask: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray | None:
    """
    Fallback: draw the convex hull edges, run HoughLines, take the 4 strongest
    non-duplicate lines, and intersect them for corners.
    """
    hull = cv2.convexHull(contour)
    h, w = mask.shape

    # Draw hull edges as a thin edge image for Hough.
    edge_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_img, [hull], 0, 255, 2)

    min_dim = min(h, w)
    lines = cv2.HoughLines(edge_img, rho=1, theta=np.pi / 180, threshold=max(30, min_dim // 8))
    if lines is None or len(lines) < 4:
        return None

    # De-duplicate lines by (rho, theta) proximity.
    unique: List[tuple[float, float]] = []
    for line in lines:
        rho, theta = float(line[0][0]), float(line[0][1])
        is_dup = False
        for ur, ut in unique:
            if abs(rho - ur) < 20 and abs(theta - ut) < np.pi / 18:
                is_dup = True
                break
        if not is_dup:
            unique.append((rho, theta))
        if len(unique) == 4:
            break

    if len(unique) < 4:
        return None

    # Intersect all pairs, keep points inside the image.
    def line_intersect(r1: float, t1: float, r2: float, t2: float) -> tuple[float, float] | None:
        det = np.cos(t1) * np.sin(t2) - np.cos(t2) * np.sin(t1)
        if abs(det) < 1e-6:
            return None
        x = (r1 * np.sin(t2) - r2 * np.sin(t1)) / det
        y = (r2 * np.cos(t1) - r1 * np.cos(t2)) / det
        return (x, y)

    corners: List[tuple[float, float]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            pt = line_intersect(unique[i][0], unique[i][1], unique[j][0], unique[j][1])
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                corners.append(pt)

    if len(corners) < 4:
        return None

    # Pick the 4 corners closest to the convex hull centroid spread.
    corners_np = np.array(corners, dtype=np.float32)
    M = cv2.moments(hull)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    dists = np.sqrt((corners_np[:, 0] - cx) ** 2 + (corners_np[:, 1] - cy) ** 2)
    # Take 4 most spread-out corners by sorting by angle from centroid.
    angles = np.arctan2(corners_np[:, 1] - cy, corners_np[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    if len(sorted_idx) >= 4:
        # Take 4 roughly evenly spaced by angle.
        step = len(sorted_idx) / 4.0
        picks = [sorted_idx[int(i * step)] for i in range(4)]
        return corners_np[picks].astype(np.int32)

    return None


def evaluate_trapezoid(
    lane_mask: np.ndarray,
    polygon: np.ndarray,
    *,
    score_coverage_weight: float = 0.65,
) -> tuple[float, float, float]:
    lane_area = float(np.count_nonzero(lane_mask))
    if lane_area <= 0:
        return 0.0, 0.0, 0.0

    trap_mask = np.zeros(lane_mask.shape, dtype=np.uint8)
    cv2.fillPoly(trap_mask, [polygon.astype(np.int32)], 1)
    trap_area = float(np.count_nonzero(trap_mask))
    if trap_area <= 0:
        return 0.0, 0.0, 0.0

    intersection = float(np.count_nonzero((trap_mask > 0) & (lane_mask > 0)))
    coverage = intersection / lane_area
    purity = intersection / trap_area
    score = score_coverage_weight * coverage + (1.0 - score_coverage_weight) * purity
    return coverage, purity, score


def find_nearest_pins_box(
    lane_mask: np.ndarray,
    pins_boxes: List[np.ndarray],
) -> np.ndarray | None:
    """
    Find the pins bbox whose bottom-center is closest to (or overlapping with)
    the top region of the lane mask. Pins mark where the lane ends (far end).
    """
    if not pins_boxes:
        return None

    # Get the centroid x of the lane mask.
    ys, xs = np.where(lane_mask > 0)
    if len(xs) == 0:
        return None
    lane_cx = float(np.mean(xs))

    best_box = None
    best_dist = float("inf")
    for box in pins_boxes:
        # box is [x1, y1, x2, y2]
        pins_cx = (box[0] + box[2]) / 2.0
        pins_bot_y = box[3]
        dist = abs(pins_cx - lane_cx)
        if dist < best_dist:
            best_dist = dist
            best_box = box

    return best_box


def build_lane_trapezoid(
    lane_mask: np.ndarray,
    pins_boxes: List[np.ndarray] | None = None,
) -> TrapezoidCandidate | None:
    """
    Extract a clean 4-point trapezoid from a single lane mask.
    Strategy: clean mask → convex hull → approxPolyDP to 4 pts → fallback to Hough.
    If pins_boxes provided, use the nearest pins detection to anchor the top edge
    (the far/narrow end of the lane).
    """
    cleaned = clean_mask(lane_mask)

    contour = largest_contour(cleaned)
    if contour is None or cv2.contourArea(contour) < 500:
        return None

    # Primary: adaptive approxPolyDP on convex hull.
    quad = approx_to_quad(contour)

    # Fallback: Hough lines on hull edges.
    if quad is None:
        quad = hough_quad_fallback(cleaned, contour)

    if quad is None:
        return None

    quad = order_quad_points(quad)

    # If we have a nearby pins box, snap the top edge y to the pins bottom.
    # The pins sit right at the end of the lane, so their bottom y is the lane's
    # far boundary — use it to anchor the top two points of the trapezoid.
    if pins_boxes:
        pins_box = find_nearest_pins_box(lane_mask, pins_boxes)
        if pins_box is not None:
            pins_bot_y = int(pins_box[3])
            # Only adjust if the pins are near the top of the current quad.
            current_top_y = int(min(quad[0, 1], quad[1, 1]))
            if abs(pins_bot_y - current_top_y) < lane_mask.shape[0] * 0.3:
                quad[0, 1] = pins_bot_y
                quad[1, 1] = pins_bot_y

    # Validate: bottom should be wider than top (perspective trapezoid).
    width_top = abs(int(quad[1, 0]) - int(quad[0, 0]))
    width_bottom = abs(int(quad[2, 0]) - int(quad[3, 0]))
    if width_top < 4 or width_bottom < 8:
        return None

    coverage, purity, score = evaluate_trapezoid(lane_mask, quad)
    y_top = int(min(quad[0, 1], quad[1, 1]))
    y_bottom = int(max(quad[2, 1], quad[3, 1]))

    return TrapezoidCandidate(
        polygon=quad,
        coverage=coverage,
        purity=purity,
        score=score,
        y_top=y_top,
        y_bottom=y_bottom,
    )


def extract_lane_and_pins_data(
    result: Any,
    lane_class_id: int,
    pins_class_id: int,
    frame_shape_hw: tuple[int, int],
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract all lane instance masks and all pins bounding boxes from a result.
    Returns (lane_masks, pins_boxes_xyxy).
    """
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return [], []

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    mask_arr = masks.data.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)
    h, w = frame_shape_hw

    lane_masks: List[np.ndarray] = []
    pins_boxes: List[np.ndarray] = []

    for i, cls_id in enumerate(cls_ids):
        cid = int(cls_id)
        if cid == lane_class_id and i < mask_arr.shape[0]:
            mask = (mask_arr[i] > 0.5).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if np.count_nonzero(mask) > 200:
                lane_masks.append(mask)
        elif cid == pins_class_id:
            pins_boxes.append(xyxy[i])

    return lane_masks, pins_boxes


TRAPEZOID_COLORS = [
    (0, 255, 255),   # cyan
    (255, 0, 255),   # magenta
    (0, 165, 255),   # orange
    (255, 255, 0),   # yellow-ish
    (0, 255, 128),   # spring green
]


def draw_trapezoid_debug(
    image: np.ndarray,
    trapezoid: TrapezoidCandidate,
    *,
    lane_index: int = 0,
    color: tuple[int, int, int] | None = None,
) -> None:
    if color is None:
        color = TRAPEZOID_COLORS[lane_index % len(TRAPEZOID_COLORS)]

    poly = trapezoid.polygon.reshape((-1, 1, 2))

    # Semi-transparent fill so you can see the trapezoid region clearly.
    fill_overlay = image.copy()
    cv2.fillPoly(fill_overlay, [trapezoid.polygon.reshape((-1, 1, 2))], color)
    cv2.addWeighted(fill_overlay, 0.2, image, 0.8, 0, dst=image)

    # Thick outline.
    cv2.polylines(image, [poly], True, color, 3, cv2.LINE_AA)

    # Draw corner points as circles for easy inspection.
    for pt_idx in range(4):
        pt = tuple(trapezoid.polygon[pt_idx].tolist())
        cv2.circle(image, pt, 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)
        # Label corners: TL=0, TR=1, BR=2, BL=3
        corner_labels = ["TL", "TR", "BR", "BL"]
        cv2.putText(
            image, corner_labels[pt_idx],
            (pt[0] + 8, pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    tl = tuple(trapezoid.polygon[0].tolist())
    text = (
        f"lane{lane_index} s={trapezoid.score:.2f} "
        f"c={trapezoid.coverage:.2f} p={trapezoid.purity:.2f}"
    )
    cv2.putText(
        image,
        text,
        (int(tl[0]), max(24, int(tl[1]) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def overlay_result_on_frame(
    frame_bgr: np.ndarray,
    result: Any,
    class_names: Dict[int, str],
    *,
    alpha: float = 0.35,
    lane_class_id: int = 1,
    enable_guided_trapezoid: bool = True,
    min_trapezoid_score: float = 0.20,
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

    if enable_guided_trapezoid:
        pins_class_id = 2
        lane_masks, pins_boxes = extract_lane_and_pins_data(
            result,
            lane_class_id=lane_class_id,
            pins_class_id=pins_class_id,
            frame_shape_hw=(h, w),
        )
        for lane_idx, lmask in enumerate(lane_masks):
            trapezoid = build_lane_trapezoid(lmask, pins_boxes=pins_boxes)
            if trapezoid is not None and trapezoid.score >= min_trapezoid_score:
                draw_trapezoid_debug(output, trapezoid, lane_index=lane_idx)

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
    lane_class_id: int,
    enable_guided_trapezoid: bool,
    min_trapezoid_score: float,
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
                    lane_class_id=lane_class_id,
                    enable_guided_trapezoid=enable_guided_trapezoid,
                    min_trapezoid_score=min_trapezoid_score,
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
    parser.add_argument("--video_path", type=str, required=True, help="Input video path (mp4, avi, mov, mkv, wmv)")
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
    parser.add_argument(
        "--lane_class_id",
        type=int,
        default=1,
        help="Class id to use as lane segmentation guidance",
    )
    parser.add_argument(
        "--disable_guided_trapezoid",
        action="store_true",
        help="Disable lane-guided trapezoid prototype overlay",
    )
    parser.add_argument(
        "--min_trapezoid_score",
        type=float,
        default=0.20,
        help="Minimum combined coverage/purity score to draw trapezoid",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    model_path = resolve_existing_path(args.model_path, base_dir=project_root)
    video_path = resolve_existing_path(args.video_path, base_dir=project_root)
    output_path = resolve_output_path(args.output_path, video_path, base_dir=project_root)

    if video_path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv", ".wmv"):
        raise ValueError(f"Unsupported video format: {video_path.suffix}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")
    if not (0.0 <= args.min_trapezoid_score <= 1.0):
        raise ValueError("--min_trapezoid_score must be in [0, 1]")

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
        lane_class_id=int(args.lane_class_id),
        enable_guided_trapezoid=not bool(args.disable_guided_trapezoid),
        min_trapezoid_score=float(args.min_trapezoid_score),
    )


if __name__ == "__main__":
    main()
