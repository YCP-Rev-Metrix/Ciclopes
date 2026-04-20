from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

LANE_LENGTH_M = 18.288
LANE_WIDTH_M = 1.0541
DEFAULT_MIN_TRAPEZOID_SCORE = 0.20

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_CHECKPOINT = (
    PIPELINE_ROOT
    / "runs"
    / "segment"
    / "outputs"
    / "yolo26n-seg-BLP"
    / "yolo26n-seg"
    / "weights"
    / "best.pt"
)


@dataclass(frozen=True)
class BallPos:
    frame_index: int
    timestamp_s: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class FrameSegmentation:
    ball_masks: List[np.ndarray]
    lane_masks: List[np.ndarray]
    pins_masks: List[np.ndarray]
    frame_shape: tuple[int, int]


@dataclass(frozen=True)
class HomographySelection:
    frame_index: int
    homography: np.ndarray
    src_corners: np.ndarray
    dst_corners: np.ndarray
    is_trapezoid: bool
    selected_lane_contours: int


@dataclass(frozen=True)
class PostprocessHealth:
    frames_scanned_for_h: int
    frames_with_lane: int
    frames_with_ball: int
    lane_polygon_count_at_h: int
    homography_determinant: float
    homography_condition_number: float
    mean_lane_coverage_ratio: float


@dataclass(frozen=True)
class PostprocessResult:
    ball_positions: List[BallPos]
    homography_selection: HomographySelection
    health: PostprocessHealth


@dataclass
class VideoFrame:
    frame_index: int
    timestamp_s: float
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


@dataclass
class TrackedLane:
    quad: np.ndarray
    centroid_x: float
    last_seen_frame: int
    best_score: float
    best_quad: np.ndarray
    best_frame_idx: int
    ball_votes: int = 0
    total_mask_area: float = 0.0
    seen_count: int = 0


@dataclass(frozen=True)
class GeometryTuning:
    pins_y_tolerance_ratio: float = 0.30
    pins_lane_width_scale: float = 1.70
    pins_x_max_adjust_px: float = 50.0
    side_refine_strip_width: float = 25.0
    post_pins_y_weight: float = 0.92
    top_edge_level_weight: float = 0.82
    top_edge_max_adjust_px: float = 42.0
    higher_corner_level_boost: float = 1.35
    residual_top_asymmetry_ratio: float = 0.22
    pins_top_x_blend: float = 0.35
    max_top_width_ratio: float = 0.98
    top_width_target_weight: float = 0.72
    top_center_pin_blend: float = 0.55
    left_top_geometry_pull: float = 0.82
    right_top_geometry_pull: float = 0.12
    target_taper_ratio: float = 0.45
    bottom_width_target_weight: float = 0.68


class TemporalSmoother:
    def __init__(
        self,
        ema_alpha: float = 0.3,
        max_match_dist: float = 80.0,
        max_corner_jump: float = 60.0,
        stale_frames: int = 30,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.max_match_dist = max_match_dist
        self.max_corner_jump = max_corner_jump
        self.stale_frames = stale_frames
        self.tracks: List[TrackedLane] = []

    def update(
        self,
        candidates: List[TrapezoidCandidate],
        frame_idx: int,
        ball_boxes: List[np.ndarray] | None = None,
        lane_mask_areas: List[float] | None = None,
    ) -> None:
        used_tracks: set[int] = set()
        cand_to_track: List[int] = []

        for ci, cand in enumerate(candidates):
            cx = float(np.mean(cand.polygon[:, 0]))

            best_ti: int | None = None
            best_dist = float("inf")
            for ti, track in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                dist = abs(cx - track.centroid_x)
                if dist < best_dist and dist < self.max_match_dist:
                    best_dist = dist
                    best_ti = ti

            if best_ti is not None:
                track = self.tracks[best_ti]
                used_tracks.add(best_ti)

                corner_dists = np.sqrt(
                    np.sum(
                        (cand.polygon.astype(float) - track.quad.astype(float)) ** 2,
                        axis=1,
                    )
                )
                max_jump = float(np.max(corner_dists))

                if max_jump < self.max_corner_jump:
                    a = self.ema_alpha
                    blended = (
                        a * cand.polygon.astype(float)
                        + (1.0 - a) * track.quad.astype(float)
                    )
                    track.quad = blended.astype(np.int32)

                track.centroid_x = float(np.mean(track.quad[:, 0]))
                track.last_seen_frame = frame_idx
                track.seen_count += 1
                if lane_mask_areas is not None and ci < len(lane_mask_areas):
                    track.total_mask_area += float(lane_mask_areas[ci])
                if cand.score > track.best_score:
                    track.best_score = float(cand.score)
                    track.best_quad = cand.polygon.copy()
                    track.best_frame_idx = int(frame_idx)
                cand_to_track.append(best_ti)
            else:
                area = 0.0
                if lane_mask_areas is not None and ci < len(lane_mask_areas):
                    area = float(lane_mask_areas[ci])
                new_idx = len(self.tracks)
                self.tracks.append(
                    TrackedLane(
                        quad=cand.polygon.copy(),
                        centroid_x=cx,
                        last_seen_frame=frame_idx,
                        best_score=float(cand.score),
                        best_quad=cand.polygon.copy(),
                        best_frame_idx=int(frame_idx),
                        total_mask_area=area,
                        seen_count=1,
                    )
                )
                cand_to_track.append(new_idx)

        if ball_boxes:
            for ball_box in ball_boxes:
                ball_cx = float(ball_box[0] + ball_box[2]) / 2.0
                ball_cy = float(ball_box[1] + ball_box[3]) / 2.0
                for ci, cand in enumerate(candidates):
                    if ci >= len(cand_to_track):
                        continue
                    ti = cand_to_track[ci]
                    if ti >= len(self.tracks):
                        continue
                    inside = cv2.pointPolygonTest(
                        cand.polygon.reshape((-1, 1, 2)).astype(np.float32),
                        (ball_cx, ball_cy),
                        measureDist=False,
                    )
                    if inside >= 0:
                        self.tracks[ti].ball_votes += 1
                        break

        self.tracks = [
            t for t in self.tracks if frame_idx - t.last_seen_frame < self.stale_frames
        ]

    def select_active_lane(self) -> TrackedLane | None:
        if not self.tracks:
            return None

        with_ball_votes = [track for track in self.tracks if track.ball_votes > 0]
        if with_ball_votes:
            return max(with_ball_votes, key=lambda track: track.ball_votes)

        return max(
            self.tracks,
            key=lambda track: track.total_mask_area / max(track.seen_count, 1),
        )


def extract_frame_segmentation(result: Any) -> FrameSegmentation:
    orig_h, orig_w = getattr(result, "orig_shape", (0, 0))
    ball_masks: List[np.ndarray] = []
    lane_masks: List[np.ndarray] = []
    pins_masks: List[np.ndarray] = []

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or len(boxes) == 0:
        return FrameSegmentation(ball_masks, lane_masks, pins_masks, (int(orig_h), int(orig_w)))

    cls_tensor = getattr(boxes, "cls", None)
    data_tensor = getattr(masks, "data", None)
    if cls_tensor is None or data_tensor is None:
        return FrameSegmentation(ball_masks, lane_masks, pins_masks, (int(orig_h), int(orig_w)))

    mask_arr = data_tensor.detach().cpu().numpy()
    cls_ids = cls_tensor.detach().cpu().numpy().astype(int)

    for i in range(min(mask_arr.shape[0], cls_ids.shape[0])):
        cls_id = int(cls_ids[i])
        mask = (mask_arr[i] > 0.5).astype(np.uint8)
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if cls_id == 0:
            ball_masks.append(mask)
        elif cls_id == 1:
            lane_masks.append(mask)
        elif cls_id == 2:
            pins_masks.append(mask)

    return FrameSegmentation(ball_masks, lane_masks, pins_masks, (int(orig_h), int(orig_w)))


def _transcode_to_mp4(video_path: str) -> str:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg to handle this video format.")

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-an",
        tmp.name,
    ]
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
        split_video.add_frame(VideoFrame(frame_index, timestamp_s, frame))
        frame_index += 1

    cap.release()
    return split_video


def split_video_into_frames(video_path: str) -> SplitVideo:
    split_video = _read_frames(video_path)

    if split_video.frames:
        check_count = min(5, len(split_video.frames))
        all_black = all(np.max(split_video.frames[i].image) == 0 for i in range(check_count))
        if all_black:
            tmp_path = _transcode_to_mp4(video_path)
            try:
                split_video = _read_frames(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    return split_video


def resolve_existing_path(path_str: str, *, base_dirs: Sequence[Path]) -> Path:
    input_path = Path(path_str).expanduser()
    candidates: List[Path] = []

    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append(Path.cwd() / input_path)
        candidates.extend(base_dir / input_path for base_dir in base_dirs)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Path does not exist: {path_str}")


def resolve_output_path(
    output_path_str: str | None,
    video_path: Path,
    *,
    base_dirs: Sequence[Path],
) -> Path:
    if not output_path_str:
        return (video_path.parent / f"{video_path.stem}_poster_seg_overlay.mp4").resolve()

    output_path = Path(output_path_str).expanduser()
    if not output_path.is_absolute():
        candidates = [Path.cwd() / output_path]
        candidates.extend(base_dir / output_path_str for base_dir in base_dirs)
        output_path = candidates[0]
        for candidate in candidates:
            if candidate.exists():
                output_path = candidate
                break

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    if output_path.suffix.lower() in video_suffixes:
        output_file = output_path
    else:
        output_file = output_path / f"{video_path.stem}_poster_seg_overlay.mp4"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    return output_file.resolve()


def _lane_dst_corners_m() -> np.ndarray:
    return np.array(
        [
            [0.0, LANE_LENGTH_M],
            [LANE_WIDTH_M, LANE_LENGTH_M],
            [LANE_WIDTH_M, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )


def clean_mask(mask: np.ndarray, *, close_size: int = 15, open_size: int = 7) -> np.ndarray:
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open, iterations=1)
    return cleaned


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def approx_to_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 20:
        return None

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

    if best_approx is not None and 4 <= len(best_approx) <= 6:
        pts = best_approx.reshape(-1, 2).astype(np.float32)
        while len(pts) > 4:
            min_loss = float("inf")
            drop_idx = 0
            for i in range(len(pts)):
                tri = np.array([pts[(i - 1) % len(pts)], pts[i], pts[(i + 1) % len(pts)]])
                area = 0.5 * abs(float(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                if area < min_loss:
                    min_loss = area
                    drop_idx = i
            pts = np.delete(pts, drop_idx, axis=0)
        return pts.astype(np.int32)

    return None


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts_f = np.asarray(pts, dtype=np.float32)
    sorted_by_y = pts_f[np.argsort(pts_f[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def hough_quad_fallback(mask: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    edges = cv2.Canny(mask * 255, 50, 150)
    min_dim = min(h, w)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, min_dim // 8))
    if lines is None or len(lines) < 4:
        return None

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

    def _line_intersect(r1: float, t1: float, r2: float, t2: float) -> tuple[float, float] | None:
        det = np.cos(t1) * np.sin(t2) - np.cos(t2) * np.sin(t1)
        if abs(det) < 1e-6:
            return None
        x = (r1 * np.sin(t2) - r2 * np.sin(t1)) / det
        y = (r2 * np.cos(t1) - r1 * np.cos(t2)) / det
        return x, y

    corners: List[tuple[float, float]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            pt = _line_intersect(unique[i][0], unique[i][1], unique[j][0], unique[j][1])
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                corners.append(pt)

    if len(corners) < 4:
        return None

    corners_np = np.array(corners, dtype=np.float32)
    moments = cv2.moments(cv2.convexHull(contour))
    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    angles = np.arctan2(corners_np[:, 1] - cy, corners_np[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    if len(sorted_idx) >= 4:
        step = len(sorted_idx) / 4.0
        picks = [sorted_idx[int(i * step)] for i in range(4)]
        return corners_np[picks].astype(np.int32)

    return None


def refine_sides_from_image(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    *,
    strip_width: float,
) -> np.ndarray:
    line_params = fit_side_lines_from_image(frame_bgr, quad, strip_width=strip_width)
    refined = quad.copy().astype(np.float32)
    for (top_idx, bot_idx), line in zip([(0, 3), (1, 2)], line_params):
        if line is None:
            continue
        new_top_x = x_at_y(line, float(refined[top_idx, 1]))
        new_bot_x = x_at_y(line, float(refined[bot_idx, 1]))
        if new_top_x is None or new_bot_x is None:
            continue
        if abs(new_top_x - refined[top_idx, 0]) < strip_width:
            refined[top_idx, 0] = new_top_x
        if abs(new_bot_x - refined[bot_idx, 0]) < strip_width:
            refined[bot_idx, 0] = new_bot_x

    h, w = frame_bgr.shape[:2]
    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)
    return refined.astype(np.int32)


def fit_side_lines_from_image(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    *,
    strip_width: float,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    fitted: list[tuple[float, float] | None] = []
    for top_idx, bot_idx in [(0, 3), (1, 2)]:
        pt_top = quad[top_idx].astype(np.float32)
        pt_bot = quad[bot_idx].astype(np.float32)

        dx = pt_bot[0] - pt_top[0]
        dy = pt_bot[1] - pt_top[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 20:
            fitted.append(None)
            continue

        nx, ny = -dy / length, dx / length
        offset = np.array([nx, ny], dtype=np.float32) * strip_width
        strip_poly = np.array(
            [pt_top + offset, pt_top - offset, pt_bot - offset, pt_bot + offset],
            dtype=np.int32,
        )

        strip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(strip_mask, [strip_poly], 255)
        strip_edges = cv2.bitwise_and(edges, strip_mask)

        min_line_len = max(int(length * 0.25), 30)
        lines = cv2.HoughLinesP(
            strip_edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=min_line_len,
            maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            fitted.append(None)
            continue

        best_line = None
        best_len = 0.0
        for seg in lines:
            sx1, sy1, sx2, sy2 = seg[0]
            sdx, sdy = abs(sx2 - sx1), abs(sy2 - sy1)
            if sdy < sdx:
                continue
            seg_len = np.sqrt(float(sdx**2 + sdy**2))
            if seg_len > best_len:
                best_len = seg_len
                best_line = seg[0]

        if best_line is None:
            fitted.append(None)
            continue

        lx1, ly1, lx2, ly2 = best_line
        if abs(ly2 - ly1) < 1:
            fitted.append(None)
            continue

        a = (lx2 - lx1) / (ly2 - ly1)
        b = lx1 - a * ly1
        fitted.append((float(a), float(b)))

    while len(fitted) < 2:
        fitted.append(None)
    return fitted[0], fitted[1]


def x_at_y(line: tuple[float, float] | None, y: float) -> float | None:
    if line is None:
        return None
    a, b = line
    return float(a * y + b)


def line_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[float, float] | None:
    dy = float(p1[1] - p0[1])
    if abs(dy) < 1e-6:
        return None
    a = float((p1[0] - p0[0]) / dy)
    b = float(p0[0] - a * p0[1])
    return a, b


def intersect_xy_lines(
    line1: tuple[float, float] | None,
    line2: tuple[float, float] | None,
) -> np.ndarray | None:
    if line1 is None or line2 is None:
        return None
    a1, b1 = line1
    a2, b2 = line2
    denom = a1 - a2
    if abs(denom) < 1e-6:
        return None
    y = (b2 - b1) / denom
    x = a1 * y + b1
    return np.array([x, y], dtype=np.float32)


def point_on_ray_with_shared_lambda(
    bottom: np.ndarray,
    vanishing: np.ndarray,
    lam: float,
) -> np.ndarray:
    lam = float(np.clip(lam, 0.01, 0.99))
    return bottom + lam * (vanishing - bottom)


def _rebuild_quad_with_geometry(
    quad: np.ndarray,
    *,
    frame_bgr: np.ndarray | None,
    pins_box: np.ndarray | None,
    image_height: int,
    tuning: GeometryTuning,
) -> np.ndarray:
    quad_f = quad.astype(np.float32)
    left_line = None
    right_line = None
    if frame_bgr is not None:
        left_line, right_line = fit_side_lines_from_image(
            frame_bgr,
            quad_f,
            strip_width=tuning.side_refine_strip_width,
        )
    if left_line is None:
        left_line = line_from_segment(quad_f[0], quad_f[3])
    if right_line is None:
        right_line = line_from_segment(quad_f[1], quad_f[2])

    top_y = float(min(quad_f[0, 1], quad_f[1, 1]))
    pins_cx = None
    if pins_box is not None:
        pins_bot_y = float(pins_box[3])
        if abs(pins_bot_y - top_y) < image_height * tuning.pins_y_tolerance_ratio:
            top_y = pins_bot_y
        pins_cx = 0.5 * (float(pins_box[0]) + float(pins_box[2]))

    bottom_left_y = float(quad_f[3, 1])
    bottom_right_y = float(quad_f[2, 1])
    bottom_y = 0.5 * (bottom_left_y + bottom_right_y)
    bottom_mid = 0.5 * (quad_f[3] + quad_f[2])

    tl_x = x_at_y(left_line, top_y)
    tr_x = x_at_y(right_line, top_y)
    bl_x = x_at_y(left_line, bottom_left_y)
    br_x = x_at_y(right_line, bottom_right_y)

    if tl_x is None:
        tl_x = float(quad_f[0, 0])
    if tr_x is None:
        tr_x = float(quad_f[1, 0])
    if bl_x is None:
        bl_x = float(quad_f[3, 0])
    if br_x is None:
        br_x = float(quad_f[2, 0])

    if pins_box is not None and pins_cx is not None:
        pins_half_w = 0.5 * float(pins_box[2] - pins_box[0])
        lane_half_at_pins = pins_half_w * tuning.pins_lane_width_scale
        expected_tl = pins_cx - lane_half_at_pins
        expected_tr = pins_cx + lane_half_at_pins
        blend = float(np.clip(tuning.pins_top_x_blend, 0.0, 1.0))
        tl_x = (1.0 - blend) * tl_x + blend * expected_tl
        tr_x = (1.0 - blend) * tr_x + blend * expected_tr
        expected_top_width = 2.0 * lane_half_at_pins
        target_top_center = (
            (1.0 - tuning.top_center_pin_blend) * (0.5 * (tl_x + tr_x))
            + tuning.top_center_pin_blend * pins_cx
        )
    else:
        expected_top_width = max(tr_x - tl_x, 1.0)
        target_top_center = 0.5 * (tl_x + tr_x)

    current_top_width = max(tr_x - tl_x, 1.0)
    target_top_width = (
        (1.0 - tuning.top_width_target_weight) * current_top_width
        + tuning.top_width_target_weight * expected_top_width
    )
    geometry_tl = target_top_center - 0.5 * target_top_width
    geometry_tr = target_top_center + 0.5 * target_top_width

    left_pull = float(np.clip(tuning.left_top_geometry_pull, 0.0, 1.0))
    right_pull = float(np.clip(tuning.right_top_geometry_pull, 0.0, 1.0))
    tl_x = (1.0 - left_pull) * tl_x + left_pull * geometry_tl
    tr_x = (1.0 - right_pull) * tr_x + right_pull * geometry_tr

    current_bottom_width = max(br_x - bl_x, 1.0)
    target_bottom_width = current_bottom_width
    if target_top_width > 0:
        geometric_bottom_width = target_top_width / max(tuning.target_taper_ratio, 1e-6)
        target_bottom_width = (
            (1.0 - tuning.bottom_width_target_weight) * current_bottom_width
            + tuning.bottom_width_target_weight * geometric_bottom_width
        )
        target_bottom_width = min(target_bottom_width, current_bottom_width)

    bottom_center_x = 0.5 * (bl_x + br_x)
    bl_x = bottom_center_x - 0.5 * target_bottom_width
    br_x = bottom_center_x + 0.5 * target_bottom_width

    if pins_box is not None and pins_cx is not None:
        centerline = line_from_segment(
            np.array([bottom_center_x, bottom_y], dtype=np.float32),
            np.array([pins_cx, top_y], dtype=np.float32),
        )
        right_side = line_from_segment(
            np.array([tr_x, top_y], dtype=np.float32),
            np.array([br_x, bottom_right_y], dtype=np.float32),
        )
        vanishing = intersect_xy_lines(right_side, centerline)
        if vanishing is not None and vanishing[1] < top_y:
            br = np.array([br_x, bottom_right_y], dtype=np.float32)
            bl = np.array([bl_x, bottom_left_y], dtype=np.float32)
            dy_right = float(vanishing[1] - br[1])
            if abs(dy_right) > 1e-6:
                lam = (top_y - br[1]) / dy_right
                lam = float(np.clip(lam, 0.01, 0.99))
                tr_pt = point_on_ray_with_shared_lambda(br, vanishing, lam)
                tl_pt = point_on_ray_with_shared_lambda(bl, vanishing, lam)
                tl_x = float(tl_pt[0])
                tr_x = float(tr_pt[0])

    width_bottom = max(br_x - bl_x, 1.0)
    width_top = max(tr_x - tl_x, 1.0)
    max_top_width = width_bottom * tuning.max_top_width_ratio
    if width_top >= max_top_width:
        top_center = target_top_center
        tl_x = top_center - 0.5 * max_top_width
        tr_x = top_center + 0.5 * max_top_width

    rebuilt = np.array(
        [
            [tl_x, top_y],
            [tr_x, top_y],
            [br_x, bottom_right_y],
            [bl_x, bottom_left_y],
        ],
        dtype=np.float32,
    )
    return rebuilt.astype(np.int32)


def validate_vanishing_point(quad: np.ndarray, img_shape_hw: tuple[int, int]) -> bool:
    h, w = img_shape_hw
    q = quad.astype(np.float64)
    tl, tr, br, bl = q[0], q[1], q[2], q[3]

    left_dx = bl[0] - tl[0]
    left_dy = bl[1] - tl[1]
    right_dx = br[0] - tr[0]
    right_dy = br[1] - tr[1]

    det = left_dx * right_dy - right_dx * left_dy
    if abs(det) < 1e-6:
        return True

    t = ((tr[0] - tl[0]) * right_dy - (tr[1] - tl[1]) * right_dx) / det
    vp_x = tl[0] + t * left_dx
    vp_y = tl[1] + t * left_dy

    if vp_y > tl[1]:
        return False
    if vp_x < -w or vp_x > 2 * w:
        return False
    return True


def evaluate_trapezoid(
    lane_mask: np.ndarray,
    polygon: np.ndarray,
    *,
    score_coverage_weight: float = 0.55,
    score_geometry_weight: float = 0.15,
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

    q = polygon.astype(np.float64)
    width_top = abs(q[1, 0] - q[0, 0])
    width_bottom = abs(q[2, 0] - q[3, 0])
    center_top = (q[0, 0] + q[1, 0]) / 2.0
    center_bottom = (q[2, 0] + q[3, 0]) / 2.0

    taper = width_top / max(width_bottom, 1.0)
    taper_score = 1.0 - min(abs(taper - 0.45) / 0.45, 1.0)

    center_shift = abs(center_top - center_bottom) / max(width_bottom, 1.0)
    symmetry_score = max(0.0, 1.0 - center_shift * 2.0)

    geometry = 0.5 * taper_score + 0.5 * symmetry_score
    overlap_weight = 1.0 - score_coverage_weight - score_geometry_weight
    score = (
        score_coverage_weight * coverage
        + overlap_weight * purity
        + score_geometry_weight * geometry
    )
    return coverage, purity, score


def _masks_to_boxes(masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))
    return boxes


def find_nearest_pins_box(lane_mask: np.ndarray, pins_boxes: Sequence[np.ndarray]) -> np.ndarray | None:
    if not pins_boxes:
        return None

    ys, xs = np.where(lane_mask > 0)
    if len(xs) == 0:
        return None
    lane_cx = float(np.mean(xs))

    best_box: np.ndarray | None = None
    best_dist = float("inf")
    for box in pins_boxes:
        pins_cx = float(box[0] + box[2]) / 2.0
        dist = abs(pins_cx - lane_cx)
        if dist < best_dist:
            best_dist = dist
            best_box = box

    return best_box


def find_nearest_pins_box_for_polygon(
    lane_polygon: np.ndarray,
    pins_boxes: Sequence[np.ndarray],
) -> np.ndarray | None:
    if not pins_boxes:
        return None

    lane_cx = float(np.mean(lane_polygon[:, 0]))
    best_box: np.ndarray | None = None
    best_dist = float("inf")
    for box in pins_boxes:
        pins_cx = float(box[0] + box[2]) / 2.0
        dist = abs(pins_cx - lane_cx)
        if dist < best_dist:
            best_dist = dist
            best_box = box

    return best_box


def build_lane_trapezoid(
    lane_mask: np.ndarray,
    *,
    frame_bgr: np.ndarray | None = None,
    pins_boxes: Sequence[np.ndarray] | None = None,
    tuning: GeometryTuning | None = None,
) -> TrapezoidCandidate | None:
    if tuning is None:
        tuning = GeometryTuning()

    cleaned = clean_mask((lane_mask > 0).astype(np.uint8))
    cleaned = keep_largest_component(cleaned)

    contour = largest_contour(cleaned)
    if contour is None or cv2.contourArea(contour) < 500:
        return None

    quad = approx_to_quad(contour)
    if quad is None:
        quad = hough_quad_fallback(cleaned, contour)
    if quad is None:
        return None

    quad = order_quad_points(quad)
    pins_box = None

    if pins_boxes:
        pins_box = find_nearest_pins_box(cleaned, pins_boxes)
        if pins_box is not None:
            pins_bot_y = int(pins_box[3])
            pins_cx = float(pins_box[0] + pins_box[2]) / 2.0
            pins_half_w = float(pins_box[2] - pins_box[0]) / 2.0
            current_top_y = int(min(quad[0, 1], quad[1, 1]))
            if abs(pins_bot_y - current_top_y) < lane_mask.shape[0] * tuning.pins_y_tolerance_ratio:
                quad[0, 1] = pins_bot_y
                quad[1, 1] = pins_bot_y

                lane_half_at_pins = pins_half_w * tuning.pins_lane_width_scale
                proposed_tl_x = pins_cx - lane_half_at_pins
                proposed_tr_x = pins_cx + lane_half_at_pins
                if abs(proposed_tl_x - quad[0, 0]) < tuning.pins_x_max_adjust_px:
                    quad[0, 0] = int(proposed_tl_x)
                if abs(proposed_tr_x - quad[1, 0]) < tuning.pins_x_max_adjust_px:
                    quad[1, 0] = int(proposed_tr_x)

    quad = _rebuild_quad_with_geometry(
        quad,
        frame_bgr=frame_bgr,
        pins_box=pins_box,
        image_height=lane_mask.shape[0],
        tuning=tuning,
    )

    if not validate_vanishing_point(quad, lane_mask.shape):
        return None

    width_top = abs(int(quad[1, 0]) - int(quad[0, 0]))
    width_bottom = abs(int(quad[2, 0]) - int(quad[3, 0]))
    if width_top < 4 or width_bottom < 8:
        return None

    coverage, purity, score = evaluate_trapezoid(cleaned, quad)
    y_top = int(min(quad[0, 1], quad[1, 1]))
    y_bottom = int(max(quad[2, 1], quad[3, 1]))
    return TrapezoidCandidate(quad, float(coverage), float(purity), float(score), y_top, y_bottom)


def _ball_contact_point_from_mask(ball_mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(ball_mask > 0)
    if xs.size == 0:
        return None
    y_contact = float(np.max(ys))
    x_contact = float(np.min(xs) + np.max(xs)) / 2.0
    return x_contact, y_contact


def _choose_ball_contact_for_lane(
    ball_masks: Sequence[np.ndarray],
    lane_polygon: np.ndarray,
) -> tuple[float, float] | None:
    if not ball_masks:
        return None

    lane_poly = lane_polygon.reshape((-1, 1, 2)).astype(np.float32)
    lane_center_x = float(np.mean(lane_polygon[:, 0]))
    best_point: tuple[float, float] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for mask in ball_masks:
        contact = _ball_contact_point_from_mask(mask)
        if contact is None:
            continue

        x_contact, y_contact = contact
        inside = cv2.pointPolygonTest(
            lane_poly,
            (float(x_contact), float(y_contact)),
            measureDist=False,
        ) >= 0

        area = float(np.count_nonzero(mask))
        key = (
            0.0 if inside else 1.0,
            -float(y_contact),
            abs(float(x_contact) - lane_center_x),
            -area,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_point = (float(x_contact), float(y_contact))

    return best_point


def _project_point_homography(pt_xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    vec = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    dst = H @ vec
    if abs(float(dst[2])) < 1e-9:
        raise RuntimeError("Invalid homography projection: denominator ~0")
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def _correct_trapezoid_top_corners(quad: np.ndarray) -> np.ndarray:
    quad_f = quad.astype(np.float64)
    tl, tr, br, bl = quad_f[0], quad_f[1], quad_f[2], quad_f[3]

    def _homogeneous(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], 1.0])

    left_line = np.cross(_homogeneous(bl), _homogeneous(tl))
    right_line = np.cross(_homogeneous(br), _homogeneous(tr))
    vanishing_h = np.cross(left_line, right_line)
    if abs(vanishing_h[2]) < 1e-10:
        return quad

    vanishing = vanishing_h[:2] / vanishing_h[2]
    if vanishing[1] > min(tl[1], tr[1]):
        return quad

    ray_left = vanishing - bl
    ray_right = vanishing - br
    ray_left_sq = float(np.dot(ray_left, ray_left))
    ray_right_sq = float(np.dot(ray_right, ray_right))
    if ray_left_sq < 1.0 or ray_right_sq < 1.0:
        return quad

    lam_left = float(np.dot(tl - bl, ray_left) / ray_left_sq)
    lam_right = float(np.dot(tr - br, ray_right) / ray_right_sq)
    if lam_left <= 0.0 or lam_left >= 1.0 or lam_right <= 0.0 or lam_right >= 1.0:
        return quad

    tl_new = bl + lam_left * ray_left
    tr_new = br + lam_right * ray_right

    result = quad.copy().astype(np.float32)
    result[0] = tl_new.astype(np.float32)
    result[1] = tr_new.astype(np.float32)
    return result


def _compute_vanishing_point_from_quad(quad: np.ndarray) -> np.ndarray | None:
    quad_f = quad.astype(np.float64)
    tl, tr, br, bl = quad_f[0], quad_f[1], quad_f[2], quad_f[3]

    def _homogeneous(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], 1.0], dtype=np.float64)

    left_line = np.cross(_homogeneous(bl), _homogeneous(tl))
    right_line = np.cross(_homogeneous(br), _homogeneous(tr))
    vanishing_h = np.cross(left_line, right_line)
    if abs(vanishing_h[2]) < 1e-10:
        return None
    return vanishing_h[:2] / vanishing_h[2]


def _ray_lambda_from_point(bottom: np.ndarray, vanishing: np.ndarray, point: np.ndarray) -> float | None:
    ray = vanishing - bottom
    ray_sq = float(np.dot(ray, ray))
    if ray_sq < 1.0:
        return None
    lam = float(np.dot(point - bottom, ray) / ray_sq)
    return float(np.clip(lam, 0.01, 0.99))


def _ray_lambda_from_target_y(bottom: np.ndarray, vanishing: np.ndarray, target_y: float) -> float | None:
    dy = float(vanishing[1] - bottom[1])
    if abs(dy) < 1e-6:
        return None
    lam = (target_y - bottom[1]) / dy
    return float(np.clip(lam, 0.01, 0.99))


def _point_on_ray_from_lambda(bottom: np.ndarray, vanishing: np.ndarray, lam: float) -> np.ndarray:
    lam = float(np.clip(lam, 0.01, 0.99))
    return bottom + lam * (vanishing - bottom)


def _apply_post_pins_top_edge_guidance(
    quad: np.ndarray,
    pins_box: np.ndarray | None,
    *,
    tuning: GeometryTuning,
) -> np.ndarray:
    if pins_box is None:
        return quad

    quad_f = quad.astype(np.float64)
    tl, tr, br, bl = quad_f[0], quad_f[1], quad_f[2], quad_f[3]
    vanishing = _compute_vanishing_point_from_quad(quad_f)
    if vanishing is None:
        return quad

    lam_left = _ray_lambda_from_point(bl, vanishing, tl)
    lam_right = _ray_lambda_from_point(br, vanishing, tr)
    if lam_left is None or lam_right is None:
        return quad

    pins_bot_y = float(pins_box[3])
    current_mean_y = 0.5 * (float(tl[1]) + float(tr[1]))
    target_pins_y = (
        (1.0 - tuning.post_pins_y_weight) * current_mean_y
        + tuning.post_pins_y_weight * pins_bot_y
    )

    pins_lam_left = _ray_lambda_from_target_y(bl, vanishing, target_pins_y)
    pins_lam_right = _ray_lambda_from_target_y(br, vanishing, target_pins_y)
    if pins_lam_left is None or pins_lam_right is None:
        return quad

    current_shared_lam = 0.5 * (lam_left + lam_right)
    pins_shared_lam = 0.5 * (pins_lam_left + pins_lam_right)
    target_shared_lam = (
        (1.0 - tuning.top_edge_level_weight) * current_shared_lam
        + tuning.top_edge_level_weight * pins_shared_lam
    )

    current_lam_delta = lam_right - lam_left
    residual_lam_delta = current_lam_delta * tuning.residual_top_asymmetry_ratio

    if lam_left > lam_right:
        residual_lam_delta /= max(tuning.higher_corner_level_boost, 1e-6)
    elif lam_right > lam_left:
        residual_lam_delta *= max(1.0 / max(tuning.higher_corner_level_boost, 1e-6), 0.0)

    target_lam_left = target_shared_lam - 0.5 * residual_lam_delta
    target_lam_right = target_shared_lam + 0.5 * residual_lam_delta

    tl_new = _point_on_ray_from_lambda(bl, vanishing, target_lam_left)
    tr_new = _point_on_ray_from_lambda(br, vanishing, target_lam_right)

    max_adjust = float(tuning.top_edge_max_adjust_px)
    tl_new[1] = np.clip(tl_new[1], tl[1] - max_adjust, tl[1] + max_adjust)
    tr_new[1] = np.clip(tr_new[1], tr[1] - max_adjust, tr[1] + max_adjust)

    tl_new = _point_on_ray_from_lambda(
        bl,
        vanishing,
        _ray_lambda_from_target_y(bl, vanishing, float(tl_new[1])) or target_lam_left,
    )
    tr_new = _point_on_ray_from_lambda(
        br,
        vanishing,
        _ray_lambda_from_target_y(br, vanishing, float(tr_new[1])) or target_lam_right,
    )

    result = quad.copy().astype(np.float32)
    result[0] = tl_new.astype(np.float32)
    result[1] = tr_new.astype(np.float32)
    return result


def run_lane_ball_postprocessing(
    segmentations_by_frame: Dict[int, FrameSegmentation],
    fps: float,
    start_frame: int,
    *,
    frames_bgr: Sequence[np.ndarray] | None = None,
    min_trapezoid_score: float = DEFAULT_MIN_TRAPEZOID_SCORE,
    tuning: GeometryTuning | None = None,
) -> PostprocessResult:
    if tuning is None:
        tuning = GeometryTuning()

    def _empty_result(scanned: int = 0, with_lane: int = 0, coverage: float = 0.0) -> PostprocessResult:
        return PostprocessResult(
            ball_positions=[],
            homography_selection=HomographySelection(
                frame_index=start_frame,
                homography=np.eye(3, dtype=np.float32),
                src_corners=np.zeros((4, 2), dtype=np.float32),
                dst_corners=np.zeros((4, 2), dtype=np.float32),
                is_trapezoid=False,
                selected_lane_contours=0,
            ),
            health=PostprocessHealth(
                frames_scanned_for_h=scanned,
                frames_with_lane=with_lane,
                frames_with_ball=0,
                lane_polygon_count_at_h=0,
                homography_determinant=0.0,
                homography_condition_number=0.0,
                mean_lane_coverage_ratio=coverage,
            ),
        )

    if not segmentations_by_frame:
        return _empty_result()

    sorted_frames = sorted(k for k in segmentations_by_frame.keys() if k >= start_frame)
    if not sorted_frames:
        return _empty_result()

    smoother = TemporalSmoother()
    coverage_values: List[float] = []
    frame_candidate_count: Dict[int, int] = {}

    scanned = 0
    frames_with_lane = 0

    for frame_index in sorted_frames:
        scanned += 1
        frame_seg = segmentations_by_frame[frame_index]
        h, w = frame_seg.frame_shape
        frame_area = float(max(h * w, 1))

        lane_area = float(sum(np.count_nonzero(mask) for mask in frame_seg.lane_masks))
        coverage_values.append(min(lane_area / frame_area, 1.0))
        if lane_area > 0:
            frames_with_lane += 1

        frame_bgr = None
        if frames_bgr is not None and frame_index < len(frames_bgr):
            frame_bgr = frames_bgr[frame_index]

        pins_boxes = _masks_to_boxes(frame_seg.pins_masks)
        ball_boxes = _masks_to_boxes(frame_seg.ball_masks)

        candidates: List[TrapezoidCandidate] = []
        lane_mask_areas: List[float] = []

        for lane_mask in frame_seg.lane_masks:
            trap = build_lane_trapezoid(
                lane_mask,
                frame_bgr=frame_bgr,
                pins_boxes=pins_boxes,
                tuning=tuning,
            )
            if trap is None or trap.score < min_trapezoid_score:
                continue
            candidates.append(trap)
            lane_mask_areas.append(float(np.count_nonzero(lane_mask)))

        frame_candidate_count[frame_index] = len(candidates)
        if candidates:
            smoother.update(
                candidates,
                frame_idx=frame_index,
                ball_boxes=ball_boxes,
                lane_mask_areas=lane_mask_areas,
            )

    active_lane = smoother.select_active_lane()
    if active_lane is None:
        return _empty_result(
            scanned=scanned,
            with_lane=frames_with_lane,
            coverage=float(np.mean(coverage_values)) if coverage_values else 0.0,
        )

    src_corners = active_lane.best_quad.astype(np.float32)
    src_corners = _correct_trapezoid_top_corners(src_corners)
    best_frame_seg = segmentations_by_frame.get(active_lane.best_frame_idx)
    if best_frame_seg is not None:
        best_pins_boxes = _masks_to_boxes(best_frame_seg.pins_masks)
        nearest_pins_box = find_nearest_pins_box_for_polygon(src_corners, best_pins_boxes)
        guided_corners = _apply_post_pins_top_edge_guidance(
            src_corners,
            nearest_pins_box,
            tuning=tuning,
        )
        guided_width_top = abs(float(guided_corners[1, 0]) - float(guided_corners[0, 0]))
        guided_width_bottom = abs(float(guided_corners[2, 0]) - float(guided_corners[3, 0]))
        if guided_width_top < guided_width_bottom:
            src_corners = guided_corners
    dst = _lane_dst_corners_m()
    homography = cv2.getPerspectiveTransform(src_corners, dst)

    selection = HomographySelection(
        frame_index=int(active_lane.best_frame_idx),
        homography=homography,
        src_corners=src_corners,
        dst_corners=dst,
        is_trapezoid=True,
        selected_lane_contours=int(frame_candidate_count.get(active_lane.best_frame_idx, 1)),
    )

    positions: List[BallPos] = []
    frames_with_ball = 0
    for frame_index in sorted(segmentations_by_frame.keys()):
        if frame_index < start_frame:
            continue

        frame_seg = segmentations_by_frame[frame_index]
        contact = _choose_ball_contact_for_lane(frame_seg.ball_masks, selection.src_corners)
        if contact is None:
            continue

        frames_with_ball += 1
        x_m, y_m = _project_point_homography(contact, selection.homography)
        x_m = float(np.clip(x_m, -0.5, LANE_WIDTH_M + 0.5))
        y_m = float(np.clip(y_m, -1.0, LANE_LENGTH_M + 1.0))
        positions.append(
            BallPos(
                frame_index=int(frame_index),
                timestamp_s=float(frame_index / max(fps, 1e-6)),
                x_m=x_m,
                y_m=y_m,
            )
        )

    health = PostprocessHealth(
        frames_scanned_for_h=int(scanned),
        frames_with_lane=int(frames_with_lane),
        frames_with_ball=int(frames_with_ball),
        lane_polygon_count_at_h=int(selection.selected_lane_contours),
        homography_determinant=float(np.linalg.det(selection.homography)),
        homography_condition_number=float(np.linalg.cond(selection.homography)),
        mean_lane_coverage_ratio=float(np.mean(coverage_values)) if coverage_values else 0.0,
    )
    return PostprocessResult(positions, selection, health)


def _alpha_blend_color(
    frame_bgr: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha_map: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)[..., None]
    color = np.empty_like(frame_bgr, dtype=np.float32)
    color[..., 0] = float(color_bgr[0])
    color[..., 1] = float(color_bgr[1])
    color[..., 2] = float(color_bgr[2])
    base = frame_bgr.astype(np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _masked_blend(
    frame_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    alpha_map: np.ndarray,
) -> np.ndarray:
    alpha = np.clip(alpha_map.astype(np.float32), 0.0, 1.0)[..., None]
    base = frame_bgr.astype(np.float32)
    overlay = overlay_bgr.astype(np.float32)
    blended = base * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_lane_overlay(frame_bgr: np.ndarray, lane_quad: np.ndarray, *, lane_alpha: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lane_mask, [lane_quad.astype(np.int32)], 255)

    sigma = max(4.0, min(h, w) * 0.006)
    feather = cv2.GaussianBlur(lane_mask, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(
        np.float32
    ) / 255.0

    y_gradient = np.linspace(0.85, 1.15, h, dtype=np.float32)[:, None]
    fill_alpha = feather * y_gradient * lane_alpha
    output = _alpha_blend_color(frame_bgr, (255, 214, 120), fill_alpha)

    glow = np.zeros_like(frame_bgr)
    cv2.line(glow, tuple(lane_quad[0]), tuple(lane_quad[3]), (255, 250, 220), 10, cv2.LINE_AA)
    cv2.line(glow, tuple(lane_quad[1]), tuple(lane_quad[2]), (255, 250, 220), 10, cv2.LINE_AA)
    cv2.line(glow, tuple(lane_quad[0]), tuple(lane_quad[1]), (255, 245, 210), 6, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=7, sigmaY=7)
    output = _masked_blend(output, glow, feather * 0.22)

    centerline = np.zeros_like(frame_bgr)
    top_mid = np.mean(lane_quad[[0, 1]], axis=0).astype(int)
    bottom_mid = np.mean(lane_quad[[3, 2]], axis=0).astype(int)
    cv2.line(centerline, tuple(top_mid), tuple(bottom_mid), (255, 236, 188), 2, cv2.LINE_AA)
    centerline = cv2.GaussianBlur(centerline, (0, 0), sigmaX=3, sigmaY=3)
    output = _masked_blend(output, centerline, feather * 0.18)

    cv2.line(output, tuple(lane_quad[0]), tuple(lane_quad[3]), (255, 245, 225), 3, cv2.LINE_AA)
    cv2.line(output, tuple(lane_quad[1]), tuple(lane_quad[2]), (255, 245, 225), 3, cv2.LINE_AA)
    cv2.line(output, tuple(lane_quad[0]), tuple(lane_quad[1]), (255, 240, 214), 2, cv2.LINE_AA)
    return output


def choose_ball_mask_for_lane(
    ball_masks: Sequence[np.ndarray],
    lane_polygon: np.ndarray | None,
) -> np.ndarray | None:
    if not ball_masks:
        return None

    lane_poly = None
    lane_center_x = 0.0
    if lane_polygon is not None:
        lane_poly = lane_polygon.reshape((-1, 1, 2)).astype(np.float32)
        lane_center_x = float(np.mean(lane_polygon[:, 0]))

    best_mask: np.ndarray | None = None
    best_key: tuple[float, float, float, float] | None = None

    for mask in ball_masks:
        contact = _ball_contact_point_from_mask(mask)
        if contact is None:
            continue

        x_contact, y_contact = contact
        inside = False
        center_delta = 0.0
        if lane_poly is not None:
            inside = (
                cv2.pointPolygonTest(
                    lane_poly,
                    (float(x_contact), float(y_contact)),
                    measureDist=False,
                )
                >= 0
            )
            center_delta = abs(float(x_contact) - lane_center_x)

        area = float(np.count_nonzero(mask))
        key = (
            0.0 if inside else 1.0,
            -float(y_contact),
            center_delta,
            -area,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_mask = mask

    return best_mask


def render_ball_overlay(frame_bgr: np.ndarray, ball_mask: np.ndarray, *, ball_alpha: float) -> np.ndarray:
    cleaned = clean_mask((ball_mask > 0).astype(np.uint8), close_size=9, open_size=5)
    cleaned = keep_largest_component(cleaned)
    if np.count_nonzero(cleaned) == 0:
        return frame_bgr

    sigma = max(2.5, min(frame_bgr.shape[:2]) * 0.0035)
    feather = cv2.GaussianBlur((cleaned * 255).astype(np.uint8), (0, 0), sigmaX=sigma, sigmaY=sigma)
    feather = feather.astype(np.float32) / 255.0

    output = _alpha_blend_color(frame_bgr, (70, 122, 255), feather * ball_alpha)

    glow = np.zeros_like(frame_bgr)
    glow[cleaned > 0] = (92, 165, 255)
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=8, sigmaY=8)
    output = _masked_blend(output, glow, feather * 0.32)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (240, 248, 255), 2, cv2.LINE_AA)
    return output


def infer_segmentations(
    model: YOLO,
    frames_bgr: Sequence[np.ndarray],
    *,
    imgsz: int,
    batch_size: int,
    conf: float,
    iou: float,
    device: str | None,
) -> Dict[int, FrameSegmentation]:
    segmentations_by_frame: Dict[int, FrameSegmentation] = {}
    total = len(frames_bgr)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr[start:end]]
        results = model.predict(
            source=batch_rgb,
            verbose=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
        )

        for offset, result in enumerate(results):
            segmentations_by_frame[start + offset] = extract_frame_segmentation(result)

        print(f"  inference {start}..{end - 1} / {total - 1}")

    return segmentations_by_frame


def run_poster_overlay(
    *,
    checkpoint_path: Path,
    video_path: Path,
    output_path: Path,
    imgsz: int,
    batch_size: int,
    conf: float,
    iou: float,
    device: str | None,
    lane_alpha: float,
    ball_alpha: float,
    min_trapezoid_score: float,
    tuning: GeometryTuning,
) -> None:
    split_video = split_video_into_frames(str(video_path))
    if not split_video.frames:
        raise RuntimeError("No frames found in video.")

    fps = split_video.fps if split_video.fps > 0 else 30.0
    frames_bgr = [frame.image for frame in split_video.frames]

    print(f"Loading model: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))

    print("Running segmentation inference...")
    segmentations_by_frame = infer_segmentations(
        model,
        frames_bgr,
        imgsz=imgsz,
        batch_size=batch_size,
        conf=conf,
        iou=iou,
        device=device,
    )

    print("Selecting lane geometry from standalone LaneBalls logic...")
    post = run_lane_ball_postprocessing(
        segmentations_by_frame,
        fps,
        start_frame=0,
        frames_bgr=frames_bgr,
        min_trapezoid_score=min_trapezoid_score,
        tuning=tuning,
    )

    lane_quad: np.ndarray | None = None
    selection = post.homography_selection
    if selection.is_trapezoid and np.any(selection.src_corners):
        lane_quad = selection.src_corners.astype(np.int32)
        print(
            f"Selected lane frame={selection.frame_index} "
            f"contours={selection.selected_lane_contours} "
            f"det={post.health.homography_determinant:.6f}"
        )
    else:
        print("Lane geometry was not resolved cleanly; rendering ball overlay only.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (split_video.width, split_video.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    total = len(frames_bgr)
    try:
        for frame_index, frame_bgr in enumerate(frames_bgr):
            output = frame_bgr.copy()
            if lane_quad is not None:
                output = render_lane_overlay(output, lane_quad, lane_alpha=lane_alpha)

            ball_mask = choose_ball_mask_for_lane(
                segmentations_by_frame[frame_index].ball_masks,
                lane_quad,
            )
            if ball_mask is not None:
                output = render_ball_overlay(output, ball_mask, ball_alpha=ball_alpha)

            writer.write(output)
            if (frame_index + 1) % 100 == 0 or frame_index == total - 1:
                print(f"  rendered {frame_index + 1} / {total}")
    finally:
        writer.release()

    print(f"Poster overlay video saved to: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a poster-style segmentation overlay video with a clean lane "
            "overlay and ball segmentation only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        "--model_path",
        dest="checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="YOLO segmentation checkpoint. Defaults to pipeline_v2/runs/.../best.pt.",
    )
    parser.add_argument(
        "--video",
        "--video_path",
        dest="video",
        type=str,
        required=True,
        help="Input video path.",
    )
    parser.add_argument(
        "--output",
        "--output_path",
        dest="output",
        type=str,
        default=None,
        help="Output .mp4 path or output directory. Defaults beside the source video.",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size.")
    parser.add_argument("--batch_size", type=int, default=16, help="Predict batch size.")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device string, e.g. '0', 'cpu', or 'cuda'.",
    )
    parser.add_argument("--lane_alpha", type=float, default=0.22, help="Lane fill intensity.")
    parser.add_argument("--ball_alpha", type=float, default=0.58, help="Ball segmentation intensity.")
    parser.add_argument(
        "--min_trapezoid_score",
        type=float,
        default=DEFAULT_MIN_TRAPEZOID_SCORE,
        help="Minimum trapezoid score used for lane selection.",
    )
    parser.add_argument(
        "--pins_lane_width_scale",
        type=float,
        default=1.70,
        help="Estimated lane half-width at the pin deck relative to the detected pins half-width.",
    )
    parser.add_argument(
        "--side_refine_strip_width",
        type=float,
        default=25.0,
        help="Width of the image strip used to refine side edges.",
    )
    parser.add_argument(
        "--post_pins_y_weight",
        type=float,
        default=0.92,
        help="How strongly to pull the far lane edge toward the detected pin line after perspective correction.",
    )
    parser.add_argument(
        "--top_edge_level_weight",
        type=float,
        default=0.82,
        help="How strongly to level the far lane edge after the pin-line pull.",
    )
    parser.add_argument(
        "--top_edge_max_adjust_px",
        type=float,
        default=42.0,
        help="Maximum vertical adjustment per far corner after perspective correction.",
    )
    parser.add_argument(
        "--higher_corner_level_boost",
        type=float,
        default=1.35,
        help="Extra leveling gain applied to whichever far corner is higher than the other.",
    )
    parser.add_argument(
        "--residual_top_asymmetry_ratio",
        type=float,
        default=0.22,
        help="How much of the original top-edge asymmetry to preserve after pin-line fitting.",
    )
    parser.add_argument(
        "--top_width_target_weight",
        type=float,
        default=0.72,
        help="How strongly to enforce the expected far-edge width from the lane geometry.",
    )
    parser.add_argument(
        "--top_center_pin_blend",
        type=float,
        default=0.55,
        help="How strongly to center the far edge around the pin-deck center.",
    )
    parser.add_argument(
        "--left_top_geometry_pull",
        type=float,
        default=0.82,
        help="How strongly to pull the top-left corner toward the geometry-implied far edge.",
    )
    parser.add_argument(
        "--right_top_geometry_pull",
        type=float,
        default=0.12,
        help="How strongly to pull the top-right corner toward the geometry-implied far edge.",
    )
    parser.add_argument(
        "--target_taper_ratio",
        type=float,
        default=0.45,
        help="Desired far-edge to near-edge width ratio for the lane trapezoid.",
    )
    parser.add_argument(
        "--bottom_width_target_weight",
        type=float,
        default=0.68,
        help="How strongly to shrink the near edge toward the geometry-implied width.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dirs = [PIPELINE_ROOT, REPO_ROOT]
    checkpoint_path = resolve_existing_path(args.checkpoint, base_dirs=base_dirs)
    video_path = resolve_existing_path(args.video, base_dirs=base_dirs)
    output_path = resolve_output_path(args.output, video_path, base_dirs=base_dirs)

    if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        raise ValueError(f"Unsupported video format: {video_path.suffix}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if not (0.0 <= args.lane_alpha <= 1.0):
        raise ValueError("--lane_alpha must be in [0, 1]")
    if not (0.0 <= args.ball_alpha <= 1.0):
        raise ValueError("--ball_alpha must be in [0, 1]")
    if not (0.0 <= args.min_trapezoid_score <= 1.0):
        raise ValueError("--min_trapezoid_score must be in [0, 1]")
    if not (0.0 <= args.post_pins_y_weight <= 1.0):
        raise ValueError("--post_pins_y_weight must be in [0, 1]")
    if not (0.0 <= args.top_edge_level_weight <= 1.0):
        raise ValueError("--top_edge_level_weight must be in [0, 1]")
    if not (0.0 <= args.residual_top_asymmetry_ratio <= 1.0):
        raise ValueError("--residual_top_asymmetry_ratio must be in [0, 1]")
    if not (0.0 <= args.top_width_target_weight <= 1.0):
        raise ValueError("--top_width_target_weight must be in [0, 1]")
    if not (0.0 <= args.top_center_pin_blend <= 1.0):
        raise ValueError("--top_center_pin_blend must be in [0, 1]")
    if not (0.0 <= args.left_top_geometry_pull <= 1.0):
        raise ValueError("--left_top_geometry_pull must be in [0, 1]")
    if not (0.0 <= args.right_top_geometry_pull <= 1.0):
        raise ValueError("--right_top_geometry_pull must be in [0, 1]")
    if not (0.0 < args.target_taper_ratio < 1.0):
        raise ValueError("--target_taper_ratio must be in (0, 1)")
    if not (0.0 <= args.bottom_width_target_weight <= 1.0):
        raise ValueError("--bottom_width_target_weight must be in [0, 1]")

    tuning = GeometryTuning(
        pins_lane_width_scale=float(args.pins_lane_width_scale),
        side_refine_strip_width=float(args.side_refine_strip_width),
        post_pins_y_weight=float(args.post_pins_y_weight),
        top_edge_level_weight=float(args.top_edge_level_weight),
        top_edge_max_adjust_px=float(args.top_edge_max_adjust_px),
        higher_corner_level_boost=float(args.higher_corner_level_boost),
        residual_top_asymmetry_ratio=float(args.residual_top_asymmetry_ratio),
        top_width_target_weight=float(args.top_width_target_weight),
        top_center_pin_blend=float(args.top_center_pin_blend),
        left_top_geometry_pull=float(args.left_top_geometry_pull),
        right_top_geometry_pull=float(args.right_top_geometry_pull),
        target_taper_ratio=float(args.target_taper_ratio),
        bottom_width_target_weight=float(args.bottom_width_target_weight),
    )

    run_poster_overlay(
        checkpoint_path=checkpoint_path,
        video_path=video_path,
        output_path=output_path,
        imgsz=int(args.imgsz),
        batch_size=int(args.batch_size),
        conf=float(args.conf),
        iou=float(args.iou),
        device=args.device,
        lane_alpha=float(args.lane_alpha),
        ball_alpha=float(args.ball_alpha),
        min_trapezoid_score=float(args.min_trapezoid_score),
        tuning=tuning,
    )


if __name__ == "__main__":
    main()
