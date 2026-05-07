from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO

# Lane physical dimensions (regulation bowling lane).
LANE_LENGTH_M = 18.288
LANE_WIDTH_M = 1.0541
DEFAULT_MIN_TRAPEZOID_SCORE = 0.20
DEFAULT_BALL_START_FRAME = 60
API_DEFAULT_BALL_START_FRAME = 40
API_DEFAULT_MAX_VIDEO_FRAMES = 600
API_DEFAULT_MAX_VIDEO_DIMENSION = 1024

logger = logging.getLogger("pipeline_v2.lane_ball_overlay")
YOLO_CLASS_NAME_BY_ID: Dict[int, str] = {0: "ball", 1: "lane", 2: "pins"}
MAX_LANE_MASKS_PER_FRAME = 4


@dataclass
class BallPos:
    frame_index: int
    timestamp_s: float
    x_m: float
    y_m: float


@dataclass
class BallPosList:
    ball_positions: List[BallPos] = field(default_factory=list)


@dataclass
class QuarterKinematics:
    quarter: int
    start_m: float
    end_m: float
    mean_speed_mps: float
    mean_acceleration_mps2: float
    sample_count: int


@dataclass
class Kinematics:
    quarters: List[QuarterKinematics] = field(default_factory=list)


@dataclass(frozen=True)
class InferencePreprocessConfig:
    imgsz: int = 1024
    conf: float = 0.10
    iou: float = 0.50


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
    ball_positions: BallPosList
    homography_selection: HomographySelection
    health: PostprocessHealth


@dataclass(frozen=True)
class TrimDiagnostics:
    kept_positions: List[BallPos]
    cut_reason: Optional[str]
    cut_frame_index: Optional[int]
    last_kept_frame_index: Optional[int]
    current_dy: Optional[float]
    median_dy: Optional[float]
    cut_x_m: Optional[float]
    cut_y_m: Optional[float]


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
    geometry_score: float = 0.0
    support_score: float = 0.0
    width_ratio: float = 0.0
    bottom_width: float = 0.0
    mask_area: float = 0.0


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
    total_score: float = 0.0
    total_geometry_score: float = 0.0
    total_support_score: float = 0.0
    best_geometry_score: float = 0.0
    best_support_score: float = 0.0


@dataclass(frozen=True)
class LaneGeometryObservation:
    frame_index: int
    polygon: np.ndarray
    score: float
    centroid_x: float
    pins_box: np.ndarray | None
    geometry_score: float
    support_score: float
    bottom_width: float
    mask_area: float


@dataclass(frozen=True)
class LaneTrackSelection:
    track: TrackedLane
    src_corners: np.ndarray
    matched_observations: List[LaneGeometryObservation]
    lane_quality: float
    ball_score: float
    ball_positions: List[BallPos]


class TemporalSmoother:
    """
    Exponential moving average on lane corner positions.
    Lanes are matched by centroid-x proximity and outlier corner jumps are rejected.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        max_match_dist: float = 80.0,
        max_corner_jump: float = 60.0,
        stale_frames: int = 100_000,
    ):
        self.ema_alpha = ema_alpha
        self.max_match_dist = max_match_dist
        self.max_corner_jump = max_corner_jump
        self.stale_frames = stale_frames
        self.tracks: List[TrackedLane] = []

    def update(
        self,
        candidates: List[TrapezoidCandidate],
        frame_idx: int,
        ball_boxes: Optional[List[np.ndarray]] = None,
        lane_mask_areas: Optional[List[float]] = None,
    ) -> None:
        used_tracks: set[int] = set()
        cand_to_track: List[int] = []

        for ci, cand in enumerate(candidates):
            cx = float(np.mean(cand.polygon[:, 0]))

            best_ti: Optional[int] = None
            best_dist = float("inf")
            for ti, track in enumerate(self.tracks):
                if ti in used_tracks:
                    continue
                d = abs(cx - track.centroid_x)
                if d < best_dist and d < self.max_match_dist:
                    best_dist = d
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
                accepted_geometry_update = max_jump < self.max_corner_jump

                if accepted_geometry_update:
                    a = self.ema_alpha
                    blended = (
                        a * cand.polygon.astype(float)
                        + (1.0 - a) * track.quad.astype(float)
                    )
                    track.quad = blended.astype(np.int32)
                    if cand.score > track.best_score:
                        track.best_score = float(cand.score)
                        track.best_quad = cand.polygon.copy()
                        track.best_frame_idx = int(frame_idx)
                        track.best_geometry_score = float(cand.geometry_score)
                        track.best_support_score = float(cand.support_score)

                track.centroid_x = float(np.mean(track.quad[:, 0]))
                track.last_seen_frame = frame_idx
                track.seen_count += 1
                if lane_mask_areas is not None and ci < len(lane_mask_areas):
                    track.total_mask_area += float(lane_mask_areas[ci])
                track.total_score += float(cand.score)
                track.total_geometry_score += float(cand.geometry_score)
                track.total_support_score += float(cand.support_score)
                if cand.score > track.best_score and accepted_geometry_update:
                    track.best_score = float(cand.score)
                    track.best_quad = cand.polygon.copy()
                    track.best_frame_idx = int(frame_idx)
                    track.best_geometry_score = float(cand.geometry_score)
                    track.best_support_score = float(cand.support_score)
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
                        ball_votes=0,
                        total_mask_area=area,
                        seen_count=1,
                        total_score=float(cand.score),
                        total_geometry_score=float(cand.geometry_score),
                        total_support_score=float(cand.support_score),
                        best_geometry_score=float(cand.geometry_score),
                        best_support_score=float(cand.support_score),
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

        return max(
            self.tracks,
            key=self._lane_quality_key,
        )

    @staticmethod
    def _lane_quality_key(track: TrackedLane) -> tuple[float, float, float, float]:
        seen = max(track.seen_count, 1)
        avg_score = track.total_score / seen
        avg_geometry = track.total_geometry_score / seen
        avg_support = track.total_support_score / seen
        avg_area = track.total_mask_area / seen
        ball_tie_break = min(track.ball_votes, 12) / 12.0
        return (
            0.45 * avg_score
            + 0.25 * avg_geometry
            + 0.20 * avg_support
            + 0.10 * min(seen / 12.0, 1.0)
            + 0.03 * ball_tie_break,
            avg_area,
            float(seen),
            float(track.ball_votes),
        )

    def summary(self) -> str:
        parts: List[str] = []
        for i, t in enumerate(self.tracks):
            avg_area = t.total_mask_area / max(t.seen_count, 1)
            parts.append(
                f"  lane track {i}: best_score={t.best_score:.3f} "
                f"geom={t.best_geometry_score:.2f} support={t.best_support_score:.2f} "
                f"best_frame={t.best_frame_idx} "
                f"ball_votes={t.ball_votes} avg_area={avg_area:.0f} "
                f"seen={t.seen_count}"
            )
        return "\n".join(parts) if parts else "  (no tracked lanes)"


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def to_model_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H,W,3), got {frame_bgr.shape}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def normalize_model_names(raw_names: Mapping[Any, Any] | None) -> Dict[int, str]:
    if not isinstance(raw_names, Mapping):
        return dict(YOLO_CLASS_NAME_BY_ID)

    out: Dict[int, str] = {}
    for key, value in raw_names.items():
        try:
            cls_id = int(key)
        except Exception:
            continue
        out[cls_id] = str(value).strip().lower()
    return out or dict(YOLO_CLASS_NAME_BY_ID)

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


def _resize_if_needed(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None or max_dim <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = float(max_dim) / float(longest)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _read_frames(
    video_path: str,
    *,
    max_frames: int | None = None,
    max_dimension: int | None = None,
) -> SplitVideo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at path: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_cap = max_frames if max_frames is not None and max_frames > 0 else None
    step = max(1, total_frames // frame_cap) if frame_cap and total_frames > frame_cap else 1
    effective_fps = fps / step if step > 1 else fps
    split_video = SplitVideo(fps=effective_fps, width=source_width, height=source_height)

    frame_index = 0
    kept = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            frame = _resize_if_needed(frame, max_dimension)
            if kept == 0:
                split_video.height, split_video.width = frame.shape[:2]
            timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            split_video.add_frame(
                VideoFrame(frame_index=kept, timestamp=timestamp_s, image=frame)
            )
            kept += 1
            if frame_cap and kept >= frame_cap:
                break
        frame_index += 1

    cap.release()
    return split_video


def split_video_into_frames(
    video_path: str,
    *,
    max_frames: int | None = None,
    max_dimension: int | None = None,
) -> SplitVideo:
    """
    Read video frames. If OpenCV can't decode the pixel format (e.g. YUV411P
    interlaced AVI), falls back to ffmpeg transcoding to a temp mp4 first.
    """
    split_video = _read_frames(
        video_path,
        max_frames=max_frames,
        max_dimension=max_dimension,
    )

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
                split_video = _read_frames(
                    tmp_path,
                    max_frames=max_frames,
                    max_dimension=max_dimension,
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    return split_video


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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
    frame_idx: int = 0,
    color: tuple[int, int, int] | None = None,
) -> None:
    if color is None:
        color = TRAPEZOID_COLORS[lane_index % len(TRAPEZOID_COLORS)]

    poly = trapezoid.polygon.reshape((-1, 1, 2))

    # Semi-transparent fill.
    fill_overlay = image.copy()
    cv2.fillPoly(fill_overlay, [trapezoid.polygon.reshape((-1, 1, 2))], color)
    cv2.addWeighted(fill_overlay, 0.2, image, 0.8, 0, dst=image)

    # Thick outline.
    cv2.polylines(image, [poly], True, color, 3, cv2.LINE_AA)

    # Corner labels reflect screen position, not lane semantics.
    corner_labels = ["TOP-L", "TOP-R", "BOT-R", "BOT-L"]
    for pt_idx in range(4):
        pt = tuple(trapezoid.polygon[pt_idx].tolist())
        cv2.circle(image, pt, 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            image, corner_labels[pt_idx],
            (pt[0] + 8, pt[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    tl = tuple(trapezoid.polygon[0].tolist())
    text = (
        f"lane{lane_index} f{frame_idx} s={trapezoid.score:.2f} "
        f"c={trapezoid.coverage:.2f} p={trapezoid.purity:.2f}"
    )
    cv2.putText(
        image, text,
        (int(tl[0]), max(24, int(tl[1]) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Mask cleanup
# ---------------------------------------------------------------------------

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


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    [Improvement 5] Isolate the largest connected component via
    cv2.connectedComponentsWithStats. Prevents the convex hull from
    inflating around stray blobs from neighboring lanes or gutters.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # Label 0 is background; find largest foreground component.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def keep_significant_components(
    mask: np.ndarray,
    *,
    min_area_px: int = 300,
    min_largest_ratio: float = 0.03,
) -> np.ndarray:
    """
    Keep all meaningful lane components instead of only the largest component.
    A bowler can split a good lane mask into multiple pieces; tiny stray blobs
    are the parts we want to remove.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    largest = float(np.max(areas)) if areas.size else 0.0
    threshold = max(float(min_area_px), largest * float(min_largest_ratio))
    kept = np.zeros(mask.shape, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= threshold:
            kept[labels == label_idx] = 1
    return kept


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest contour by area from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# Quad extraction
# ---------------------------------------------------------------------------

def approx_to_quad(contour: np.ndarray) -> np.ndarray | None:
    """
    Adaptive approxPolyDP: binary-search epsilon until we get exactly 4 vertices.
    Works on the convex hull of the contour to ignore concavities from bubbly masks.
    """
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

    # Accept 4-6 point results and reduce to 4 by dropping least-area vertex.
    if best_approx is not None and 4 <= len(best_approx) <= 6:
        pts = best_approx.reshape(-1, 2).astype(np.float32)
        while len(pts) > 4:
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
    Order 4 points strictly by image position:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = pts.astype(np.float32)
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.int32)


def hough_quad_fallback(mask: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
    """
    Fallback: draw the convex hull edges, run HoughLines, take the 4 strongest
    non-duplicate lines, and intersect them for corners.
    """
    hull = cv2.convexHull(contour)
    h, w = mask.shape
    edge_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_img, [hull], 0, 255, 2)

    min_dim = min(h, w)
    lines = cv2.HoughLines(edge_img, rho=1, theta=np.pi / 180, threshold=max(30, min_dim // 8))
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
    moments = cv2.moments(hull)
    if moments["m00"] == 0:
        return None
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    angles = np.arctan2(corners_np[:, 1] - cy, corners_np[:, 0] - cx)
    sorted_idx = np.argsort(angles)
    if len(sorted_idx) >= 4:
        step = len(sorted_idx) / 4.0
        picks = [sorted_idx[int(i * step)] for i in range(4)]
        return corners_np[picks].astype(np.int32)

    return None


# ---------------------------------------------------------------------------
# Edge refinement from raw image
# ---------------------------------------------------------------------------

def refine_sides_from_image(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    """
    [Improvement 2] Use Canny + HoughLinesP on the raw BGR frame in a narrow
    strip along each side edge of the quad. The gutter boundaries are
    high-contrast edges in the raw image and give much sharper side-line
    estimates than anything derived from the mask alone.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    refined = quad.copy().astype(np.float32)

    # Side edges: TL→BL (left), TR→BR (right).
    sides = [(0, 3), (1, 2)]

    for top_idx, bot_idx in sides:
        pt_top = refined[top_idx]
        pt_bot = refined[bot_idx]

        dx = pt_bot[0] - pt_top[0]
        dy = pt_bot[1] - pt_top[1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length < 20:
            continue

        # Perpendicular direction.
        nx, ny = -dy / length, dx / length
        strip_width = 25.0

        # Build a strip polygon around the side edge.
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
            strip_edges, 1, np.pi / 180,
            threshold=30, minLineLength=min_line_len, maxLineGap=20,
        )
        if lines is None or len(lines) == 0:
            continue

        # Pick the longest near-vertical segment.
        best_line = None
        best_len = 0.0
        for seg in lines:
            sx1, sy1, sx2, sy2 = seg[0]
            sdx, sdy = abs(sx2 - sx1), abs(sy2 - sy1)
            if sdy < sdx:
                continue  # skip more-horizontal lines
            seg_len = np.sqrt(float(sdx ** 2 + sdy ** 2))
            if seg_len > best_len:
                best_len = seg_len
                best_line = seg[0]

        if best_line is None:
            continue

        lx1, ly1, lx2, ly2 = best_line
        if abs(ly2 - ly1) < 1:
            continue

        # Fit x = a*y + b and extrapolate to quad top/bottom y.
        a = (lx2 - lx1) / (ly2 - ly1)
        b = lx1 - a * ly1

        new_top_x = a * refined[top_idx, 1] + b
        new_bot_x = a * refined[bot_idx, 1] + b

        if abs(new_top_x - refined[top_idx, 0]) < strip_width:
            refined[top_idx, 0] = new_top_x
        if abs(new_bot_x - refined[bot_idx, 0]) < strip_width:
            refined[bot_idx, 0] = new_bot_x

    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)
    return refined.astype(np.int32)


# ---------------------------------------------------------------------------
# Geometric validation
# ---------------------------------------------------------------------------

def validate_vanishing_point(
    quad: np.ndarray,
    img_shape_hw: tuple[int, int],
) -> bool:
    """
    [Improvement 4] Check that the two side edges (TL→BL and TR→BR) converge
    toward a vanishing point that is above the image and roughly centered
    horizontally. This enforces perspective geometry — parallel lane edges
    must converge upward in a camera view.
    """
    h, w = img_shape_hw
    q = quad.astype(np.float64)
    tl, tr, br, bl = q[0], q[1], q[2], q[3]

    # Left side direction: TL → BL (downward in image).
    left_dx = bl[0] - tl[0]
    left_dy = bl[1] - tl[1]
    # Right side direction: TR → BR (downward in image).
    right_dx = br[0] - tr[0]
    right_dy = br[1] - tr[1]

    # Intersect the two side lines (extended upward).
    # Line 1: tl + t * (left_dx, left_dy)
    # Line 2: tr + s * (right_dx, right_dy)
    det = left_dx * right_dy - right_dx * left_dy
    if abs(det) < 1e-6:
        # Parallel sides — acceptable for a far-away view.
        return True

    t = ((tr[0] - tl[0]) * right_dy - (tr[1] - tl[1]) * right_dx) / det
    vp_x = tl[0] + t * left_dx
    vp_y = tl[1] + t * left_dy

    # Vanishing point should be above the image (negative t means upward from TL).
    if vp_y > tl[1]:
        return False

    # Should be roughly centered — not wildly off to one side.
    if vp_x < -w or vp_x > 2 * w:
        return False

    return True


def _robust_fit_x_of_y(points_xy: np.ndarray) -> tuple[float, float, float] | None:
    """Fit x = a*y + b with simple MAD-based outlier rejection."""
    if points_xy.shape[0] < 8:
        return None

    pts = points_xy.astype(np.float64)
    keep = np.ones(pts.shape[0], dtype=bool)
    coeff: np.ndarray | None = None

    for _ in range(4):
        if int(np.count_nonzero(keep)) < 8:
            return None
        y = pts[keep, 1]
        x = pts[keep, 0]
        coeff = np.polyfit(y, x, 1)
        pred = coeff[0] * pts[:, 1] + coeff[1]
        residual = np.abs(pts[:, 0] - pred)
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        threshold = max(8.0, med + 3.0 * 1.4826 * max(mad, 1.0))
        keep = residual <= threshold

    if coeff is None:
        return None
    inlier_ratio = float(np.count_nonzero(keep) / max(points_xy.shape[0], 1))
    return float(coeff[0]), float(coeff[1]), inlier_ratio


def _x_at_y(line: tuple[float, float, float] | tuple[float, float], y: float) -> float:
    return float(line[0] * y + line[1])


def _line_x_of_y_from_points(
    p0: np.ndarray,
    p1: np.ndarray,
) -> tuple[float, float] | None:
    dy = float(p1[1] - p0[1])
    if abs(dy) < 1e-6:
        return None
    slope = float(p1[0] - p0[0]) / dy
    intercept = float(p0[0]) - slope * float(p0[1])
    return slope, intercept


def _mask_boundary_xs_near_y(
    mask: np.ndarray,
    y: float,
    *,
    half_window: int = 4,
) -> tuple[float, float] | None:
    """Return robust left/right mask boundaries near a scanline."""
    y_center = int(round(float(y)))
    y0 = max(0, y_center - half_window)
    y1 = min(mask.shape[0], y_center + half_window + 1)
    xs = np.where(mask[y0:y1, :] > 0)[1]
    if xs.size < 8:
        return None
    return float(np.percentile(xs, 0.5)), float(np.percentile(xs, 99.5))


def _scanline_boundary_samples(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract robust left/right lane boundary samples from dense segmentation.
    Percentiles are used instead of min/max so small edge bubbles do not steer
    the lane model.
    """
    ys, _ = np.where(mask > 0)
    if ys.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32)

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    if y_max - y_min < 20:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros((0,), dtype=np.float32)

    sample_rows = np.unique(np.linspace(y_min, y_max, num=90).astype(np.int32))
    left_pts: List[tuple[float, float]] = []
    right_pts: List[tuple[float, float]] = []
    widths: List[float] = []

    for y in sample_rows:
        y0 = max(0, int(y) - 1)
        y1 = min(mask.shape[0], int(y) + 2)
        xs = np.where(mask[y0:y1, :] > 0)[1]
        if xs.size < 12:
            continue

        left_x = float(np.percentile(xs, 1.0))
        right_x = float(np.percentile(xs, 99.0))
        width = right_x - left_x
        if width < 10:
            continue

        left_pts.append((left_x, float(y)))
        right_pts.append((right_x, float(y)))
        widths.append(width)

    return (
        np.asarray(left_pts, dtype=np.float32),
        np.asarray(right_pts, dtype=np.float32),
        np.asarray(widths, dtype=np.float32),
    )


def _geometry_scores(
    polygon: np.ndarray,
    img_shape_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """
    Return geometry_score, width_ratio(top/bottom), bottom_width, y_span_ratio.
    A real lane viewed from the approach should be wider near the camera than
    at the pins.
    """
    h, w = img_shape_hw
    q = polygon.astype(np.float64)
    width_top = abs(float(q[1, 0] - q[0, 0]))
    width_bottom = abs(float(q[2, 0] - q[3, 0]))
    if width_top <= 0.0 or width_bottom <= 0.0:
        return 0.0, 999.0, width_bottom, 0.0

    width_ratio = width_top / max(width_bottom, 1.0)
    y_span = float(max(q[2, 1], q[3, 1]) - min(q[0, 1], q[1, 1]))
    y_span_ratio = y_span / max(float(h), 1.0)

    taper_score = 1.0 - min(abs(width_ratio - 0.45) / 0.45, 1.0)
    if width_ratio >= 0.98:
        taper_score = 0.0

    # Some bowling cameras only see the far lane/pin-deck portion; require
    # perspective-consistent taper, but do not demand that the lane occupy most
    # of the frame height.
    bottom_width_score = min(width_bottom / max(0.06 * float(w), 1.0), 1.0)
    span_score = min(y_span_ratio / 0.16, 1.0)

    center_top = 0.5 * (q[0, 0] + q[1, 0])
    center_bottom = 0.5 * (q[3, 0] + q[2, 0])
    center_shift = abs(center_top - center_bottom) / max(width_bottom, 1.0)
    symmetry_score = max(0.0, 1.0 - 1.8 * center_shift)

    vp_score = 1.0 if validate_vanishing_point(polygon, img_shape_hw) else 0.0
    geometry = (
        0.35 * taper_score
        + 0.25 * bottom_width_score
        + 0.20 * span_score
        + 0.10 * symmetry_score
        + 0.10 * vp_score
    )
    return float(np.clip(geometry, 0.0, 1.0)), float(width_ratio), float(width_bottom), float(y_span_ratio)


def _lane_band_support_score(
    lane_mask: np.ndarray,
    polygon: np.ndarray,
    *,
    bands: int = 14,
) -> float:
    trap_mask = np.zeros(lane_mask.shape, dtype=np.uint8)
    cv2.fillPoly(trap_mask, [polygon.astype(np.int32)], 1)

    q = polygon.astype(np.float64)
    y0 = int(max(0, np.floor(min(q[0, 1], q[1, 1]))))
    y1 = int(min(lane_mask.shape[0] - 1, np.ceil(max(q[2, 1], q[3, 1]))))
    if y1 <= y0:
        return 0.0

    support_values: List[float] = []
    for a, b in zip(
        np.linspace(y0, y1, bands + 1)[:-1],
        np.linspace(y0, y1, bands + 1)[1:],
    ):
        yy0 = int(max(0, np.floor(a)))
        yy1 = int(min(lane_mask.shape[0], np.ceil(b)))
        if yy1 <= yy0:
            continue
        band_trap = trap_mask[yy0:yy1, :] > 0
        trap_area = int(np.count_nonzero(band_trap))
        if trap_area <= 0:
            continue
        band_lane = lane_mask[yy0:yy1, :] > 0
        fill = float(np.count_nonzero(band_trap & band_lane)) / float(trap_area)
        support_values.append(min(fill / 0.55, 1.0))

    if not support_values:
        return 0.0
    support = float(np.mean(support_values))
    populated = float(np.count_nonzero(np.asarray(support_values) > 0.20)) / float(len(support_values))
    return float(np.clip(0.65 * support + 0.35 * populated, 0.0, 1.0))


def _fit_quad_from_dense_mask(
    lane_mask: np.ndarray,
    *,
    reject_counts: Optional[Dict[str, int]] = None,
) -> np.ndarray | None:
    def reject(reason: str) -> None:
        if reject_counts is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    left_pts, right_pts, widths = _scanline_boundary_samples(lane_mask)
    if left_pts.shape[0] < 10 or right_pts.shape[0] < 10:
        reject("dense_few_scanline_samples")
        return None

    left_line = _robust_fit_x_of_y(left_pts)
    right_line = _robust_fit_x_of_y(right_pts)
    if left_line is None or right_line is None:
        reject("dense_line_fit_failed")
        return None
    if min(left_line[2], right_line[2]) < 0.55:
        reject("dense_low_line_inliers")
        return None

    all_y = np.concatenate([left_pts[:, 1], right_pts[:, 1]])
    top_y = float(np.percentile(all_y, 2.0))
    bottom_y = float(np.percentile(all_y, 99.5))

    if bottom_y - top_y < lane_mask.shape[0] * 0.045:
        reject("dense_short_y_span")
        return None

    top_boundary = _mask_boundary_xs_near_y(lane_mask, top_y)
    bottom_boundary = _mask_boundary_xs_near_y(lane_mask, bottom_y)
    if top_boundary is not None:
        tl_x, tr_x = top_boundary
    else:
        tl_x = _x_at_y(left_line, top_y)
        tr_x = _x_at_y(right_line, top_y)
    if bottom_boundary is not None:
        bl_x, br_x = bottom_boundary
    else:
        bl_x = _x_at_y(left_line, bottom_y)
        br_x = _x_at_y(right_line, bottom_y)

    if not np.all(np.isfinite([tl_x, tr_x, br_x, bl_x])):
        reject("dense_nonfinite_quad")
        return None
    if tr_x <= tl_x or br_x <= bl_x:
        reject("dense_invalid_width_order")
        return None

    quad = np.array(
        [
            [tl_x, top_y],
            [tr_x, top_y],
            [br_x, bottom_y],
            [bl_x, bottom_y],
        ],
        dtype=np.float32,
    )
    quad[:, 0] = np.clip(quad[:, 0], 0, lane_mask.shape[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, lane_mask.shape[0] - 1)

    if widths.size >= 8:
        width_top = float(tr_x - tl_x)
        width_bottom = float(br_x - bl_x)
        observed_growth = float(np.percentile(widths, 85.0) - np.percentile(widths, 15.0))
        if width_bottom <= width_top and observed_growth > 5.0:
            reject("dense_bottom_not_wider")
            return None

    return quad.astype(np.int32)


def _polygon_area_px(polygon: np.ndarray) -> float:
    return float(abs(cv2.contourArea(polygon.reshape((-1, 1, 2)).astype(np.float32))))


def _penalize_nested_lane_subsets(candidates: List[TrapezoidCandidate]) -> None:
    """
    Down-rank a small lane-like subset that sits inside a larger plausible lane.
    This specifically targets ball-path/occlusion artifacts that otherwise win
    because the ball boxes fall inside them.
    """
    if len(candidates) < 2:
        return

    areas = [_polygon_area_px(c.polygon) for c in candidates]
    for i, cand in enumerate(candidates):
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if areas[j] < areas[i] * 1.35:
                continue
            inside = 0
            for pt in cand.polygon.astype(np.float32):
                if cv2.pointPolygonTest(
                    other.polygon.reshape((-1, 1, 2)).astype(np.float32),
                    (float(pt[0]), float(pt[1])),
                    False,
                ) >= 0:
                    inside += 1
            if inside >= 3:
                cand.score *= 0.45
                cand.geometry_score *= 0.70
                cand.support_score *= 0.85
                break


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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
    geometry_score, _, _, _ = _geometry_scores(polygon, lane_mask.shape)
    support_score = _lane_band_support_score(lane_mask, polygon)
    overlap_score = score_coverage_weight * coverage + (1.0 - score_coverage_weight) * purity
    score = 0.55 * overlap_score + 0.25 * geometry_score + 0.20 * support_score
    return coverage, purity, score


# ---------------------------------------------------------------------------
# Pins anchoring
# ---------------------------------------------------------------------------

def find_nearest_pins_box(
    lane_mask: np.ndarray,
    pins_boxes: List[np.ndarray],
) -> np.ndarray | None:
    """
    Find the pins bbox whose center-x is closest to the lane mask centroid-x.
    """
    if not pins_boxes:
        return None

    ys, xs = np.where(lane_mask > 0)
    if len(xs) == 0:
        return None
    lane_cx = float(np.mean(xs))

    best_box = None
    best_dist = float("inf")
    for box in pins_boxes:
        pins_cx = (box[0] + box[2]) / 2.0
        dist = abs(pins_cx - lane_cx)
        if dist < best_dist:
            best_dist = dist
            best_box = box

    return best_box


def find_nearest_pins_box_for_polygon(
    lane_polygon: np.ndarray,
    pins_boxes: List[np.ndarray],
) -> np.ndarray | None:
    if not pins_boxes:
        return None

    lane_cx = float(np.mean(lane_polygon[:, 0]))
    best_box = None
    best_dist = float("inf")
    for box in pins_boxes:
        pins_cx = float(box[0] + box[2]) / 2.0
        dist = abs(pins_cx - lane_cx)
        if dist < best_dist:
            best_dist = dist
            best_box = box
    return best_box


def build_lane_geometry_observation(
    frame_index: int,
    candidate: TrapezoidCandidate,
    pins_boxes: List[np.ndarray],
) -> LaneGeometryObservation:
    pins_box = find_nearest_pins_box_for_polygon(candidate.polygon, pins_boxes)
    return LaneGeometryObservation(
        frame_index=int(frame_index),
        polygon=candidate.polygon.copy(),
        score=float(candidate.score),
        centroid_x=float(np.mean(candidate.polygon[:, 0])),
        pins_box=(pins_box.copy() if pins_box is not None else None),
        geometry_score=float(candidate.geometry_score),
        support_score=float(candidate.support_score),
        bottom_width=float(candidate.bottom_width),
        mask_area=float(candidate.mask_area),
    )


def _aggregate_pins_box_from_observations(
    observations: Sequence[LaneGeometryObservation],
) -> np.ndarray | None:
    pins = [obs.pins_box for obs in observations if obs.pins_box is not None]
    if not pins:
        return None
    return np.median(np.asarray(pins, dtype=np.float32), axis=0).astype(np.int32)


def _aggregate_lane_quad_from_observations(
    observations: Sequence[LaneGeometryObservation],
    img_shape_hw: tuple[int, int],
) -> np.ndarray | None:
    """
    Estimate one stable lane quad from many accepted observations.

    The segmentation model is treated as the primary source of truth here:
    aggregate observed corner coordinates directly instead of re-fitting side
    lines or pulling the top edge toward pins/vanishing geometry. Geometry is
    only used afterward as a sanity check.
    """
    if len(observations) < 3:
        return None

    ranked = sorted(
        observations,
        key=lambda obs: (
            obs.score
            + 0.35 * obs.geometry_score
            + 0.25 * obs.support_score
            + 0.10 * min(obs.mask_area / 100_000.0, 1.0)
        ),
        reverse=True,
    )[:40]

    quads = np.asarray([obs.polygon for obs in ranked], dtype=np.float32)
    quad = np.median(quads, axis=0).astype(np.float32)

    # Keep the top and bottom edges level enough for a stable homography, but
    # trust the outer segmentation envelope at the bottom. Bowler occlusion
    # often creates subset masks after release; using a median bottom corner
    # lets those subsets shrink the lane into the ball path.
    top_y = float(np.median(0.5 * (quads[:, 0, 1] + quads[:, 1, 1])))
    bottom_y = float(np.median(0.5 * (quads[:, 3, 1] + quads[:, 2, 1])))
    quad[0, 1] = top_y
    quad[1, 1] = top_y
    quad[2, 1] = bottom_y
    quad[3, 1] = bottom_y

    top_left_xs: List[float] = []
    top_right_xs: List[float] = []
    bottom_left_xs: List[float] = []
    bottom_right_xs: List[float] = []
    for q in quads:
        left_line = _line_x_of_y_from_points(q[0], q[3])
        right_line = _line_x_of_y_from_points(q[1], q[2])
        if left_line is None or right_line is None:
            continue
        top_left_xs.append(_x_at_y(left_line, top_y))
        top_right_xs.append(_x_at_y(right_line, top_y))
        bottom_left_xs.append(_x_at_y(left_line, bottom_y))
        bottom_right_xs.append(_x_at_y(right_line, bottom_y))

    if len(bottom_left_xs) >= 3:
        quad[0, 0] = float(np.median(top_left_xs))
        quad[1, 0] = float(np.median(top_right_xs))
        quad[3, 0] = float(np.percentile(bottom_left_xs, 10.0))
        quad[2, 0] = float(np.percentile(bottom_right_xs, 90.0))

    quad[:, 0] = np.clip(quad[:, 0], 0, img_shape_hw[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, img_shape_hw[0] - 1)

    geometry_score, width_ratio, _, y_span_ratio = _geometry_scores(quad, img_shape_hw)
    if width_ratio >= 0.98 or geometry_score < 0.16 or y_span_ratio < 0.045:
        return None
    return quad.astype(np.int32)


# ---------------------------------------------------------------------------
# Lane homography helpers
# ---------------------------------------------------------------------------

def _lane_dst_corners_m() -> np.ndarray:
    """Destination rectangle in metres: TL, TR, BR, BL."""
    return np.array(
        [
            [0.0, LANE_LENGTH_M],
            [LANE_WIDTH_M, LANE_LENGTH_M],
            [LANE_WIDTH_M, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _masks_to_boxes(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Convert binary masks to [x1, y1, x2, y2] bounding boxes."""
    boxes: List[np.ndarray] = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))
    return boxes


def _ball_contact_point_from_mask(ball_mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(ball_mask > 0)
    if xs.size == 0:
        return None

    y_contact = float(np.max(ys))
    x_contact = float(np.min(xs) + np.max(xs)) / 2.0
    return x_contact, y_contact


def _choose_ball_contact_for_lane(
    ball_masks: List[np.ndarray],
    lane_polygon: np.ndarray,
) -> tuple[float, float] | None:
    """
    Pick the best ball contact point for a given lane polygon.

    Priority (lower tuple key is better):
      1. Ball inside the lane polygon over outside.
      2. Higher y (closer to camera / further down the lane).
      3. Closer to the lane centre-x.
      4. Larger mask area.
    """
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


def _project_point_homography(
    pt_xy: tuple[float, float], H: np.ndarray,
) -> tuple[float, float]:
    """Project a pixel-space point through a 3×3 homography to lane metres."""
    vec = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    dst = H @ vec
    if abs(float(dst[2])) < 1e-9:
        raise RuntimeError("Invalid homography projection: denominator ~0")
    x = float(dst[0] / dst[2])
    y = float(dst[1] / dst[2])
    return x, y


def _project_ball_positions_for_lane(
    segmentations_by_frame: Dict[int, FrameSegmentation],
    fps: float,
    ball_start_frame: int,
    src_corners: np.ndarray,
    homography: np.ndarray,
) -> List[BallPos]:
    positions: List[BallPos] = []
    for frame_index in sorted(segmentations_by_frame.keys()):
        if frame_index < ball_start_frame:
            continue

        frame_seg = segmentations_by_frame[frame_index]
        contact = _choose_ball_contact_for_lane(frame_seg.ball_masks, src_corners)
        if contact is None:
            continue

        x_m, y_m = _project_point_homography(contact, homography)
        positions.append(
            BallPos(
                frame_index=int(frame_index),
                timestamp_s=float(frame_index / max(fps, 1e-6)),
                x_m=float(x_m),
                y_m=float(y_m),
            )
        )
    return positions


def _ball_path_score(
    positions: List[BallPos],
    *,
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> float:
    if not positions:
        return 0.0

    sorted_pos = sorted(positions, key=lambda p: p.frame_index)
    in_bounds = [
        p for p in sorted_pos
        if _position_in_lane_bounds(p, lane_width_m, lane_length_m)
    ]
    if not in_bounds:
        return 0.0

    start_idx, end_idx = _best_ball_motion_interval(
        sorted_pos,
        lane_width_m=lane_width_m,
        lane_length_m=lane_length_m,
    )
    interval = sorted_pos[start_idx:end_idx]
    interval_in_bounds = [
        p for p in interval
        if _position_in_lane_bounds(p, lane_width_m, lane_length_m)
    ]
    if not interval_in_bounds:
        return 0.0

    y_span = max(p.y_m for p in interval_in_bounds) - min(p.y_m for p in interval_in_bounds)
    frame_span = interval_in_bounds[-1].frame_index - interval_in_bounds[0].frame_index + 1
    valid_ratio = len(in_bounds) / max(len(sorted_pos), 1)
    interval_ratio = len(interval_in_bounds) / max(len(sorted_pos), 1)
    span_score = min(max(y_span, 0.0) / 4.0, 1.0)
    duration_score = min(frame_span / 16.0, 1.0)

    return float(
        np.clip(
            0.35 * valid_ratio
            + 0.30 * interval_ratio
            + 0.20 * span_score
            + 0.15 * duration_score,
            0.0,
            1.0,
        )
    )


def _track_average_area(track: TrackedLane) -> float:
    return float(track.total_mask_area / max(track.seen_count, 1))


def _quad_bottom_width_px(quad: np.ndarray) -> float:
    q = quad.astype(np.float64)
    return abs(float(q[2, 0] - q[3, 0]))


def _quad_top_center_x(quad: np.ndarray) -> float:
    q = quad.astype(np.float64)
    return 0.5 * float(q[0, 0] + q[1, 0])


def _quad_centroid_x(quad: np.ndarray) -> float:
    return float(np.mean(quad.astype(np.float64)[:, 0]))


def _observation_matches_track(
    obs: LaneGeometryObservation,
    track: TrackedLane,
) -> bool:
    """
    Associate occluded/subset lane detections with the same physical lane.
    The far/top edge is the stable lane identity cue; bottom/centroid can shift
    substantially when the bowler hides part of the near lane.
    """
    obs_top_cx = _quad_top_center_x(obs.polygon)
    track_top_cx = _quad_top_center_x(track.best_quad)
    if abs(obs_top_cx - track_top_cx) <= 55.0:
        return True

    obs_q = obs.polygon.astype(np.float64)
    track_q = track.best_quad.astype(np.float64)
    same_right_edge = (
        abs(float(obs_q[1, 0] - track_q[1, 0])) <= 65.0
        and abs(float(obs_q[2, 0] - track_q[2, 0])) <= 45.0
    )
    if same_right_edge:
        return True

    centroid_tol = max(90.0, track.best_support_score * 140.0)
    return abs(obs.centroid_x - _quad_centroid_x(track.best_quad)) <= centroid_tol


def _lane_track_selection_metrics(
    selection: LaneTrackSelection,
    *,
    max_avg_area: float,
    max_bottom_width: float,
) -> tuple[float, float, float, float]:
    """
    Score a candidate homography lane without letting ball overlap rescue a
    lane-like sliver. Ball support is useful for resolving adjacent lanes, but
    only after the candidate has a credible footprint compared with peers.
    """
    avg_area = _track_average_area(selection.track)
    bottom_width = _quad_bottom_width_px(selection.src_corners)

    area_score = (
        min(avg_area / max(0.60 * max_avg_area, 1.0), 1.0)
        if max_avg_area > 0.0
        else 0.0
    )
    width_score = (
        min(bottom_width / max(0.70 * max_bottom_width, 1.0), 1.0)
        if max_bottom_width > 0.0
        else 0.0
    )
    footprint_score = min(area_score, width_score)
    ball_gate = 1.0 if footprint_score >= 0.65 else 0.35 * footprint_score
    gated_ball_score = selection.ball_score * ball_gate

    score = float(
        np.clip(
            0.50 * selection.lane_quality
            + 0.20 * gated_ball_score
            + 0.18 * area_score
            + 0.12 * width_score,
            0.0,
            1.0,
        )
    )
    return score, area_score, width_score, gated_ball_score


def _correct_trapezoid_top_corners(
    quad: np.ndarray,
) -> np.ndarray:
    """
    Snap each far-end (top) corner onto its vanishing ray to correct
    lateral drift without changing its depth along the ray.

    The bottom corners (close to camera) are trusted.  The top corners
    (far from camera, near the pins) can drift laterally off the true
    leg line.  This projects each top corner onto the line from its
    corresponding bottom corner through the vanishing point.

    Each corner keeps its own depth (λ) — we do NOT average them.
    Averaging would break a corner that was already correct by pulling
    it toward one that was wrong.

    Corner order: [tl, tr, br, bl]  (same as order_quad_points output).
    """
    quad_f = quad.astype(np.float64)
    tl, tr, br, bl = quad_f[0], quad_f[1], quad_f[2], quad_f[3]

    # Vanishing point from leg lines.
    def _homogeneous(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], 1.0])

    L_left = np.cross(_homogeneous(bl), _homogeneous(tl))
    L_right = np.cross(_homogeneous(br), _homogeneous(tr))
    V_h = np.cross(L_left, L_right)

    if abs(V_h[2]) < 1e-10:
        return quad

    V = V_h[:2] / V_h[2]

    # Sanity: vanishing point should be above the top corners.
    if V[1] > min(tl[1], tr[1]):
        return quad

    # Project each top corner onto its own vanishing ray.
    ray_left = V - bl
    ray_right = V - br

    ray_left_sq = float(np.dot(ray_left, ray_left))
    ray_right_sq = float(np.dot(ray_right, ray_right))

    if ray_left_sq < 1.0 or ray_right_sq < 1.0:
        return quad

    # Each corner gets its own λ — the closest point on its ray.
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


# ---------------------------------------------------------------------------
# Core: build lane trapezoid
# ---------------------------------------------------------------------------

def build_lane_trapezoid(
    lane_mask: np.ndarray,
    *,
    frame_bgr: np.ndarray | None = None,
    pins_boxes: List[np.ndarray] | None = None,
    reject_counts: Optional[Dict[str, int]] = None,
) -> TrapezoidCandidate | None:
    """
    Extract a clean 4-point trapezoid from a single lane mask.

    Strategy: clean mask -> convex hull -> approxPolyDP to 4 pts -> fallback to Hough.
    If pins_boxes provided, use the nearest pins detection to anchor the top edge
    (the far/narrow end of the lane).
    """
    _ = frame_bgr
    def reject(reason: str) -> None:
        if reject_counts is not None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    cleaned = clean_mask((lane_mask > 0).astype(np.uint8))
    cleaned = keep_significant_components(cleaned)

    quad = _fit_quad_from_dense_mask(
        cleaned,
        reject_counts=reject_counts,
    )
    if quad is None:
        contour_mask = keep_largest_component(cleaned)
        contour = largest_contour(contour_mask)
        if contour is None or cv2.contourArea(contour) < 500:
            reject("contour_missing_or_small")
            return None

        quad = approx_to_quad(contour)
        if quad is None:
            quad = hough_quad_fallback(contour_mask, contour)

    if quad is None:
        reject("fallback_quad_failed")
        return None

    quad = order_quad_points(quad)

    width_top = abs(int(quad[1, 0]) - int(quad[0, 0]))
    width_bottom = abs(int(quad[2, 0]) - int(quad[3, 0]))
    if width_top < 4 or width_bottom < 8:
        reject("too_narrow")
        return None
    if width_top >= width_bottom * 0.98:
        reject("top_not_narrower")
        return None
    if not validate_vanishing_point(quad, cleaned.shape):
        reject("bad_vanishing_point")
        return None

    geometry_score, width_ratio, bottom_width, y_span_ratio = _geometry_scores(quad, cleaned.shape)
    support_score = _lane_band_support_score(cleaned, quad)
    if geometry_score < 0.16 or support_score < 0.12 or y_span_ratio < 0.045:
        reject("low_geometry_or_support")
        return None

    coverage, purity, score = evaluate_trapezoid(cleaned, quad)
    y_top = int(min(quad[0, 1], quad[1, 1]))
    y_bottom = int(max(quad[2, 1], quad[3, 1]))

    return TrapezoidCandidate(
        polygon=quad,
        coverage=coverage,
        purity=purity,
        score=score,
        y_top=y_top,
        y_bottom=y_bottom,
        geometry_score=geometry_score,
        support_score=support_score,
        width_ratio=width_ratio,
        bottom_width=bottom_width,
        mask_area=float(np.count_nonzero(cleaned)),
    )


# ---------------------------------------------------------------------------
# Detection extraction
# ---------------------------------------------------------------------------

def extract_frame_segmentation(result: Any) -> FrameSegmentation:
    """
    Convert one Ultralytics Results object to binary masks by class.
    """
    orig_h, orig_w = getattr(result, "orig_shape", (0, 0))
    if not orig_h or not orig_w:
        raise ValueError("Missing valid orig_shape on YOLO result")

    ball_masks: List[np.ndarray] = []
    lane_masks: List[np.ndarray] = []
    pins_masks: List[np.ndarray] = []

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or len(boxes) == 0:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

    cls_tensor = getattr(boxes, "cls", None)
    data_tensor = getattr(masks, "data", None)
    if cls_tensor is None or data_tensor is None:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

    mask_arr = data_tensor.detach().cpu().numpy()
    cls_ids = cls_tensor.detach().cpu().numpy().astype(int)

    if mask_arr.ndim != 3:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

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

    if len(lane_masks) > MAX_LANE_MASKS_PER_FRAME:
        lane_masks = sorted(
            lane_masks,
            key=lambda m: int(np.count_nonzero(m)),
            reverse=True,
        )[:MAX_LANE_MASKS_PER_FRAME]

    return FrameSegmentation(
        ball_masks=ball_masks,
        lane_masks=lane_masks,
        pins_masks=pins_masks,
        frame_shape=(int(orig_h), int(orig_w)),
    )


# ---------------------------------------------------------------------------
# API-aligned postprocessing and smoothing
# ---------------------------------------------------------------------------

_X_MARGIN = 0.15
_STEP_DROP_RATIO = 0.25
_MIN_HISTORY = 3
_FIT_WINDOW = 5
_CURVE_FIT_WINDOW = 14
_MAX_DEPARTURE_DX_M = 0.28
_SMOOTH_SIGMA_M = 0.04
_CONTACT_Y_TOL_M = 0.08
_END_Y_TOL_M = 0.35


def _linear_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    coeffs = np.polyfit(frames, values, 1)
    return float(coeffs[0])


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """
    Pool adjacent violators algorithm for a non-decreasing 1D fit.
    This keeps the reported lane-depth trajectory physically forward-moving
    without needing sklearn.
    """
    if values.size <= 1:
        return values.astype(np.float64, copy=True)

    block_starts: List[int] = []
    block_ends: List[int] = []
    block_weights: List[float] = []
    block_means: List[float] = []

    for idx, value in enumerate(values.astype(np.float64)):
        block_starts.append(idx)
        block_ends.append(idx + 1)
        block_weights.append(1.0)
        block_means.append(float(value))

        while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
            w_prev = block_weights[-2]
            w_last = block_weights[-1]
            merged_weight = w_prev + w_last
            merged_mean = (
                block_means[-2] * w_prev + block_means[-1] * w_last
            ) / merged_weight

            block_ends[-2] = block_ends[-1]
            block_weights[-2] = merged_weight
            block_means[-2] = merged_mean

            block_starts.pop()
            block_ends.pop()
            block_weights.pop()
            block_means.pop()

    out = np.empty(values.shape, dtype=np.float64)
    for start, end, mean in zip(block_starts, block_ends, block_means):
        out[start:end] = mean
    return out


def _spread_flat_forward_segments(values: np.ndarray) -> np.ndarray:
    """
    Isotonic regression can turn a short backward jitter into a flat plateau.
    A bowling ball should keep moving, so bounded flat runs are replaced with
    a straight forward interpolation between surrounding depths.
    """
    if values.size <= 2:
        return values.astype(np.float64, copy=True)

    out = values.astype(np.float64, copy=True)
    n = int(out.size)
    idx = 0
    while idx < n:
        end = idx + 1
        while end < n and abs(float(out[end] - out[idx])) < 1e-9:
            end += 1

        run_len = end - idx
        if run_len > 1 and idx > 0 and end < n:
            prev_y = float(out[idx - 1])
            next_y = float(out[end])
            if next_y > prev_y:
                out[idx:end] = np.linspace(prev_y, next_y, run_len + 2)[1:-1]

        idx = end

    return np.maximum.accumulate(out)


def _anchor_endpoint_values(
    smooth: np.ndarray,
    raw: np.ndarray,
    *,
    anchor_len: int = 5,
) -> np.ndarray:
    if smooth.size <= 2 or raw.size != smooth.size:
        return smooth.astype(np.float64, copy=True)

    out = smooth.astype(np.float64, copy=True)
    n = min(anchor_len, out.size)
    front_weights = np.linspace(1.0, 0.0, n)
    back_weights = np.linspace(0.0, 1.0, n)
    out[:n] = front_weights * raw[:n] + (1.0 - front_weights) * out[:n]
    out[-n:] = back_weights * raw[-n:] + (1.0 - back_weights) * out[-n:]
    return out


def trim_raw_detections(
    positions: List[BallPos],
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    return diagnose_trim_raw_detections(
        positions,
        lane_width_m=lane_width_m,
        lane_length_m=lane_length_m,
    ).kept_positions


def _position_in_lane_bounds(
    pos: BallPos,
    lane_width_m: float,
    lane_length_m: float,
) -> bool:
    x_lo = -_X_MARGIN
    x_hi = lane_width_m + _X_MARGIN
    return x_lo <= pos.x_m <= x_hi and -0.20 <= pos.y_m <= lane_length_m + 0.40


def _best_ball_motion_interval(
    sorted_pos: List[BallPos],
    *,
    lane_width_m: float,
    lane_length_m: float,
) -> tuple[int, int]:
    """
    Select the best contiguous, physically plausible ball-motion interval.
    This intentionally allows gaps from missed detections, but it does not let
    early pre-contact artifacts or late jumps define the smoothed trajectory.
    """
    if not sorted_pos:
        return 0, 0
    if len(sorted_pos) == 1:
        return 0, 1

    segments: List[tuple[int, int]] = []
    start = 0
    previous = sorted_pos[0]

    for idx in range(1, len(sorted_pos)):
        curr = sorted_pos[idx]
        frame_gap = max(curr.frame_index - previous.frame_index, 1)
        dy = curr.y_m - previous.y_m
        dx = curr.x_m - previous.x_m

        invalid = not _position_in_lane_bounds(curr, lane_width_m, lane_length_m)
        invalid = invalid or curr.y_m > lane_length_m + _END_Y_TOL_M
        too_large_gap = frame_gap > 36
        backwards = dy < -0.35
        implausible_jump = abs(dx) > 0.42 or dy / frame_gap > 0.75

        if invalid or too_large_gap or backwards or implausible_jump:
            if idx - start >= 1:
                segments.append((start, idx))
            start = idx

        previous = curr

    if len(sorted_pos) - start >= 1:
        segments.append((start, len(sorted_pos)))

    if not segments:
        return 0, len(sorted_pos)

    def _segment_key(seg: tuple[int, int]) -> tuple[float, float, float]:
        a, b = seg
        vals = sorted_pos[a:b]
        y_span = max(p.y_m for p in vals) - min(p.y_m for p in vals)
        in_bounds = sum(
            1 for p in vals if _position_in_lane_bounds(p, lane_width_m, lane_length_m)
        )
        frame_span = vals[-1].frame_index - vals[0].frame_index
        return (float(in_bounds), float(y_span), float(frame_span))

    best = max(segments, key=_segment_key)
    a, b = best

    # Start at lane contact. A small negative y is allowed for homography noise,
    # but clearly pre-lane points should not seed smoothing or kinematics.
    while b - a >= 2 and sorted_pos[a].y_m < -_CONTACT_Y_TOL_M:
        a += 1

    # Trim leading points that do not participate in the dominant forward motion.
    while b - a >= 4:
        head = sorted_pos[a : min(a + 4, b)]
        dy_head = head[-1].y_m - head[0].y_m
        if dy_head >= -0.05 and _position_in_lane_bounds(sorted_pos[a], lane_width_m, lane_length_m):
            break
        a += 1

    # Trim trailing points after the ball leaves the mapped lane or motion reverses.
    while b - a >= 4:
        tail_prev = sorted_pos[b - 2]
        tail = sorted_pos[b - 1]
        if (
            _position_in_lane_bounds(tail, lane_width_m, lane_length_m)
            and tail.y_m <= lane_length_m + _END_Y_TOL_M
            and tail.y_m - tail_prev.y_m >= -0.20
        ):
            break
        b -= 1

    return a, b


def diagnose_trim_raw_detections(
    positions: List[BallPos],
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> TrimDiagnostics:
    if len(positions) < 2:
        return TrimDiagnostics(
            kept_positions=list(positions),
            cut_reason=None,
            cut_frame_index=None,
            last_kept_frame_index=positions[-1].frame_index if positions else None,
            current_dy=None,
            median_dy=None,
            cut_x_m=None,
            cut_y_m=None,
        )

    sorted_pos = sorted(positions, key=lambda p: p.frame_index)

    start_idx, end_idx = _best_ball_motion_interval(
        sorted_pos,
        lane_width_m=lane_width_m,
        lane_length_m=lane_length_m,
    )
    kept = list(sorted_pos[start_idx:end_idx])
    cut_reason: Optional[str] = None
    cut_frame_index: Optional[int] = None
    current_dy: Optional[float] = None
    median_dy: Optional[float] = None
    cut_x_m: Optional[float] = None
    cut_y_m: Optional[float] = None

    if start_idx > 0:
        cut_reason = "pre_contact_or_artifact_interval"
        cut_frame_index = kept[0].frame_index if kept else sorted_pos[start_idx].frame_index
        cut_x_m = kept[0].x_m if kept else sorted_pos[start_idx].x_m
        cut_y_m = kept[0].y_m if kept else sorted_pos[start_idx].y_m
    elif end_idx < len(sorted_pos):
        cut_reason = "post_track_artifact_interval"
        cut = sorted_pos[end_idx]
        cut_frame_index = cut.frame_index
        cut_x_m = cut.x_m
        cut_y_m = cut.y_m

    if len(kept) >= 2:
        y_steps = np.diff(np.asarray([p.y_m for p in kept], dtype=np.float64))
        current_dy = float(y_steps[-1])
        if y_steps.size > 1:
            median_dy = float(np.median(y_steps))

    if len(kept) < len(sorted_pos):
        logger.info(
            "trim_raw_detections: %d→%d detections, kept frames %s-%s",
            len(sorted_pos),
            len(kept),
            kept[0].frame_index if kept else None,
            kept[-1].frame_index if kept else None,
        )

    return TrimDiagnostics(
        kept_positions=kept,
        cut_reason=cut_reason,
        cut_frame_index=cut_frame_index,
        last_kept_frame_index=kept[-1].frame_index if kept else None,
        current_dy=current_dy,
        median_dy=median_dy,
        cut_x_m=cut_x_m,
        cut_y_m=cut_y_m,
    )


def interpolate_ball_positions(
    positions: List[BallPos],
    fps: float,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    if len(positions) < 2:
        return list(positions)

    sorted_pos = sorted(positions, key=lambda p: p.frame_index)

    seen: set[int] = set()
    unique: List[BallPos] = []
    for p in sorted_pos:
        if p.frame_index not in seen:
            seen.add(p.frame_index)
            unique.append(p)
    sorted_pos = unique

    if len(sorted_pos) < 2:
        return list(sorted_pos)

    frames = np.array([p.frame_index for p in sorted_pos], dtype=np.float64)
    xs = np.array([p.x_m for p in sorted_pos], dtype=np.float64)
    ys = np.array([p.y_m for p in sorted_pos], dtype=np.float64)

    first_frame = int(sorted_pos[0].frame_index)
    last_frame = int(sorted_pos[-1].frame_index)
    all_frames = np.arange(first_frame, last_frame + 1, dtype=np.float64)

    n = len(sorted_pos)
    if n >= 4:
        s = n * (_SMOOTH_SIGMA_M ** 2)
        spline_x = UnivariateSpline(frames, xs, k=3, s=s)
        spline_y = UnivariateSpline(frames, ys, k=3, s=s)
        smooth_x = spline_x(all_frames)
        smooth_y = spline_y(all_frames)
        raw_x = np.interp(all_frames, frames, xs)
        raw_y = np.interp(all_frames, frames, ys)
        smooth_x = _anchor_endpoint_values(np.asarray(smooth_x, dtype=np.float64), raw_x)
        smooth_y = _anchor_endpoint_values(np.asarray(smooth_y, dtype=np.float64), raw_y)
    else:
        smooth_x = np.interp(all_frames, frames, xs)
        smooth_y = np.interp(all_frames, frames, ys)

    smooth_y = _isotonic_non_decreasing(np.asarray(smooth_y, dtype=np.float64))
    smooth_y = _spread_flat_forward_segments(smooth_y)
    smooth_y = np.clip(smooth_y, 0.0, lane_length_m)

    safe_fps = max(fps, 1e-6)
    dense: List[BallPos] = []
    for i, frame_value in enumerate(all_frames):
        fi = int(frame_value)
        dense.append(
            BallPos(
                frame_index=fi,
                timestamp_s=float(fi / safe_fps),
                x_m=float(smooth_x[i]),
                y_m=float(smooth_y[i]),
            )
        )

    logger.info(
        "Interpolation: %d sparse → %d dense smoothed positions (frames %d–%d)",
        len(sorted_pos),
        len(dense),
        first_frame,
        last_frame,
    )
    return dense


def append_departure_point(
    positions: List[BallPos],
    fps: float,
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    if len(positions) < 2:
        return list(positions)

    result = list(positions)
    tail = result[-min(_CURVE_FIT_WINDOW, len(result)):]
    frames_t = np.array([p.frame_index for p in tail], dtype=np.float64)
    xs_t = np.array([p.x_m for p in tail], dtype=np.float64)
    ys_t = np.array([p.y_m for p in tail], dtype=np.float64)

    fit_n = min(_FIT_WINDOW, len(frames_t))
    vy = _linear_slope(frames_t[-fit_n:], ys_t[-fit_n:])

    if abs(vy) < 1e-4:
        return result

    last = result[-1]
    x_lo = -_X_MARGIN
    x_hi = lane_width_m + _X_MARGIN

    target_y = lane_length_m
    if last.y_m >= lane_length_m - 0.02:
        return result

    target_x = _predict_curve_x_at_y(
        ys_t,
        xs_t,
        target_y=target_y,
        last_x=last.x_m,
    )
    target_x = float(np.clip(target_x, last.x_m - _MAX_DEPARTURE_DX_M, last.x_m + _MAX_DEPARTURE_DX_M))
    target_x = float(np.clip(target_x, x_lo, x_hi))

    frames_per_m = 1.0 / max(vy, 1e-4)
    df = int(np.clip(round((target_y - last.y_m) * frames_per_m), 1, 60))
    dep_frame = last.frame_index + df
    safe_fps = max(fps, 1e-6)
    result.append(
        BallPos(
            frame_index=dep_frame,
            timestamp_s=float(dep_frame / safe_fps),
            x_m=target_x,
            y_m=float(target_y),
        )
    )
    logger.info(
        "Departure point: frame %d (x=%.3f y=%.3f, curve_fit=True)",
        dep_frame,
        result[-1].x_m,
        result[-1].y_m,
    )

    return result


def _predict_curve_x_at_y(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    target_y: float,
    last_x: float,
) -> float:
    if len(ys) < 2:
        return float(last_x)

    order = np.argsort(ys)
    y_sorted = ys[order].astype(np.float64)
    x_sorted = xs[order].astype(np.float64)

    unique_y: List[float] = []
    unique_x: List[float] = []
    for y in np.unique(y_sorted):
        vals = x_sorted[np.abs(y_sorted - y) < 1e-9]
        unique_y.append(float(y))
        unique_x.append(float(np.median(vals)))

    y = np.asarray(unique_y, dtype=np.float64)
    x = np.asarray(unique_x, dtype=np.float64)
    if y.size < 2:
        return float(last_x)

    y_span = float(y[-1] - y[0])
    if y_span < 0.35:
        return float(last_x)

    linear = np.polyfit(y, x, 1)
    x_linear = float(np.polyval(linear, target_y))
    if y.size < 5 or y_span < 1.0:
        return x_linear

    quad = np.polyfit(y, x, 2)
    x_quad = float(np.polyval(quad, target_y))
    curvature = abs(float(quad[0]))
    if not np.isfinite(x_quad) or curvature > 0.08:
        return x_linear

    return float(0.65 * x_quad + 0.35 * x_linear)


def describe_processing_stages(
    raw_positions: List[BallPos],
    trim_diag: TrimDiagnostics,
    smooth_positions: List[BallPos],
    final_positions: List[BallPos],
) -> List[str]:
    lines: List[str] = []

    if trim_diag.cut_reason is None:
        lines.append("trim: no cut triggered")
    else:
        lines.append(
            "trim: "
            f"reason={trim_diag.cut_reason} "
            f"cut_frame={trim_diag.cut_frame_index} "
            f"last_kept={trim_diag.last_kept_frame_index} "
            f"dy={trim_diag.current_dy if trim_diag.current_dy is not None else float('nan'):.3f} "
            f"median_dy={trim_diag.median_dy if trim_diag.median_dy is not None else float('nan'):.3f} "
            f"cut_x={trim_diag.cut_x_m if trim_diag.cut_x_m is not None else float('nan'):.3f} "
            f"cut_y={trim_diag.cut_y_m if trim_diag.cut_y_m is not None else float('nan'):.3f}"
        )

    dense_added = len(smooth_positions) - len(trim_diag.kept_positions)
    if dense_added > 0:
        lines.append(
            f"interp: added {dense_added} frame(s) between "
            f"{smooth_positions[0].frame_index} and {smooth_positions[-1].frame_index}"
        )
    elif smooth_positions:
        lines.append(
            f"interp: smoothing only, no densification "
            f"(frames {smooth_positions[0].frame_index}-{smooth_positions[-1].frame_index})"
        )
    else:
        lines.append("interp: no smoothed positions")

    extra_added = len(final_positions) - len(smooth_positions)
    if extra_added > 0:
        last_smooth = smooth_positions[-1]
        last_final = final_positions[-1]
        lines.append(
            "extrap: "
            f"appended departure point at frame {last_final.frame_index} "
            f"from last smooth frame {last_smooth.frame_index} "
            f"(x={last_final.x_m:.3f}, y={last_final.y_m:.3f})"
        )
    else:
        lines.append("extrap: no appended departure point")

    if raw_positions:
        sorted_raw = sorted(raw_positions, key=lambda p: p.frame_index)
        xs = np.asarray([p.x_m for p in sorted_raw], dtype=np.float64)
        ys = np.asarray([p.y_m for p in sorted_raw], dtype=np.float64)
        frames = np.asarray([p.frame_index for p in sorted_raw], dtype=np.int32)
        in_bounds = sum(
            1 for p in sorted_raw
            if _position_in_lane_bounds(p, LANE_WIDTH_M, LANE_LENGTH_M)
        )
        largest_gap = int(np.max(np.diff(frames))) if frames.size > 1 else 0
        lines.append(
            f"raw span: frames {sorted_raw[0].frame_index}-{sorted_raw[-1].frame_index} "
            f"largest_gap={largest_gap} in_bounds={in_bounds}/{len(sorted_raw)} "
            f"x_range={float(np.min(xs)):.3f}..{float(np.max(xs)):.3f} "
            f"y_range={float(np.min(ys)):.3f}..{float(np.max(ys)):.3f}"
        )
        head = ", ".join(
            f"{p.frame_index}:{p.x_m:.2f},{p.y_m:.2f}" for p in sorted_raw[:4]
        )
        tail = ", ".join(
            f"{p.frame_index}:{p.x_m:.2f},{p.y_m:.2f}" for p in sorted_raw[-4:]
        )
        lines.append(f"raw head: {head}")
        if tail != head:
            lines.append(f"raw tail: {tail}")

    return lines


def compute_kinematics_per_quarter(
    ball_positions: List[BallPos],
    lane_length_m: float = LANE_LENGTH_M,
) -> Kinematics:
    if len(ball_positions) < 2:
        return Kinematics(
            quarters=[
                QuarterKinematics(
                    quarter=i + 1,
                    start_m=i * lane_length_m / 4.0,
                    end_m=(i + 1) * lane_length_m / 4.0,
                    mean_speed_mps=0.0,
                    mean_acceleration_mps2=0.0,
                    sample_count=0,
                )
                for i in range(4)
            ]
        )

    positions = sorted(ball_positions, key=lambda p: p.frame_index)
    sample_y: List[float] = []
    speeds: List[float] = []
    accels: List[float] = []

    last_speed = None
    for i in range(1, len(positions)):
        prev = positions[i - 1]
        curr = positions[i]
        dt = max(curr.timestamp_s - prev.timestamp_s, 1e-6)
        dx = curr.x_m - prev.x_m
        dy = curr.y_m - prev.y_m
        speed = float(np.hypot(dx, dy) / dt)
        accel = 0.0 if last_speed is None else float((speed - last_speed) / dt)
        last_speed = speed

        sample_y.append(float(np.clip(curr.y_m, 0.0, lane_length_m)))
        speeds.append(speed)
        accels.append(accel)

    quarters: List[QuarterKinematics] = []
    q_len = lane_length_m / 4.0
    for i in range(4):
        q_start = i * q_len
        q_end = (i + 1) * q_len
        idx = [j for j, y in enumerate(sample_y) if q_start <= y <= q_end]

        if idx:
            q_speeds = np.array([speeds[j] for j in idx], dtype=np.float64)
            q_accels = np.array([accels[j] for j in idx], dtype=np.float64)
            mean_speed = float(np.mean(q_speeds))
            mean_accel = float(np.mean(q_accels))
            count = len(idx)
        else:
            mean_speed = 0.0
            mean_accel = 0.0
            count = 0

        quarters.append(
            QuarterKinematics(
                quarter=i + 1,
                start_m=q_start,
                end_m=q_end,
                mean_speed_mps=mean_speed,
                mean_acceleration_mps2=mean_accel,
                sample_count=count,
            )
        )

    return Kinematics(quarters=quarters)


def run_lane_ball_postprocessing(
    segmentations_by_frame: Dict[int, FrameSegmentation],
    fps: float,
    start_frame: int,
    *,
    ball_start_frame: Optional[int] = None,
    frames_bgr: Optional[Sequence[np.ndarray]] = None,
    min_trapezoid_score: float = DEFAULT_MIN_TRAPEZOID_SCORE,
) -> PostprocessResult:
    if ball_start_frame is None:
        ball_start_frame = start_frame

    def _empty_result(scanned: int = 0, with_lane: int = 0, coverage: float = 0.0) -> PostprocessResult:
        return PostprocessResult(
            ball_positions=BallPosList(ball_positions=[]),
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
        logger.warning("No segmentations provided — returning empty results")
        return _empty_result()

    sorted_frames = sorted(k for k in segmentations_by_frame.keys() if k >= start_frame)
    if not sorted_frames:
        logger.warning("No frames >= start_frame=%d — returning empty results", start_frame)
        return _empty_result()

    smoother = TemporalSmoother()
    coverage_values: List[float] = []
    frame_candidate_count: Dict[int, int] = {}
    lane_observations: List[LaneGeometryObservation] = []
    first_frame_shape: tuple[int, int] | None = None
    lane_debug_counts: Dict[str, int] = {}
    lane_debug_examples: List[str] = []

    scanned = 0
    frames_with_lane = 0

    for frame_index in sorted_frames:
        scanned += 1
        frame_seg = segmentations_by_frame[frame_index]
        h, w = frame_seg.frame_shape
        if first_frame_shape is None:
            first_frame_shape = (h, w)
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

        lane_debug_counts["lane_masks_seen"] = (
            lane_debug_counts.get("lane_masks_seen", 0) + len(frame_seg.lane_masks)
        )
        for lane_idx, lane_mask in enumerate(frame_seg.lane_masks):
            trap = build_lane_trapezoid(
                lane_mask,
                frame_bgr=frame_bgr,
                pins_boxes=pins_boxes,
                reject_counts=lane_debug_counts,
            )
            if trap is None:
                continue
            if trap.score < min_trapezoid_score:
                lane_debug_counts["below_min_score_pre_nested"] = (
                    lane_debug_counts.get("below_min_score_pre_nested", 0) + 1
                )
                if len(lane_debug_examples) < 8:
                    lane_debug_examples.append(
                        f"f{frame_index} lane{lane_idx}: score={trap.score:.3f} "
                        f"geom={trap.geometry_score:.3f} support={trap.support_score:.3f} "
                        f"coverage={trap.coverage:.3f} purity={trap.purity:.3f} "
                        f"ratio={trap.width_ratio:.3f}"
                    )
                continue
            candidates.append(trap)
            lane_mask_areas.append(float(np.count_nonzero(lane_mask)))

        _penalize_nested_lane_subsets(candidates)
        before_nested_filter = len(candidates)
        candidates = [cand for cand in candidates if cand.score >= min_trapezoid_score]
        if len(candidates) < before_nested_filter:
            lane_debug_counts["below_min_score_after_nested"] = (
                lane_debug_counts.get("below_min_score_after_nested", 0)
                + (before_nested_filter - len(candidates))
            )
        lane_mask_areas = [
            float(cand.mask_area if cand.mask_area > 0 else _polygon_area_px(cand.polygon))
            for cand in candidates
        ]
        for trap in candidates:
            lane_observations.append(
                build_lane_geometry_observation(
                    frame_index,
                    trap,
                    pins_boxes,
                )
            )

        frame_candidate_count[frame_index] = len(candidates)
        if candidates:
            smoother.update(
                candidates,
                frame_idx=frame_index,
                ball_boxes=ball_boxes,
                lane_mask_areas=lane_mask_areas,
            )

    print("  lane candidate diagnostics:")
    for key in sorted(lane_debug_counts):
        print(f"    {key}={lane_debug_counts[key]}")
    if lane_debug_examples:
        print("  low-score candidate examples:")
        for line in lane_debug_examples:
            print(f"    {line}")

    if not smoother.tracks:
        logger.warning(
            "No active lane found from start_frame=%d — returning empty results",
            start_frame,
        )
        return _empty_result(
            scanned=scanned,
            with_lane=frames_with_lane,
            coverage=float(np.mean(coverage_values)) if coverage_values else 0.0,
        )

    logger.info("Lane tracks:\n%s", smoother.summary())

    dst = _lane_dst_corners_m()
    frames_with_ball_masks = sum(
        1
        for frame_index, frame_seg in segmentations_by_frame.items()
        if frame_index >= ball_start_frame and frame_seg.ball_masks
    )
    track_selections: List[LaneTrackSelection] = []

    for track in smoother.tracks:
        matched_observations = [
            obs
            for obs in lane_observations
            if _observation_matches_track(obs, track)
        ]

        src_corners = track.best_quad.astype(np.float32)
        if first_frame_shape is not None:
            aggregated_quad = _aggregate_lane_quad_from_observations(
                matched_observations,
                first_frame_shape,
            )
            if aggregated_quad is not None:
                src_corners = aggregated_quad.astype(np.float32)

        homography = cv2.getPerspectiveTransform(src_corners, dst)
        raw_ball_positions = _project_ball_positions_for_lane(
            segmentations_by_frame,
            fps,
            ball_start_frame,
            src_corners,
            homography,
        )
        track_selections.append(
            LaneTrackSelection(
                track=track,
                src_corners=src_corners,
                matched_observations=list(matched_observations),
                lane_quality=float(TemporalSmoother._lane_quality_key(track)[0]),
                ball_score=_ball_path_score(raw_ball_positions),
                ball_positions=raw_ball_positions,
            )
        )

    max_avg_area = max((_track_average_area(sel.track) for sel in track_selections), default=0.0)
    max_bottom_width = max((_quad_bottom_width_px(sel.src_corners) for sel in track_selections), default=0.0)
    selection_metrics = {
        id(sel): _lane_track_selection_metrics(
            sel,
            max_avg_area=max_avg_area,
            max_bottom_width=max_bottom_width,
        )
        for sel in track_selections
    }
    max_ball_score = max((sel.ball_score for sel in track_selections), default=0.0)
    strong_ball_score_threshold = 0.35
    if max_ball_score >= strong_ball_score_threshold:
        selected_track = max(
            track_selections,
            key=lambda sel: 0.35 * sel.lane_quality + 0.65 * sel.ball_score,
        )
        selection_sort_key = lambda item: 0.35 * item.lane_quality + 0.65 * item.ball_score
        selection_reason = "strong_ball_projection"
    else:
        selected_track = max(
            track_selections,
            key=lambda sel: (
                sel.lane_quality + 0.03 * min(sel.track.ball_votes, 12) / 12.0,
                _track_average_area(sel.track),
                float(sel.track.seen_count),
            ),
        )
        selection_sort_key = (
            lambda item: item.lane_quality
            + 0.03 * min(item.track.ball_votes, 12) / 12.0
        )
        selection_reason = "lane_quality_low_ball_projection"

    active_lane = selected_track.track
    matched_observations = selected_track.matched_observations
    src_corners = selected_track.src_corners
    homography = cv2.getPerspectiveTransform(src_corners, dst)

    print(f"  lane track selection ({selection_reason} max_ball_q={max_ball_score:.3f}):")
    for idx, sel in enumerate(
        sorted(
            track_selections,
            key=selection_sort_key,
            reverse=True,
        )[:8]
    ):
        sel_score, area_q, width_q, gated_ball_q = selection_metrics[id(sel)]
        print(
            f"    rank{idx}: lane_q={sel.lane_quality:.3f} "
            f"ball_q={sel.ball_score:.3f} gated_ball_q={gated_ball_q:.3f} "
            f"sel_q={sel_score:.3f} area_q={area_q:.3f} width_q={width_q:.3f} "
            f"top_cx={_quad_top_center_x(sel.src_corners):.0f}px "
            f"bot_l={sel.src_corners[3, 0]:.0f}px bot_r={sel.src_corners[2, 0]:.0f}px "
            f"bottom_w={_quad_bottom_width_px(sel.src_corners):.0f}px "
            f"avg_area={_track_average_area(sel.track):.0f}px "
            f"seen={sel.track.seen_count} votes={sel.track.ball_votes} "
            f"best_frame={sel.track.best_frame_idx}"
        )

    logger.info(
        "Active lane: best_frame=%d best_score=%.3f ball_votes=%d seen=%d lane_q=%.3f ball_q=%.3f",
        active_lane.best_frame_idx,
        active_lane.best_score,
        active_lane.ball_votes,
        active_lane.seen_count,
        selected_track.lane_quality,
        selected_track.ball_score,
    )
    if matched_observations:
        logger.info(
            "Aggregated lane geometry from %d matching observations",
            len(matched_observations),
        )

    selection = HomographySelection(
        frame_index=int(active_lane.best_frame_idx),
        homography=homography,
        src_corners=src_corners,
        dst_corners=dst,
        is_trapezoid=True,
        selected_lane_contours=int(max(len(matched_observations), frame_candidate_count.get(active_lane.best_frame_idx, 1))),
    )

    positions = [
        BallPos(
            frame_index=p.frame_index,
            timestamp_s=p.timestamp_s,
            x_m=float(np.clip(p.x_m, -0.5, LANE_WIDTH_M + 0.5)),
            y_m=float(np.clip(p.y_m, -1.0, LANE_LENGTH_M + 1.0)),
        )
        for p in selected_track.ball_positions
    ]
    frames_with_ball = len(positions)

    logger.info(
        "Ball projection: %d frames with ball masks, %d with valid contact, %d positions emitted",
        frames_with_ball_masks,
        frames_with_ball,
        len(positions),
    )

    det = float(np.linalg.det(selection.homography))
    cond = float(np.linalg.cond(selection.homography))

    health = PostprocessHealth(
        frames_scanned_for_h=int(scanned),
        frames_with_lane=int(frames_with_lane),
        frames_with_ball=int(frames_with_ball),
        lane_polygon_count_at_h=int(selection.selected_lane_contours),
        homography_determinant=det,
        homography_condition_number=cond,
        mean_lane_coverage_ratio=float(np.mean(coverage_values)) if coverage_values else 0.0,
    )

    return PostprocessResult(
        ball_positions=BallPosList(ball_positions=positions),
        homography_selection=selection,
        health=health,
    )


# ---------------------------------------------------------------------------
# Frame overlay
# ---------------------------------------------------------------------------

def draw_chosen_lane(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    best_frame_idx: int = 0,
    best_score: float = 0.0,
) -> None:
    """Draw the chosen active-lane trapezoid persistently on every frame."""
    color = (0, 255, 0)  # bright green — distinct from per-frame debug colors

    # Semi-transparent fill.
    fill = image.copy()
    cv2.fillPoly(fill, [quad.reshape((-1, 1, 2))], color)
    cv2.addWeighted(fill, 0.15, image, 0.85, 0, dst=image)

    cv2.polylines(image, [quad.reshape((-1, 1, 2))], True, color, 3, cv2.LINE_AA)

    corner_labels = ["TOP-L", "TOP-R", "BOT-R", "BOT-L"]
    for i in range(4):
        pt = tuple(quad[i].tolist())
        cv2.circle(image, pt, 7, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, 7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image, corner_labels[i],
            (pt[0] + 10, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )

    # Label at top.
    tl = tuple(quad[0].tolist())
    if best_score > 0.0:
        text = f"ACTIVE LANE (from f{best_frame_idx}, s={best_score:.2f})"
    else:
        text = f"ACTIVE LANE (from f{best_frame_idx})"
    cv2.putText(
        image, text,
        (int(tl[0]), max(20, int(tl[1]) - 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )


def draw_ball_contact(
    image: np.ndarray,
    contact_px: tuple[float, float],
    ball_pos: BallPos,
) -> None:
    """Draw the ball contact point and its metre-space coordinates."""
    cx, cy = int(contact_px[0]), int(contact_px[1])
    color = (0, 200, 255)  # orange-yellow

    cv2.circle(image, (cx, cy), 8, color, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 8, (0, 0, 0), 2, cv2.LINE_AA)

    label = f"({ball_pos.x_m:.2f}m, {ball_pos.y_m:.2f}m)"
    cv2.putText(
        image, label,
        (cx + 12, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
    )


def overlay_detections_on_frame(
    frame_bgr: np.ndarray,
    result: Any,
    class_names: Dict[int, str],
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay segmentation masks + boxes + labels (no trapezoid logic)."""
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
            output, label, (x1 + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0, dst=output)
    return output


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

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
    ball_start_frame: int,
    max_video_frames: int | None,
    max_video_dimension: int | None,
) -> None:
    _ = lane_class_id  # Kept for CLI compatibility; mirrored pipeline uses fixed API class ids.

    model = YOLO(str(model_path))

    names_raw = getattr(model, "names", None)
    class_names = normalize_model_names(names_raw)

    split_video = split_video_into_frames(
        str(video_path),
        max_frames=max_video_frames,
        max_dimension=max_video_dimension,
    )
    if not split_video.frames:
        raise RuntimeError("No frames found in video.")

    fps = split_video.fps if split_video.fps > 0 else 30.0
    frames = split_video.frames
    total = len(frames)
    print(
        "Video frames: "
        f"{total} @ {fps:.3f} fps, size={split_video.width}x{split_video.height}, "
        f"ball_start_frame={ball_start_frame}"
    )

    print("Pass 1: running inference and extracting API-aligned segmentations...")
    all_results: List[Any] = []
    segmentations_by_frame: Dict[int, FrameSegmentation] = {}

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_frames = frames[start:end]
        batch_bgr = [vf.image for vf in batch_frames]
        batch_rgb = [to_model_rgb(frame) for frame in batch_bgr]

        results = model.predict(
            source=batch_rgb,
            verbose=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
        )

        for fi, result in enumerate(results):
            frame_idx = start + fi
            all_results.append(result)
            segmentations_by_frame[frame_idx] = extract_frame_segmentation(result)

        print(f"  inference {start}..{end - 1} / {total - 1}")

    chosen_quad: np.ndarray | None = None
    chosen_best_frame = 0
    chosen_best_score = 0.0
    homography: np.ndarray | None = None
    ball_positions: List[BallPos] = []
    frame_ball_data: Dict[int, tuple[tuple[float, float], BallPos]] = {}

    if enable_guided_trapezoid:
        print("\nPostprocess: running mirrored LaneBalls pipeline...")
        post = run_lane_ball_postprocessing(
            segmentations_by_frame=segmentations_by_frame,
            fps=fps,
            start_frame=0,
            ball_start_frame=ball_start_frame,
            frames_bgr=[frame.image for frame in frames],
            min_trapezoid_score=min_trapezoid_score,
        )

        raw_positions = post.ball_positions.ball_positions
        trim_diag = diagnose_trim_raw_detections(raw_positions)
        clean_positions = trim_diag.kept_positions
        smooth_positions = interpolate_ball_positions(clean_positions, fps)
        final_positions = append_departure_point(smooth_positions, fps)
        kinematics = compute_kinematics_per_quarter(final_positions)

        if post.homography_selection.is_trapezoid:
            chosen_quad = post.homography_selection.src_corners.astype(np.int32)
            chosen_best_frame = int(post.homography_selection.frame_index)
            homography = post.homography_selection.homography

        ball_positions = final_positions

        if homography is not None:
            inv_homography = np.linalg.inv(homography)
            for bp in ball_positions:
                if bp.frame_index < 0 or bp.frame_index >= total:
                    continue
                try:
                    px, py = _project_point_homography((bp.x_m, bp.y_m), inv_homography)
                except RuntimeError:
                    continue
                px = float(np.clip(px, 0.0, max(split_video.width - 1, 0)))
                py = float(np.clip(py, 0.0, max(split_video.height - 1, 0)))
                frame_ball_data[bp.frame_index] = ((px, py), bp)

        print("  postprocess health:")
        print(f"    frames_scanned_for_h={post.health.frames_scanned_for_h}")
        print(f"    frames_with_lane={post.health.frames_with_lane}")
        print(f"    frames_with_ball={post.health.frames_with_ball}")
        print(f"    lane_polygon_count_at_h={post.health.lane_polygon_count_at_h}")
        print(f"    homography_determinant={post.health.homography_determinant:.6f}")
        print(f"    homography_condition_number={post.health.homography_condition_number:.1f}")
        print(f"    mean_lane_coverage_ratio={post.health.mean_lane_coverage_ratio:.4f}")
        if chosen_quad is not None:
            print("  homography src corners (image px):")
            quad_labels = ["TOP-L", "TOP-R", "BOT-R", "BOT-L"]
            for label, pt in zip(quad_labels, chosen_quad.tolist()):
                print(f"    {label}=({int(pt[0])}, {int(pt[1])})")
        print("  position counts:")
        print(f"    raw={len(raw_positions)} trimmed={len(clean_positions)} "
              f"smoothed={len(smooth_positions)} final={len(final_positions)}")
        print("  processing diagnostics:")
        for line in describe_processing_stages(
            raw_positions,
            trim_diag,
            smooth_positions,
            final_positions,
        ):
            print(f"    {line}")
        print("  kinematics:")
        for quarter in kinematics.quarters:
            print(
                f"    Q{quarter.quarter}: speed={quarter.mean_speed_mps:.3f} m/s  "
                f"accel={quarter.mean_acceleration_mps2:.3f} m/s^2  "
                f"samples={quarter.sample_count}"
            )
    else:
        print("\nPostprocess disabled: rendering detections only.")

    print(f"\nPass 2: rendering output video...")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (split_video.width, split_video.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    try:
        for fi in range(total):
            frame_bgr = frames[fi].image
            result = all_results[fi]

            output = overlay_detections_on_frame(
                frame_bgr, result, class_names, alpha=alpha,
            )

            if chosen_quad is not None:
                draw_chosen_lane(
                    output, chosen_quad,
                    best_frame_idx=chosen_best_frame,
                    best_score=chosen_best_score,
                )

            if fi in frame_ball_data:
                contact_px, bp = frame_ball_data[fi]
                draw_ball_contact(output, contact_px, bp)

            writer.write(output)

            if (fi + 1) % 100 == 0 or fi == total - 1:
                print(f"  rendered {fi + 1} / {total}")
    finally:
        writer.release()

    print(f"\nOverlay video saved to: {output_path}")

    # Print ball position summary.
    if ball_positions:
        print(f"\n--- Ball Positions ({len(ball_positions)} points) ---")
        print(f"{'frame':>6} {'time_s':>8} {'x_m':>8} {'y_m':>8} {'img_x':>8} {'img_y':>8}")
        for bp in ball_positions:
            contact_px = frame_ball_data.get(bp.frame_index, ((float("nan"), float("nan")), bp))[0]
            print(
                f"{bp.frame_index:6d} {bp.timestamp_s:8.3f} "
                f"{bp.x_m:8.3f} {bp.y_m:8.3f} "
                f"{contact_px[0]:8.1f} {contact_px[1]:8.1f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "--ball_start_frame",
        type=int,
        default=None,
        help=(
            "Ignore ball homography/contact processing before this frame "
            f"(default {DEFAULT_BALL_START_FRAME}, or {API_DEFAULT_BALL_START_FRAME} with --api_service_mode)"
        ),
    )
    parser.add_argument(
        "--api_service_mode",
        action="store_true",
        help=(
            "Mirror API video envelope: max 600 frames, max dimension 1024, "
            "and API default ball start frame when --ball_start_frame is omitted"
        ),
    )
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=None,
        help="Subsample/cap decoded frames like the API service (use 600 to mirror API)",
    )
    parser.add_argument(
        "--max_video_dimension",
        type=int,
        default=None,
        help="Resize longest decoded frame edge like the API service (use 1024 to mirror API)",
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
    ball_start_frame = (
        int(args.ball_start_frame)
        if args.ball_start_frame is not None
        else (API_DEFAULT_BALL_START_FRAME if args.api_service_mode else DEFAULT_BALL_START_FRAME)
    )
    max_video_frames = (
        int(args.max_video_frames)
        if args.max_video_frames is not None
        else (API_DEFAULT_MAX_VIDEO_FRAMES if args.api_service_mode else None)
    )
    max_video_dimension = (
        int(args.max_video_dimension)
        if args.max_video_dimension is not None
        else (API_DEFAULT_MAX_VIDEO_DIMENSION if args.api_service_mode else None)
    )

    if ball_start_frame < 0:
        raise ValueError("--ball_start_frame must be >= 0")
    if max_video_frames is not None and max_video_frames <= 0:
        raise ValueError("--max_video_frames must be > 0")
    if max_video_dimension is not None and max_video_dimension <= 0:
        raise ValueError("--max_video_dimension must be > 0")

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
        ball_start_frame=ball_start_frame,
        max_video_frames=max_video_frames,
        max_video_dimension=max_video_dimension,
    )


if __name__ == "__main__":
    main()
