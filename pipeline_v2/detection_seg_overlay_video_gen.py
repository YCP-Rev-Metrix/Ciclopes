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

# Lane physical dimensions (regulation bowling lane).
LANE_LENGTH_M = 18.288
LANE_WIDTH_M = 1.0541


@dataclass
class BallPos:
    frame_index: int
    timestamp_s: float
    x_m: float
    y_m: float


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


class TemporalSmoother:
    """
    [Improvement 1] Exponential moving average on per-lane corner positions.
    Matches lanes across frames by centroid-x proximity. Rejects outlier
    frames where corners jump too far, preserving the smoothed estimate.
    Tracks the single best-scoring frame per lane for homography selection.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        max_match_dist: float = 80.0,
        max_corner_jump: float = 60.0,
        stale_frames: int = 30,
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
        ball_boxes: List[np.ndarray] | None = None,
        lane_mask_areas: List[float] | None = None,
    ) -> List[TrapezoidCandidate]:
        used_tracks: set[int] = set()
        smoothed: List[TrapezoidCandidate] = []
        # Map candidate index → track index for ball association.
        cand_to_track: List[int] = []

        for ci, cand in enumerate(candidates):
            cx = float(np.mean(cand.polygon[:, 0]))

            # Match to closest existing track.
            best_ti: int | None = None
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

                if max_jump < self.max_corner_jump:
                    a = self.ema_alpha
                    blended = (
                        a * cand.polygon.astype(float)
                        + (1.0 - a) * track.quad.astype(float)
                    )
                    track.quad = blended.astype(np.int32)
                # else: outlier frame — keep previous smoothed quad

                track.centroid_x = float(np.mean(track.quad[:, 0]))
                track.last_seen_frame = frame_idx
                track.seen_count += 1
                if lane_mask_areas and ci < len(lane_mask_areas):
                    track.total_mask_area += lane_mask_areas[ci]
                if cand.score > track.best_score:
                    track.best_score = cand.score
                    track.best_quad = cand.polygon.copy()
                    track.best_frame_idx = frame_idx

                smoothed.append(
                    TrapezoidCandidate(
                        polygon=track.quad.copy(),
                        coverage=cand.coverage,
                        purity=cand.purity,
                        score=cand.score,
                        y_top=int(np.min(track.quad[:, 1])),
                        y_bottom=int(np.max(track.quad[:, 1])),
                    )
                )
                cand_to_track.append(best_ti)
            else:
                # New track.
                area = 0.0
                if lane_mask_areas and ci < len(lane_mask_areas):
                    area = lane_mask_areas[ci]
                new_idx = len(self.tracks)
                self.tracks.append(
                    TrackedLane(
                        quad=cand.polygon.copy(),
                        centroid_x=cx,
                        last_seen_frame=frame_idx,
                        best_score=cand.score,
                        best_quad=cand.polygon.copy(),
                        best_frame_idx=frame_idx,
                        ball_votes=0,
                        total_mask_area=area,
                        seen_count=1,
                    )
                )
                smoothed.append(cand)
                cand_to_track.append(new_idx)

        # Associate balls with lane tracks.
        if ball_boxes:
            for ball_box in ball_boxes:
                ball_cx = float(ball_box[0] + ball_box[2]) / 2.0
                ball_cy = float(ball_box[1] + ball_box[3]) / 2.0
                ball_pt = np.array([ball_cx, ball_cy], dtype=np.float32)

                for ci, cand in enumerate(candidates):
                    if ci < len(cand_to_track):
                        ti = cand_to_track[ci]
                        if ti < len(self.tracks):
                            dist = cv2.pointPolygonTest(
                                cand.polygon.reshape((-1, 1, 2)).astype(np.float32),
                                (ball_cx, ball_cy),
                                measureDist=False,
                            )
                            if dist >= 0:
                                self.tracks[ti].ball_votes += 1
                                break  # one ball → one lane

        # Expire stale tracks.
        self.tracks = [
            t for t in self.tracks if frame_idx - t.last_seen_frame < self.stale_frames
        ]
        return smoothed

    def select_active_lane(self) -> TrackedLane | None:
        """
        Pick the active (bowler's) lane. Priority:
        1. Lane with the most ball votes (ball was detected inside it).
        2. Fallback: lane with the largest average mask area (closest to camera).
        """
        if not self.tracks:
            return None

        # Any tracks with ball votes?
        tracks_with_balls = [t for t in self.tracks if t.ball_votes > 0]
        if tracks_with_balls:
            return max(tracks_with_balls, key=lambda t: t.ball_votes)

        # Fallback: largest average mask area = closest/most prominent lane.
        return max(
            self.tracks,
            key=lambda t: t.total_mask_area / max(t.seen_count, 1),
        )

    def summary(self) -> str:
        parts: List[str] = []
        for i, t in enumerate(self.tracks):
            avg_area = t.total_mask_area / max(t.seen_count, 1)
            parts.append(
                f"  lane track {i}: best_score={t.best_score:.3f} "
                f"best_frame={t.best_frame_idx} "
                f"ball_votes={t.ball_votes} avg_area={avg_area:.0f} "
                f"seen={t.seen_count}"
            )
        return "\n".join(parts) if parts else "  (no tracked lanes)"


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

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

    # Corner points with labels.
    corner_labels = ["TL", "TR", "BR", "BL"]
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
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.
    Standard ordering for homography source points.
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
    [Improvement 6] Fallback: run Canny on the actual cleaned mask edges
    (not the synthetic hull drawing), then HoughLines to find the 4 dominant
    lines and intersect them for corners.
    """
    h, w = mask.shape

    # Canny on the real mask boundary — picks up the actual edge shape.
    edges = cv2.Canny(mask * 255, 50, 150)

    min_dim = min(h, w)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=max(30, min_dim // 8))
    if lines is None or len(lines) < 4:
        return None

    # De-duplicate by (rho, theta) proximity.
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

    def _line_intersect(
        r1: float, t1: float, r2: float, t2: float,
    ) -> tuple[float, float] | None:
        det = np.cos(t1) * np.sin(t2) - np.cos(t2) * np.sin(t1)
        if abs(det) < 1e-6:
            return None
        x = (r1 * np.sin(t2) - r2 * np.sin(t1)) / det
        y = (r2 * np.cos(t1) - r1 * np.cos(t2)) / det
        return (x, y)

    corners: List[tuple[float, float]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            pt = _line_intersect(unique[i][0], unique[i][1], unique[j][0], unique[j][1])
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                corners.append(pt)

    if len(corners) < 4:
        return None

    corners_np = np.array(corners, dtype=np.float32)
    M = cv2.moments(cv2.convexHull(contour))
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
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
    mask: np.ndarray,
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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

    # Geometric quality: reward when bottom is wider than top (perspective)
    # and sides are roughly symmetric.
    q = polygon.astype(np.float64)
    width_top = abs(q[1, 0] - q[0, 0])
    width_bot = abs(q[2, 0] - q[3, 0])
    center_top = (q[0, 0] + q[1, 0]) / 2.0
    center_bot = (q[2, 0] + q[3, 0]) / 2.0

    # Taper ratio: ideal ~0.3-0.6 for bowling lane perspective.
    taper = width_top / max(width_bot, 1.0)
    taper_score = 1.0 - min(abs(taper - 0.45) / 0.45, 1.0)

    # Symmetry: top and bottom centers should be close horizontally.
    center_shift = abs(center_top - center_bot) / max(width_bot, 1.0)
    symmetry_score = max(0.0, 1.0 - center_shift * 2.0)

    geometry = 0.5 * taper_score + 0.5 * symmetry_score

    overlap_weight = 1.0 - score_coverage_weight - score_geometry_weight
    score = (
        score_coverage_weight * coverage
        + overlap_weight * purity
        + score_geometry_weight * geometry
    )
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
    """
    Find the bottom contact point of a ball mask — the point where the ball
    touches the lane surface. Uses the bottom-most row of the mask and the
    mean x at that row.
    """
    ys, xs = np.where(ball_mask > 0)
    if xs.size == 0:
        return None

    y_contact = int(np.max(ys))
    xs_at_contact = xs[ys == y_contact]
    if xs_at_contact.size == 0:
        return None
    x_contact = float(np.mean(xs_at_contact))
    return x_contact, float(y_contact)


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
) -> TrapezoidCandidate | None:
    """
    Extract a clean 4-point trapezoid from a single lane mask.

    Pipeline:
      1. Morph cleanup (close+open)
      2. [Imp 5] Connected-component isolation → largest blob only
      3. Convex hull → adaptive approxPolyDP to 4 pts
      4. [Imp 6] Fallback: Canny on mask edges → Hough
      5. Order as TL, TR, BR, BL
      6. [Imp 3] Pins anchoring (both x and y)
      7. [Imp 2] Refine side edges from raw image
      8. [Imp 4] Vanishing-point validation
      9. Score with geometric priors
    """
    cleaned = clean_mask(lane_mask)

    # [Improvement 5] Isolate the single largest connected component.
    cleaned = keep_largest_component(cleaned)

    contour = largest_contour(cleaned)
    if contour is None or cv2.contourArea(contour) < 500:
        return None

    # Primary: adaptive approxPolyDP on convex hull.
    quad = approx_to_quad(contour)

    # [Improvement 6] Fallback: Canny on real mask edges → Hough.
    if quad is None:
        quad = hough_quad_fallback(cleaned, contour)

    if quad is None:
        return None

    quad = order_quad_points(quad)

    # [Improvement 3] Pins anchoring — use both y AND x from the pins box.
    if pins_boxes:
        pins_box = find_nearest_pins_box(lane_mask, pins_boxes)
        if pins_box is not None:
            pins_bot_y = int(pins_box[3])
            pins_cx = float(pins_box[0] + pins_box[2]) / 2.0
            pins_half_w = float(pins_box[2] - pins_box[0]) / 2.0
            current_top_y = int(min(quad[0, 1], quad[1, 1]))

            if abs(pins_bot_y - current_top_y) < lane_mask.shape[0] * 0.3:
                quad[0, 1] = pins_bot_y
                quad[1, 1] = pins_bot_y

                # Lane is ~1.7x wider than pin deck at the pin line.
                lane_half_at_pins = pins_half_w * 1.7
                proposed_tl_x = pins_cx - lane_half_at_pins
                proposed_tr_x = pins_cx + lane_half_at_pins

                # Only apply if the adjustment is reasonable.
                if abs(proposed_tl_x - quad[0, 0]) < 50:
                    quad[0, 0] = int(proposed_tl_x)
                if abs(proposed_tr_x - quad[1, 0]) < 50:
                    quad[1, 0] = int(proposed_tr_x)

    # [Improvement 2] Refine side edges using gutter edges in the raw image.
    if frame_bgr is not None:
        quad = refine_sides_from_image(frame_bgr, quad, cleaned)

    # [Improvement 4] Reject quads that don't form a valid perspective trapezoid.
    if not validate_vanishing_point(quad, lane_mask.shape):
        return None

    # Basic size validation.
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


# ---------------------------------------------------------------------------
# Detection extraction
# ---------------------------------------------------------------------------

def extract_detections(
    result: Any,
    lane_class_id: int,
    pins_class_id: int,
    ball_class_id: int,
    frame_shape_hw: tuple[int, int],
) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Extract all lane instance masks, pins bounding boxes, ball bounding
    boxes, and ball masks from a result.
    Returns (lane_masks, pins_boxes_xyxy, ball_boxes_xyxy, ball_masks).
    """
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or getattr(masks, "data", None) is None:
        return [], [], [], []

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    mask_arr = masks.data.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)
    h, w = frame_shape_hw

    lane_masks: List[np.ndarray] = []
    pins_boxes: List[np.ndarray] = []
    ball_boxes: List[np.ndarray] = []
    ball_masks: List[np.ndarray] = []

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
        elif cid == ball_class_id:
            ball_boxes.append(xyxy[i])
            if i < mask_arr.shape[0]:
                bmask = (mask_arr[i] > 0.5).astype(np.uint8)
                if bmask.shape != (h, w):
                    bmask = cv2.resize(bmask, (w, h), interpolation=cv2.INTER_NEAREST)
                ball_masks.append(bmask)

    return lane_masks, pins_boxes, ball_boxes, ball_masks


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

    corner_labels = ["TL", "TR", "BR", "BL"]
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
    text = f"ACTIVE LANE (from f{best_frame_idx}, s={best_score:.2f})"
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
) -> None:
    model = YOLO(str(model_path))

    names_raw = getattr(model, "names", None) or {0: "ball", 1: "lane", 2: "pins"}
    class_names = {int(k): str(v) for k, v in names_raw.items()}

    split_video = split_video_into_frames(str(video_path))
    if not split_video.frames:
        raise RuntimeError("No frames found in video.")

    fps = split_video.fps if split_video.fps > 0 else 30.0
    frames = split_video.frames
    total = len(frames)

    # ---------------------------------------------------------------
    # Pass 1: inference + trapezoid tracking + ball-lane association.
    # Store results and per-frame ball masks for pass 2.
    # ---------------------------------------------------------------
    print("Pass 1: running inference and tracking lanes...")
    smoother = TemporalSmoother() if enable_guided_trapezoid else None
    ball_class_id = 0
    pins_class_id = 2
    all_results: List[Any] = []
    all_ball_masks: Dict[int, List[np.ndarray]] = {}

    h, w = split_video.height, split_video.width

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_frames = frames[start:end]

        batch_bgr = [vf.image for vf in batch_frames]
        batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_bgr]

        results = model.predict(
            source=batch_rgb,
            verbose=False,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            retina_masks=True,
        )

        for fi, (frame_bgr, result) in enumerate(zip(batch_bgr, results)):
            frame_idx = start + fi
            all_results.append(result)

            if smoother is not None and enable_guided_trapezoid:
                lane_masks, pins_boxes, ball_boxes, ball_masks = extract_detections(
                    result,
                    lane_class_id=lane_class_id,
                    pins_class_id=pins_class_id,
                    ball_class_id=ball_class_id,
                    frame_shape_hw=(h, w),
                )

                # Store ball masks for pass 2 ball-position projection.
                if ball_masks:
                    all_ball_masks[frame_idx] = ball_masks

                candidates: List[TrapezoidCandidate] = []
                mask_areas: List[float] = []
                for lmask in lane_masks:
                    trap = build_lane_trapezoid(
                        lmask, frame_bgr=frame_bgr, pins_boxes=pins_boxes,
                    )
                    if trap is not None and trap.score >= min_trapezoid_score:
                        candidates.append(trap)
                        mask_areas.append(float(np.count_nonzero(lmask)))

                if candidates:
                    smoother.update(
                        candidates, frame_idx,
                        ball_boxes=ball_boxes,
                        lane_mask_areas=mask_areas,
                    )

        print(f"  inference {start}..{end - 1} / {total - 1}")

    # ---------------------------------------------------------------
    # Select active lane, compute homography, project ball positions.
    # ---------------------------------------------------------------
    chosen_quad: np.ndarray | None = None
    chosen_best_frame = 0
    chosen_best_score = 0.0
    homography: np.ndarray | None = None
    ball_positions: List[BallPos] = []
    # Map frame_index → (contact_px, BallPos) for overlay drawing.
    frame_ball_data: Dict[int, tuple[tuple[float, float], BallPos]] = {}

    if smoother is not None:
        active = smoother.select_active_lane()
        if active is not None:
            chosen_best_frame = active.best_frame_idx
            chosen_best_score = active.best_score

            # Correct far-end top corners using vanishing-point geometry.
            src_corners = _correct_trapezoid_top_corners(active.best_quad.astype(np.float32))
            chosen_quad = src_corners.astype(np.int32)

            # Compute homography: pixel quad → real-world lane metres.
            dst_corners = _lane_dst_corners_m()
            homography = cv2.getPerspectiveTransform(
                src_corners.astype(np.float32), dst_corners,
            )

            det = float(np.linalg.det(homography))
            cond = float(np.linalg.cond(homography))

            print(f"\nActive lane selected:")
            print(f"  ball_votes={active.ball_votes}  "
                  f"avg_area={active.total_mask_area / max(active.seen_count, 1):.0f}  "
                  f"best_score={active.best_score:.3f}  "
                  f"best_frame={active.best_frame_idx}")
            print(f"  homography det={det:.6f}  cond={cond:.1f}")

            # Project ball contact points through homography.
            for frame_idx in sorted(all_ball_masks.keys()):
                contact = _choose_ball_contact_for_lane(
                    all_ball_masks[frame_idx],
                    src_corners,
                )
                if contact is None:
                    continue

                try:
                    x_m, y_m = _project_point_homography(contact, homography)
                except RuntimeError:
                    continue

                x_m = float(np.clip(x_m, -0.5, LANE_WIDTH_M + 0.5))
                y_m = float(np.clip(y_m, -1.0, LANE_LENGTH_M + 1.0))

                bp = BallPos(
                    frame_index=frame_idx,
                    timestamp_s=float(frame_idx / max(fps, 1e-6)),
                    x_m=x_m,
                    y_m=y_m,
                )
                ball_positions.append(bp)
                frame_ball_data[frame_idx] = (contact, bp)

            print(f"  ball positions projected: {len(ball_positions)} / "
                  f"{len(all_ball_masks)} frames with ball masks")

        print(f"\nAll lane tracks:\n{smoother.summary()}")

    # ---------------------------------------------------------------
    # Pass 2: render overlay video with chosen lane + ball positions.
    # ---------------------------------------------------------------
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
        print(f"{'frame':>6} {'time_s':>8} {'x_m':>8} {'y_m':>8}")
        for bp in ball_positions:
            print(f"{bp.frame_index:6d} {bp.timestamp_s:8.3f} {bp.x_m:8.3f} {bp.y_m:8.3f}")


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
