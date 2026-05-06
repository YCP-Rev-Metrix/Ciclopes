from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from core.LaneBalls.Preprocessing import FrameSegmentation
from core.LaneBalls.models import BallPos, BallPosList

logger = logging.getLogger("ciclopes.lane_ball_postprocessing")

LANE_LENGTH_M = 18.288
LANE_WIDTH_M = 1.0541
DEFAULT_MIN_TRAPEZOID_SCORE = 0.20


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
    pins_box: Optional[np.ndarray]
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

    def select_active_lane(self) -> Optional[TrackedLane]:
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


def keep_significant_components(
    mask: np.ndarray,
    *,
    min_area_px: int = 300,
    min_largest_ratio: float = 0.03,
) -> np.ndarray:
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


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def approx_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri < 20:
        return None

    lo, hi = 0.005, 0.15
    best_approx: Optional[np.ndarray] = None
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
                tri = np.array(
                    [
                        pts[(i - 1) % len(pts)],
                        pts[i],
                        pts[(i + 1) % len(pts)],
                    ]
                )
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


def hough_quad_fallback(mask: np.ndarray, contour: np.ndarray) -> Optional[np.ndarray]:
    hull = cv2.convexHull(contour)
    h, w = mask.shape
    edge_img = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(edge_img, [hull], 0, 255, 2)

    min_dim = min(h, w)
    lines = cv2.HoughLines(
        edge_img,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, min_dim // 8),
    )
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

    def _line_intersect(r1: float, t1: float, r2: float, t2: float) -> Optional[tuple[float, float]]:
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
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    refined = quad.copy().astype(np.float32)
    sides = [(0, 3), (1, 2)]

    for top_idx, bot_idx in sides:
        pt_top = refined[top_idx]
        pt_bot = refined[bot_idx]

        dx = pt_bot[0] - pt_top[0]
        dy = pt_bot[1] - pt_top[1]
        length = np.sqrt(dx**2 + dy**2)
        if length < 20:
            continue

        nx, ny = -dy / length, dx / length
        strip_width = 25.0

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
            continue

        lx1, ly1, lx2, ly2 = best_line
        if abs(ly2 - ly1) < 1:
            continue

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


def _robust_fit_x_of_y(points_xy: np.ndarray) -> Optional[tuple[float, float, float]]:
    if points_xy.shape[0] < 8:
        return None

    pts = points_xy.astype(np.float64)
    keep = np.ones(pts.shape[0], dtype=bool)
    coeff: Optional[np.ndarray] = None

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


def _mask_boundary_xs_near_y(
    mask: np.ndarray,
    y: float,
    *,
    half_window: int = 4,
) -> Optional[tuple[float, float]]:
    y_center = int(round(float(y)))
    y0 = max(0, y_center - half_window)
    y1 = min(mask.shape[0], y_center + half_window + 1)
    xs = np.where(mask[y0:y1, :] > 0)[1]
    if xs.size < 8:
        return None
    return float(np.percentile(xs, 2.0)), float(np.percentile(xs, 98.0))


def _scanline_boundary_samples(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        left_x = float(np.percentile(xs, 3.0))
        right_x = float(np.percentile(xs, 97.0))
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
    return (
        float(np.clip(geometry, 0.0, 1.0)),
        float(width_ratio),
        float(width_bottom),
        float(y_span_ratio),
    )


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
) -> Optional[np.ndarray]:
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
    top_y = float(np.percentile(all_y, 3.0))
    bottom_y = float(np.percentile(all_y, 97.0))

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


def find_nearest_pins_box(
    lane_mask: np.ndarray,
    pins_boxes: Sequence[np.ndarray],
) -> Optional[np.ndarray]:
    if not pins_boxes:
        return None

    ys, xs = np.where(lane_mask > 0)
    if len(xs) == 0:
        return None
    lane_cx = float(np.mean(xs))

    best_box: Optional[np.ndarray] = None
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
) -> Optional[np.ndarray]:
    if not pins_boxes:
        return None

    lane_cx = float(np.mean(lane_polygon[:, 0]))
    best_box: Optional[np.ndarray] = None
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
    pins_boxes: Sequence[np.ndarray],
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
) -> Optional[np.ndarray]:
    pins = [obs.pins_box for obs in observations if obs.pins_box is not None]
    if not pins:
        return None
    return np.median(np.asarray(pins, dtype=np.float32), axis=0).astype(np.int32)


def _aggregate_lane_quad_from_observations(
    observations: Sequence[LaneGeometryObservation],
    img_shape_hw: tuple[int, int],
) -> Optional[np.ndarray]:
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

    top_y = float(np.median(0.5 * (quads[:, 0, 1] + quads[:, 1, 1])))
    bottom_y = float(np.median(0.5 * (quads[:, 3, 1] + quads[:, 2, 1])))
    quad[0, 1] = top_y
    quad[1, 1] = top_y
    quad[2, 1] = bottom_y
    quad[3, 1] = bottom_y

    quad[:, 0] = np.clip(quad[:, 0], 0, img_shape_hw[1] - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, img_shape_hw[0] - 1)

    geometry_score, width_ratio, _, y_span_ratio = _geometry_scores(quad, img_shape_hw)
    if width_ratio >= 0.98 or geometry_score < 0.16 or y_span_ratio < 0.045:
        return None
    return quad.astype(np.int32)


def build_lane_trapezoid(
    lane_mask: np.ndarray,
    *,
    frame_bgr: Optional[np.ndarray] = None,
    pins_boxes: Optional[Sequence[np.ndarray]] = None,
    reject_counts: Optional[Dict[str, int]] = None,
) -> Optional[TrapezoidCandidate]:
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
        coverage=float(coverage),
        purity=float(purity),
        score=float(score),
        y_top=y_top,
        y_bottom=y_bottom,
        geometry_score=float(geometry_score),
        support_score=float(support_score),
        width_ratio=float(width_ratio),
        bottom_width=float(bottom_width),
        mask_area=float(np.count_nonzero(cleaned)),
    )


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


def _ball_contact_point_from_mask(ball_mask: np.ndarray) -> Optional[tuple[float, float]]:
    ys, xs = np.where(ball_mask > 0)
    if xs.size == 0:
        return None

    # Use the bottom of the mask for y and the horizontal center of the full
    # bounding box for x.  Using only xs at y == y_contact is fragile: the
    # bottom-most row may contain just one or two pixels at a corner of the
    # detection, shifting x away from the true bottom-middle.
    y_contact = float(np.max(ys))
    x_contact = float(np.min(xs) + np.max(xs)) / 2.0
    return x_contact, y_contact


def _choose_ball_contact_for_lane(
    ball_masks: Sequence[np.ndarray],
    lane_polygon: np.ndarray,
) -> Optional[tuple[float, float]]:
    if not ball_masks:
        return None

    lane_poly = lane_polygon.reshape((-1, 1, 2)).astype(np.float32)
    lane_center_x = float(np.mean(lane_polygon[:, 0]))

    best_point: Optional[tuple[float, float]] = None
    best_key: Optional[tuple[float, float, float, float]] = None

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


_X_MARGIN = 0.15
_CONTACT_Y_TOL_M = 0.08
_END_Y_TOL_M = 0.35


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
        return float(in_bounds), float(y_span), float(frame_span)

    best = max(segments, key=_segment_key)
    a, b = best

    while b - a >= 2 and sorted_pos[a].y_m < -_CONTACT_Y_TOL_M:
        a += 1

    while b - a >= 4:
        head = sorted_pos[a : min(a + 4, b)]
        dy_head = head[-1].y_m - head[0].y_m
        if dy_head >= -0.05 and _position_in_lane_bounds(sorted_pos[a], lane_width_m, lane_length_m):
            break
        a += 1

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

    # ── Step 1: vanishing point from leg lines ───────────────────────────
    def _homogeneous(p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], 1.0])

    L_left = np.cross(_homogeneous(bl), _homogeneous(tl))
    L_right = np.cross(_homogeneous(br), _homogeneous(tr))
    V_h = np.cross(L_left, L_right)

    if abs(V_h[2]) < 1e-10:
        logger.info("Trapezoid correction: legs parallel, skipping")
        return quad

    V = V_h[:2] / V_h[2]

    # Sanity: vanishing point should be above the top corners
    if V[1] > min(tl[1], tr[1]):
        logger.warning(
            "Trapezoid correction: vanishing point below top corners "
            "(V_y=%.1f, tl_y=%.1f), skipping", V[1], tl[1],
        )
        return quad

    # ── Step 2: project each top corner onto its own vanishing ray ───────
    ray_left = V - bl
    ray_right = V - br

    ray_left_sq = float(np.dot(ray_left, ray_left))
    ray_right_sq = float(np.dot(ray_right, ray_right))

    if ray_left_sq < 1.0 or ray_right_sq < 1.0:
        return quad

    # Each corner gets its own λ — the closest point on its ray
    lam_left = float(np.dot(tl - bl, ray_left) / ray_left_sq)
    lam_right = float(np.dot(tr - br, ray_right) / ray_right_sq)

    if lam_left <= 0.0 or lam_left >= 1.0 or lam_right <= 0.0 or lam_right >= 1.0:
        logger.warning(
            "Trapezoid correction: λ out of range (L=%.4f R=%.4f), skipping",
            lam_left, lam_right,
        )
        return quad

    tl_new = bl + lam_left * ray_left
    tr_new = br + lam_right * ray_right

    correction_tl = float(np.linalg.norm(tl_new - tl))
    correction_tr = float(np.linalg.norm(tr_new - tr))

    logger.info(
        "Trapezoid correction: λ_L=%.4f λ_R=%.4f, "
        "tl shifted %.1fpx, tr shifted %.1fpx",
        lam_left, lam_right, correction_tl, correction_tr,
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
    first_frame_shape: Optional[tuple[int, int]] = None
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

    logger.info(
        "Lane candidate diagnostics: %s",
        " ".join(f"{key}={lane_debug_counts[key]}" for key in sorted(lane_debug_counts)),
    )
    if lane_debug_examples:
        logger.info("Low-score candidate examples: %s", " | ".join(lane_debug_examples))

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
        active_centroid_x = float(np.mean(track.best_quad[:, 0]))
        matched_observations = [
            obs
            for obs in lane_observations
            if abs(obs.centroid_x - active_centroid_x) <= max(90.0, track.best_support_score * 140.0)
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

    max_ball_score = max((sel.ball_score for sel in track_selections), default=0.0)
    if max_ball_score >= 0.08:
        selected_track = max(
            track_selections,
            key=lambda sel: 0.35 * sel.lane_quality + 0.65 * sel.ball_score,
        )
    else:
        selected_track = max(track_selections, key=lambda sel: sel.lane_quality)

    active_lane = selected_track.track
    matched_observations = selected_track.matched_observations
    src_corners = selected_track.src_corners
    homography = cv2.getPerspectiveTransform(src_corners, dst)

    logger.info(
        "Lane track selection: %s",
        " | ".join(
            f"lane_q={sel.lane_quality:.3f} ball_q={sel.ball_score:.3f} "
            f"seen={sel.track.seen_count} votes={sel.track.ball_votes} "
            f"best_frame={sel.track.best_frame_idx}"
            for sel in sorted(
                track_selections,
                key=lambda item: 0.35 * item.lane_quality + 0.65 * item.ball_score,
                reverse=True,
            )[:8]
        ),
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
        frames_with_ball_masks, frames_with_ball, len(positions),
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
