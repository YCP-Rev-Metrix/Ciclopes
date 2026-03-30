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
    Exponential moving average on lane corner positions.
    Lanes are matched by centroid-x proximity and outlier corner jumps are rejected.
    """

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
                        ball_votes=0,
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

    def select_active_lane(self) -> Optional[TrackedLane]:
        if not self.tracks:
            return None

        tracks_with_ball_votes = [t for t in self.tracks if t.ball_votes > 0]
        if tracks_with_ball_votes:
            return max(tracks_with_ball_votes, key=lambda t: t.ball_votes)

        return max(
            self.tracks,
            key=lambda t: t.total_mask_area / max(t.seen_count, 1),
        )


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
    h, w = mask.shape
    edges = cv2.Canny(mask * 255, 50, 150)

    min_dim = min(h, w)
    lines = cv2.HoughLines(
        edges,
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
    width_bot = abs(q[2, 0] - q[3, 0])
    center_top = (q[0, 0] + q[1, 0]) / 2.0
    center_bot = (q[2, 0] + q[3, 0]) / 2.0

    taper = width_top / max(width_bot, 1.0)
    taper_score = 1.0 - min(abs(taper - 0.45) / 0.45, 1.0)

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


def build_lane_trapezoid(
    lane_mask: np.ndarray,
    *,
    frame_bgr: Optional[np.ndarray] = None,
    pins_boxes: Optional[Sequence[np.ndarray]] = None,
) -> Optional[TrapezoidCandidate]:
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

    if pins_boxes:
        pins_box = find_nearest_pins_box(cleaned, pins_boxes)
        if pins_box is not None:
            pins_bot_y = int(pins_box[3])
            pins_cx = float(pins_box[0] + pins_box[2]) / 2.0
            pins_half_w = float(pins_box[2] - pins_box[0]) / 2.0
            current_top_y = int(min(quad[0, 1], quad[1, 1]))

            if abs(pins_bot_y - current_top_y) < lane_mask.shape[0] * 0.3:
                quad[0, 1] = pins_bot_y
                quad[1, 1] = pins_bot_y

                lane_half_at_pins = pins_half_w * 1.7
                proposed_tl_x = pins_cx - lane_half_at_pins
                proposed_tr_x = pins_cx + lane_half_at_pins

                if abs(proposed_tl_x - quad[0, 0]) < 50:
                    quad[0, 0] = int(proposed_tl_x)
                if abs(proposed_tr_x - quad[1, 0]) < 50:
                    quad[1, 0] = int(proposed_tr_x)

    if frame_bgr is not None:
        quad = refine_sides_from_image(frame_bgr, quad)

    if not validate_vanishing_point(quad, lane_mask.shape):
        return None

    width_top = abs(int(quad[1, 0]) - int(quad[0, 0]))
    width_bottom = abs(int(quad[2, 0]) - int(quad[3, 0]))
    if width_top < 4 or width_bottom < 8:
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

    y_contact = int(np.max(ys))
    xs_at_contact = xs[ys == y_contact]
    if xs_at_contact.size == 0:
        return None
    x_contact = float(np.mean(xs_at_contact))
    return x_contact, float(y_contact)


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
    frames_bgr: Optional[Sequence[np.ndarray]] = None,
    min_trapezoid_score: float = DEFAULT_MIN_TRAPEZOID_SCORE,
) -> PostprocessResult:
    if not segmentations_by_frame:
        raise ValueError("No segmentations provided")

    sorted_frames = sorted(k for k in segmentations_by_frame.keys() if k >= start_frame)
    if not sorted_frames:
        raise ValueError(f"No frames >= start_frame={start_frame}")

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

        lane_area = float(sum(np.count_nonzero(m) for m in frame_seg.lane_masks))
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
            )
            if trap is None:
                continue
            if trap.score < min_trapezoid_score:
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
        raise RuntimeError(
            "Failed to select an active lane from segmentations "
            f"from start_frame={start_frame}"
        )

    logger.info(
        "Active lane: best_frame=%d best_score=%.3f ball_votes=%d seen=%d",
        active_lane.best_frame_idx, active_lane.best_score,
        active_lane.ball_votes, active_lane.seen_count,
    )

    src_corners = active_lane.best_quad.astype(np.float32)
    src_corners = _correct_trapezoid_top_corners(src_corners)
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
    frames_with_ball_masks = 0

    for frame_index in sorted(segmentations_by_frame.keys()):
        if frame_index < start_frame:
            continue

        frame_seg = segmentations_by_frame[frame_index]
        if frame_seg.ball_masks:
            frames_with_ball_masks += 1

        contact = _choose_ball_contact_for_lane(
            frame_seg.ball_masks,
            selection.src_corners,
        )
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
