from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.LaneBalls.Preprocessing import FrameSegmentation
from core.LaneBalls.models import BallPos, BallPosList


LANE_LENGTH_M = 18.288 
LANE_WIDTH_M = 1.0541  


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


def _order_corners_tl_tr_br_bl(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


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


def _mask_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _get_center_weighted_lane_mask(
    lane_masks: Sequence[np.ndarray], frame_shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    h, w = frame_shape
    merged = np.zeros((h, w), dtype=np.uint8)
    if not lane_masks:
        return merged, 0

    center_x = 0.5 * w
    scored: List[tuple[float, np.ndarray]] = []

    for lane_mask in lane_masks:
        mask = (lane_mask > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 64.0:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0.0:
                continue
            cx = m["m10"] / m["m00"]
            center_dist_ratio = abs(cx - center_x) / max(center_x, 1.0)
            score = center_dist_ratio - (area / (h * w))
            single = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(single, [c], -1, 1, thickness=-1)
            scored.append((score, single))

    if not scored:
        return merged, 0

    scored.sort(key=lambda x: x[0])
    selected = scored[:2]
    for _, single_mask in selected:
        merged = np.maximum(merged, single_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    return merged, len(selected)


def _extract_lane_corners(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    points = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
    hull = cv2.convexHull(points).reshape(-1, 2)

    peri = cv2.arcLength(hull.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 0.03 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(-1, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(hull.reshape(-1, 1, 2))
        quad = cv2.boxPoints(rect).astype(np.float32)

    return _order_corners_tl_tr_br_bl(quad)


def _is_trapezoid(corners: np.ndarray, frame_shape: tuple[int, int]) -> bool:
    h, w = frame_shape
    tl, tr, br, bl = corners
    top_width = float(np.linalg.norm(tr - tl))
    bottom_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))

    poly_area = cv2.contourArea(corners.astype(np.float32))
    min_area = 0.01 * float(h * w)

    if poly_area < min_area:
        return False
    if bottom_width <= top_width * 1.03:
        return False
    if left_height <= 8.0 or right_height <= 8.0:
        return False
    if tl[1] >= bl[1] or tr[1] >= br[1]:
        return False
    return True


def _build_homography_from_segmentations(
    segmentations_by_frame: Dict[int, FrameSegmentation],
    start_frame: int,
) -> tuple[HomographySelection, dict[str, float]]:
    if not segmentations_by_frame:
        raise ValueError("No segmentations provided")

    sorted_frames = sorted(k for k in segmentations_by_frame.keys() if k >= start_frame)
    if not sorted_frames:
        raise ValueError(f"No frames >= start_frame={start_frame}")

    frames_with_lane = 0
    scanned = 0
    coverage_values: List[float] = []

    for frame_index in sorted_frames:
        scanned += 1
        frame_seg = segmentations_by_frame[frame_index]
        lane_mask, n_selected = _get_center_weighted_lane_mask(
            frame_seg.lane_masks, frame_seg.frame_shape
        )
        coverage_values.append(_mask_coverage(lane_mask))
        if np.count_nonzero(lane_mask) == 0:
            continue
        frames_with_lane += 1

        corners = _extract_lane_corners(lane_mask)
        if corners is None:
            continue

        is_trapezoid = _is_trapezoid(corners, frame_seg.frame_shape)
        if not is_trapezoid:
            continue

        dst = _lane_dst_corners_m()
        H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)

        selection = HomographySelection(
            frame_index=frame_index,
            homography=H,
            src_corners=corners,
            dst_corners=dst,
            is_trapezoid=True,
            selected_lane_contours=n_selected,
        )
        metrics = {
            "frames_scanned_for_h": float(scanned),
            "frames_with_lane": float(frames_with_lane),
            "mean_lane_coverage_ratio": float(np.mean(coverage_values)) if coverage_values else 0.0,
        }
        return selection, metrics

    raise RuntimeError(
        "Failed to find a trapezoidal lane segmentation for homography "
        f"from start_frame={start_frame}"
    )


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


def _choose_ball_mask(ball_masks: Sequence[np.ndarray], frame_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not ball_masks:
        return None
    h, w = frame_shape
    center_x = 0.5 * w
    best_score = float("inf")
    best_mask: Optional[np.ndarray] = None
    for mask in ball_masks:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        cx = float(np.mean(xs))
        area = float(xs.size)
        score = abs(cx - center_x) / max(center_x, 1.0) - area / float(h * w)
        if score < best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _project_point_homography(pt_xy: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    vec = np.array([pt_xy[0], pt_xy[1], 1.0], dtype=np.float64)
    dst = H @ vec
    if abs(float(dst[2])) < 1e-9:
        raise RuntimeError("Invalid homography projection: denominator ~0")
    x = float(dst[0] / dst[2])
    y = float(dst[1] / dst[2])
    return x, y


def run_lane_ball_postprocessing(
    segmentations_by_frame: Dict[int, FrameSegmentation],
    fps: float,
    start_frame: int,
) -> PostprocessResult:
    selection, h_metrics = _build_homography_from_segmentations(
        segmentations_by_frame=segmentations_by_frame,
        start_frame=start_frame,
    )

    positions: List[BallPos] = []
    frames_with_ball = 0

    for frame_index in sorted(segmentations_by_frame.keys()):
        if frame_index < selection.frame_index:
            continue
        frame_seg = segmentations_by_frame[frame_index]
        best_ball = _choose_ball_mask(frame_seg.ball_masks, frame_seg.frame_shape)
        if best_ball is None:
            continue
        contact = _ball_contact_point_from_mask(best_ball)
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

    det = float(np.linalg.det(selection.homography))
    cond = float(np.linalg.cond(selection.homography))

    health = PostprocessHealth(
        frames_scanned_for_h=int(h_metrics["frames_scanned_for_h"]),
        frames_with_lane=int(h_metrics["frames_with_lane"]),
        frames_with_ball=int(frames_with_ball),
        lane_polygon_count_at_h=selection.selected_lane_contours,
        homography_determinant=det,
        homography_condition_number=cond,
        mean_lane_coverage_ratio=float(h_metrics["mean_lane_coverage_ratio"]),
    )

    return PostprocessResult(
        ball_positions=BallPosList(ball_positions=positions),
        homography_selection=selection,
        health=health,
    )
