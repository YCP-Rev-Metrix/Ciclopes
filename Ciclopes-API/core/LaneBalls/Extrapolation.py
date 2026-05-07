from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from core.LaneBalls.models import BallPos
from core.LaneBalls.Postprocessing import LANE_LENGTH_M, LANE_WIDTH_M

logger = logging.getLogger("ciclopes.extrapolation")

# Lateral margin for hard OOB
_X_MARGIN = 0.15

# When the y-step between consecutive detections drops below this fraction
# of the running median y-step, the ball has clustered / stalled.
_STEP_DROP_RATIO = 0.25

# Minimum detections needed to establish a running median step size.
_MIN_HISTORY = 3

# How many trailing points to fit the departure curve from.
_FIT_WINDOW = 5
_CURVE_FIT_WINDOW = 14
_MAX_DEPARTURE_DX_M = 0.28

_CONTACT_Y_TOL_M = 0.08
_END_Y_TOL_M = 0.35


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
            "trim_raw_detections: %d->%d detections, kept frames %s-%s",
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


def append_departure_point(
    positions: List[BallPos],
    fps: float,
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    """
    After trimming and interpolation, append a single extrapolated point
    at the lane boundary (pins or gutter) based on the tail trajectory.

    Returns:
        Positions with at most one appended departure point.
    """
    if len(positions) < 2:
        return list(positions)

    result = list(positions)

    tail = result[-min(_CURVE_FIT_WINDOW, len(result)):]
    frames_t = np.array([p.frame_index for p in tail], dtype=np.float64)
    xs_t = np.array([p.x_m for p in tail], dtype=np.float64)
    ys_t = np.array([p.y_m for p in tail], dtype=np.float64)

    vy = _linear_slope(frames_t[-min(_FIT_WINDOW, len(frames_t)):], ys_t[-min(_FIT_WINDOW, len(ys_t)):])
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
        dep_frame, result[-1].x_m, result[-1].y_m,
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


def _linear_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    coeffs = np.polyfit(frames, values, 1)
    return float(coeffs[0])
