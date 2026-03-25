from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from core.LaneBalls.models import BallPos
from core.LaneBalls.Postprocessing import LANE_LENGTH_M, LANE_WIDTH_M

logger = logging.getLogger("ciclopes.extrapolation")

# Lateral margin for hard OOB
_X_MARGIN = 0.02

# When the y-step between consecutive detections drops below this fraction
# of the running median y-step, the ball has clustered / stalled.
_STEP_DROP_RATIO = 0.25

# Minimum detections needed to establish a running median step size.
_MIN_HISTORY = 3

# How many trailing points to fit the departure line from.
_FIT_WINDOW = 5


def trim_raw_detections(
    positions: List[BallPos],
    lane_width_m: float = LANE_WIDTH_M,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    """
    Walk raw detections in frame order and cut the moment the step
    pattern breaks — either the ball clusters (y-step collapses) or
    drifts off the lane (x leaves bounds).

    This runs on the sparse raw detections BEFORE interpolation so the
    spline never sees the cluster.

    Returns:
        Trimmed list ending at the last legitimate detection.
    """
    if len(positions) < 2:
        return list(positions)

    sorted_pos = sorted(positions, key=lambda p: p.frame_index)

    x_lo = -_X_MARGIN
    x_hi = lane_width_m + _X_MARGIN

    kept: List[BallPos] = [sorted_pos[0]]
    y_steps: List[float] = []

    for i in range(1, len(sorted_pos)):
        prev = sorted_pos[i - 1]
        curr = sorted_pos[i]

        dy = curr.y_m - prev.y_m
        y_steps.append(dy)

        # ── Hard OOB ─────────────────────────────────────────────────────
        if curr.x_m < x_lo or curr.x_m > x_hi or curr.y_m > lane_length_m:
            break

        # ── Step-distance check ──────────────────────────────────────────
        # Once we have enough history, compare current y-step to the
        # running median.  A sudden collapse means the ball has arrived
        # at the pins / stalled.
        if len(y_steps) > _MIN_HISTORY:
            median_dy = float(np.median(y_steps[:-1]))  # exclude current

            # If the ball was making forward progress (median_dy > 0) and
            # the current step is a tiny fraction of that, it's a cluster.
            if median_dy > 0 and dy < median_dy * _STEP_DROP_RATIO:
                break

            # Ball going backward (bounced off pins / noise)
            if median_dy > 0 and dy < 0:
                break

        kept.append(curr)

    if len(kept) < len(sorted_pos):
        logger.info(
            "trim_raw_detections: %d→%d detections, cut at frame %d "
            "(last kept y=%.3f)",
            len(sorted_pos),
            len(kept),
            kept[-1].frame_index,
            kept[-1].y_m,
        )

    return kept


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

    tail = result[-min(_FIT_WINDOW, len(result)):]
    frames_t = np.array([p.frame_index for p in tail], dtype=np.float64)
    xs_t = np.array([p.x_m for p in tail], dtype=np.float64)
    ys_t = np.array([p.y_m for p in tail], dtype=np.float64)

    vx = _linear_slope(frames_t, xs_t)
    vy = _linear_slope(frames_t, ys_t)

    if float(np.hypot(vx, vy)) < 1e-4:
        return result

    last = result[-1]
    x_lo = -_X_MARGIN
    x_hi = lane_width_m + _X_MARGIN

    for df in range(1, 61):
        x = last.x_m + vx * df
        y = last.y_m + vy * df
        if x < x_lo or x > x_hi or y > lane_length_m or y < 0.0:
            safe_fps = max(fps, 1e-6)
            dep_frame = last.frame_index + df
            result.append(
                BallPos(
                    frame_index=dep_frame,
                    timestamp_s=float(dep_frame / safe_fps),
                    x_m=float(np.clip(x, x_lo, x_hi)),
                    y_m=float(np.clip(y, 0.0, lane_length_m)),
                )
            )
            logger.info(
                "Departure point: frame %d (x=%.3f y=%.3f)",
                dep_frame, result[-1].x_m, result[-1].y_m,
            )
            break

    return result


def _linear_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    coeffs = np.polyfit(frames, values, 1)
    return float(coeffs[0])
