from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.interpolate import UnivariateSpline

from core.LaneBalls.models import BallPos
from core.LaneBalls.Postprocessing import LANE_LENGTH_M

logger = logging.getLogger("ciclopes.interpolation")

# Smoothing factor — higher = smoother curve, lower = closer to raw points.
# This is passed as scipy's `s` parameter (sum-of-squared-residuals budget).
# For lane-metre coordinates the raw noise is typically ~0.02–0.08 m per
# detection, so s ≈ N * σ² works well.  We derive it from the data below.
_SMOOTH_SIGMA_M = 0.04  # assumed per-detection noise in metres


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
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


def interpolate_ball_positions(
    positions: List[BallPos],
    fps: float,
    lane_length_m: float = LANE_LENGTH_M,
) -> List[BallPos]:
    """
    Smooth and densify sparse ball detections using a smoothing spline.

    Unlike a plain CubicSpline (which passes through every noisy point),
    UnivariateSpline fits a smooth curve that *approximates* the raw
    detections — removing jitter from detection noise and homography error.

    Produces one BallPos per frame between the first and last detection.

    Args:
        positions: Sparse ball positions from detection + homography.
        fps:       Video frame rate.

    Returns:
        Dense, smoothed list of BallPos.
    """
    if len(positions) < 2:
        return list(positions)

    sorted_pos = sorted(positions, key=lambda p: p.frame_index)

    # Deduplicate by frame_index — keep the first occurrence
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
        # Smoothing budget: s = N * σ² — lets the spline deviate from raw
        # points by up to ~σ on average, which removes detection jitter.
        s = n * (_SMOOTH_SIGMA_M ** 2)

        # Spline degree 3 (cubic), smoothing factor s
        spline_x = UnivariateSpline(frames, xs, k=3, s=s)
        spline_y = UnivariateSpline(frames, ys, k=3, s=s)
        smooth_x = spline_x(all_frames)
        smooth_y = spline_y(all_frames)
        raw_x = np.interp(all_frames, frames, xs)
        raw_y = np.interp(all_frames, frames, ys)
        smooth_x = _anchor_endpoint_values(np.asarray(smooth_x, dtype=np.float64), raw_x)
        smooth_y = _anchor_endpoint_values(np.asarray(smooth_y, dtype=np.float64), raw_y)
    else:
        # Too few points for a meaningful smooth — linear fallback
        smooth_x = np.interp(all_frames, frames, xs)
        smooth_y = np.interp(all_frames, frames, ys)

    smooth_y = _isotonic_non_decreasing(np.asarray(smooth_y, dtype=np.float64))
    smooth_y = _spread_flat_forward_segments(smooth_y)
    smooth_y = np.clip(smooth_y, 0.0, lane_length_m)

    safe_fps = max(fps, 1e-6)
    dense: List[BallPos] = []
    for i, f in enumerate(all_frames):
        fi = int(f)
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
        len(sorted_pos), len(dense), first_frame, last_frame,
    )
    return dense
