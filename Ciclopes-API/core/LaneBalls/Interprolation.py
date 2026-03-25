from __future__ import annotations

import logging
from typing import List

import numpy as np
from scipy.interpolate import UnivariateSpline

from core.LaneBalls.models import BallPos

logger = logging.getLogger("ciclopes.interpolation")

# Smoothing factor — higher = smoother curve, lower = closer to raw points.
# This is passed as scipy's `s` parameter (sum-of-squared-residuals budget).
# For lane-metre coordinates the raw noise is typically ~0.02–0.08 m per
# detection, so s ≈ N * σ² works well.  We derive it from the data below.
_SMOOTH_SIGMA_M = 0.04  # assumed per-detection noise in metres


def interpolate_ball_positions(
    positions: List[BallPos],
    fps: float,
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
    else:
        # Too few points for a meaningful smooth — linear fallback
        smooth_x = np.interp(all_frames, frames, xs)
        smooth_y = np.interp(all_frames, frames, ys)

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
