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
_CURVE_FIT_WINDOW = 18
_MAX_DEPARTURE_DX_M = 0.36

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

    # ── Loft-resume prefix trim ──────────────────────────────────────────
    # If there's a transient stall near the start (ball lofted, hit, briefly
    # near-stationary, then resumed rolling), drop everything up to where
    # consistent forward motion resumes. Clean trajectories have no such
    # stall and are untouched.
    if b - a >= 8:
        kept_y = np.asarray([sorted_pos[i].y_m for i in range(a, b)], dtype=np.float64)
        kept_dy = np.diff(kept_y)
        if kept_dy.size >= 6:
            mid_lo = int(len(kept_dy) * 0.20)
            mid_hi = int(len(kept_dy) * 0.80)
            mid = kept_dy[mid_lo:mid_hi] if mid_hi > mid_lo else kept_dy
            mid_pos = mid[mid > 0]
            if mid_pos.size >= 3:
                median_dy = float(np.median(mid_pos))
                stall_thr = 0.30 * median_dy
                resume_thr = 0.45 * median_dy
                # locate a stall (>=2 consecutive small/negative dy) in the
                # first 40% of the kept window
                scan_end = max(2, int(len(kept_dy) * 0.40))
                stall_end = -1  # local index into kept_dy
                streak = 0
                for k in range(scan_end):
                    if kept_dy[k] < stall_thr:
                        streak += 1
                        if streak >= 2:
                            stall_end = k
                    else:
                        streak = 0
                if stall_end >= 0:
                    # find first index after stall where 3 consecutive forward
                    # steps each clear resume_thr
                    for k in range(stall_end + 1, len(kept_dy) - 2):
                        if all(kept_dy[k + j] >= resume_thr for j in range(3)):
                            a = a + k
                            break

    # ── Tail-cluster trim ────────────────────────────────────────────────
    # Drop trailing samples whose dy collapses well below median (ball
    # detector latched onto pins / debris) or whose dx spikes far beyond
    # the run's lateral jitter.
    if b - a >= 8:
        kept_y = np.asarray([sorted_pos[i].y_m for i in range(a, b)], dtype=np.float64)
        kept_x = np.asarray([sorted_pos[i].x_m for i in range(a, b)], dtype=np.float64)
        kept_dy = np.diff(kept_y)
        kept_dx = np.diff(kept_x)
        mid_lo = int(len(kept_dy) * 0.20)
        mid_hi = max(mid_lo + 1, int(len(kept_dy) * 0.80))
        mid_dy_pos = kept_dy[mid_lo:mid_hi]
        mid_dy_pos = mid_dy_pos[mid_dy_pos > 0]
        mid_dx_std = float(np.std(kept_dx[mid_lo:mid_hi])) if mid_hi > mid_lo else 0.0
        if mid_dy_pos.size >= 3:
            median_dy = float(np.median(mid_dy_pos))
            stall_thr = 0.25 * median_dy
            dx_spike = max(2.5 * mid_dx_std, 0.08)
            # walk back conservatively, capped at 6 trims and never below 6 left
            trims = 0
            while (b - a) > 6 and trims < 6:
                last_dy = kept_dy[-1]
                last_dx = abs(kept_dx[-1])
                if last_dy < stall_thr or last_dx > dx_spike:
                    b -= 1
                    kept_dy = kept_dy[:-1]
                    kept_dx = kept_dx[:-1]
                    trims += 1
                else:
                    break

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

    # If the tail has stalled (interp flattened by isotonic), fall back to
    # the median per-frame dy across the whole run so we still extrapolate
    # to the lane end at a plausible speed instead of bailing out.
    full_ys = np.array([p.y_m for p in result], dtype=np.float64)
    full_dy = np.diff(full_ys)
    full_dy_pos = full_dy[full_dy > 1e-4]
    median_vy = float(np.median(full_dy_pos)) if full_dy_pos.size >= 3 else 0.0
    if vy < 0.30 * median_vy and median_vy > 1e-4:
        vy = median_vy

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

    # Tail-weighted least-squares: the late (more-curved) part of a hook is
    # what continues to the pins, so weight recent samples more heavily.
    weights = np.linspace(0.5, 1.5, y.size)

    linear = np.polyfit(y, x, 1, w=weights)
    x_linear = float(np.polyval(linear, target_y))

    # ── Committed hook direction (inertia) ─────────────────────────────
    # A bowling ball cannot reverse hook direction in the last fraction of
    # the trajectory — too much inertia. Detect the *committed* direction
    # from the long baseline (early-third median x → late-third median x)
    # and use it to (a) reject prediction reversals, (b) exaggerate when a
    # real hook is present.
    hook_sign = 0.0
    hook_strength = 0.0
    if y.size >= 9:
        third = y.size // 3
        early_x = float(np.median(x[:third]))
        late_x = float(np.median(x[-third:]))
        global_dx = late_x - early_x
        early_y = float(np.median(y[:third]))
        late_y = float(np.median(y[-third:]))
        global_dy = late_y - early_y
        if abs(global_dx) >= 0.04 and global_dy > 1.0:
            hook_sign = 1.0 if global_dx > 0 else -1.0
            hook_strength = abs(global_dx)

    if y.size < 5 or y_span < 1.0:
        # Even in the linear-only branch, don't allow predictions that
        # reverse a clearly committed hook.
        if hook_sign != 0.0 and (x_linear - float(last_x)) * hook_sign < 0.0:
            return float(last_x)
        return x_linear

    quad = np.polyfit(y, x, 2, w=weights)
    x_quad = float(np.polyval(quad, target_y))
    curvature_signed = float(quad[0])
    curvature = abs(curvature_signed)
    if not np.isfinite(x_quad) or curvature > 0.20:
        if hook_sign != 0.0 and (x_linear - float(last_x)) * hook_sign < 0.0:
            return float(last_x)
        return x_linear

    # If the quadratic curvature direction agrees with the recent dx trend
    # AND with the committed global hook, lean harder on the quadratic.
    quad_blend = 0.75  # bumped from 0.65 — slightly more curve credit
    exaggerate = 0.0   # extra push past the quadratic when hook confirmed
    if y.size >= 6:
        tail_n = max(3, y.size // 3)
        tail_dx = float(x[-1] - x[-tail_n])
        tail_dy = float(y[-1] - y[-tail_n])
        if abs(tail_dy) > 1e-6:
            recent_slope = tail_dx / tail_dy
            full_slope = float(linear[0])
            slope_delta = recent_slope - full_slope
            curvature_agrees = curvature_signed * slope_delta > 0.0
            # Hook is "confirmed" when curvature, recent slope deviation,
            # and the global committed direction all agree.
            global_agrees = (
                hook_sign != 0.0
                and curvature_signed * hook_sign > 0.0
            )
            if curvature_agrees and abs(slope_delta) > 0.02:
                quad_blend = 0.90
            if global_agrees:
                # Push slightly past the quadratic prediction in the same
                # direction the ball is already curving — this captures the
                # late-stage hook acceleration that a 2nd-order fit under-
                # predicts. 15% of the deviation from the linear baseline.
                exaggerate = 0.15

    blended = quad_blend * x_quad + (1.0 - quad_blend) * x_linear
    if exaggerate != 0.0:
        blended = blended + exaggerate * (x_quad - x_linear)

    # Inertia: never let the prediction reverse the committed hook.
    if hook_sign != 0.0:
        if (blended - float(last_x)) * hook_sign < 0.0:
            blended = float(last_x)

    return float(blended)


def _linear_slope(frames: np.ndarray, values: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    coeffs = np.polyfit(frames, values, 1)
    return float(coeffs[0])
