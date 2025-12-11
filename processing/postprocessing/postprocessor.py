from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Literal
from scipy import interpolate
import warnings


# ---------- Geometry / Warping ----------

class Warp:
    """Homography and mask warping utilities for BEV transformation."""
    
    # Class-level flag to only warn once about negative determinant
    _warned_negative_det: bool = False

    @staticmethod
    def compute_homography(
        src_pts: np.ndarray, 
        dst_pts: np.ndarray,
        min_inlier_ratio: float = 0.25,  # Relaxed from 0.5 for real-world robustness
        max_condition_number: float = 1e8,  # Relaxed from 1e6
    ) -> np.ndarray:
        """
        Compute homography with RANSAC and validate the result.
        
        Parameters
        ----------
        src_pts : np.ndarray
            Source points (N, 2)
        dst_pts : np.ndarray
            Destination points (N, 2)
        min_inlier_ratio : float
            Minimum ratio of inliers required (0.0 to 1.0)
        max_condition_number : float
            Maximum condition number before matrix is considered degenerate
            
        Returns
        -------
        np.ndarray
            3x3 homography matrix
        """
        H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            raise RuntimeError("cv2.findHomography failed. Check your correspondences.")
        
        # Check inlier ratio
        if status is not None and len(status) > 0:
            inlier_ratio = float(np.sum(status)) / len(status)
            if inlier_ratio < min_inlier_ratio:
                raise RuntimeError(
                    f"Homography inlier ratio {inlier_ratio:.2%} is below threshold "
                    f"{min_inlier_ratio:.2%}. Check point correspondences."
                )
        
        # Check for degenerate matrix (near-singular)
        try:
            cond = np.linalg.cond(H)
            if cond > max_condition_number:
                raise RuntimeError(
                    f"Homography condition number {cond:.2e} exceeds threshold "
                    f"{max_condition_number:.2e}. Matrix is near-singular."
                )
        except np.linalg.LinAlgError:
            raise RuntimeError("Homography matrix is singular.")
        
        # Check determinant sign (negative indicates reflection - warn only once)
        det = np.linalg.det(H)
        if det < 0 and not Warp._warned_negative_det:
            Warp._warned_negative_det = True
            warnings.warn(
                f"Homography has negative determinant ({det:.4f}), indicating a reflection. "
                "This is unusual for camera-to-BEV transforms.",
                UserWarning
            )
        
        return H

    @staticmethod
    def warp_mask(mask: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask.squeeze()
        return cv2.warpPerspective(mask, H, out_size, flags=cv2.INTER_NEAREST)

    @staticmethod
    def image_to_bev_point(pt_xy: Tuple[float, float], H: np.ndarray) -> Optional[Tuple[float, float]]:
        """Transform a single point from image coordinates to BEV coordinates."""
        x, y = pt_xy
        vec = np.array([x, y, 1.0], dtype=np.float64).reshape(3, 1)
        dst = H @ vec
        w = float(dst[2, 0])
        if abs(w) < 1e-9:
            return None
        return float(dst[0, 0] / w), float(dst[1, 0] / w)

    @staticmethod
    def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Compute centroid of nonzero pixels in a mask."""
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    # ----- Lane-quad detection (for auto-H from first lane mask) -----

    @staticmethod
    def _order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    @staticmethod
    def quad_from_lane_mask(lane_mask: np.ndarray) -> Optional[np.ndarray]:
        if lane_mask.ndim == 3:
            lane_mask = lane_mask.squeeze()

        # Binary cleanup to stabilize corners
        bin_mask = (lane_mask > 0).astype(np.uint8) * 255
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        lane_cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(lane_cnt, True)

        # Try polygonal approximation
        approx = cv2.approxPolyDP(lane_cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return Warp._order_corners_clockwise(quad)

        # Fallback: oriented rectangle
        rect = cv2.minAreaRect(lane_cnt)
        box = cv2.boxPoints(rect).astype(np.float32)
        return Warp._order_corners_clockwise(box)

    @staticmethod
    def default_bev_corners(out_size: Tuple[int, int]) -> np.ndarray:
        W, H = out_size
        return np.array([[0, 0],
                         [W - 1, 0],
                         [W - 1, H - 1],
                         [0, H - 1]], dtype=np.float32)


# ---------- Temporal buffer & kinematics ----------

@dataclass
class _TimestampedPosition:
    """Position with associated frame index for proper dt calculation."""
    x: float
    y: float
    frame_idx: int


class BufferCalcs:
    """
    Temporal buffer for tracking positions, velocities, and accelerations.
    
    Properly handles missing frames by storing frame indices with each position
    and computing velocity using the actual time gap between observations.
    """

    def __init__(self, buffer_len: int = 12, dt: float = 1.0 / 30.0):
        self.buffer_len = int(buffer_len)
        self.dt = float(dt)
        self._positions: List[_TimestampedPosition] = []
        self._velocities: List[Tuple[float, float, int]] = []  # (vx, vy, frame_idx)
        self._accelerations: List[Tuple[float, float]] = []

    def reset(self) -> None:
        """Clear all buffered data."""
        self._positions.clear()
        self._velocities.clear()
        self._accelerations.clear()

    def add(self, pos: Optional[Tuple[float, float]], frame_idx: int) -> None:
        """
        Add a position observation.
        
        Parameters
        ----------
        pos : Optional[Tuple[float, float]]
            Position (x, y) or None if no detection
        frame_idx : int
            Frame index for proper temporal alignment
        """
        if pos is None:
            return

        # Validate position is finite
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            raise ValueError(f"Non-finite position detected: {pos}")

        ts_pos = _TimestampedPosition(x=pos[0], y=pos[1], frame_idx=frame_idx)
        self._positions.append(ts_pos)
        if len(self._positions) > self.buffer_len:
            self._positions.pop(0)

        # Compute velocity using actual frame gap
        if len(self._positions) >= 2:
            curr = self._positions[-1]
            prev = self._positions[-2]
            frame_gap = curr.frame_idx - prev.frame_idx
            
            if frame_gap > 0:
                effective_dt = self.dt * frame_gap
                vx = (curr.x - prev.x) / effective_dt
                vy = (curr.y - prev.y) / effective_dt
                self._velocities.append((vx, vy, frame_idx))
                if len(self._velocities) > self.buffer_len:
                    self._velocities.pop(0)

        # Compute acceleration
        if len(self._velocities) >= 2:
            vx2, vy2, idx2 = self._velocities[-1]
            vx1, vy1, idx1 = self._velocities[-2]
            frame_gap = idx2 - idx1
            
            if frame_gap > 0:
                effective_dt = self.dt * frame_gap
                ax = (vx2 - vx1) / effective_dt
                ay = (vy2 - vy1) / effective_dt
                self._accelerations.append((ax, ay))
                if len(self._accelerations) > self.buffer_len:
                    self._accelerations.pop(0)

    @property
    def positions(self) -> List[Tuple[float, float]]:
        """Return positions as simple (x, y) tuples."""
        return [(p.x, p.y) for p in self._positions]

    def latest_position(self) -> Optional[Tuple[float, float]]:
        if not self._positions:
            return None
        p = self._positions[-1]
        return (p.x, p.y)

    def latest_velocity(self) -> Optional[Tuple[float, float]]:
        if not self._velocities:
            return None
        vx, vy, _ = self._velocities[-1]
        return (vx, vy)

    def latest_acceleration(self) -> Optional[Tuple[float, float]]:
        return self._accelerations[-1] if self._accelerations else None

    def trajectory(self) -> List[Tuple[float, float]]:
        return self.positions


# ---------- High-level orchestration ----------

@dataclass
class ProcessResult:
    """Result of processing a single frame."""
    bev_centroid: Optional[Tuple[float, float]]
    velocity: Optional[Tuple[float, float]]
    acceleration: Optional[Tuple[float, float]]
    warped_lane: np.ndarray
    warped_ball: np.ndarray


@dataclass
class LaneMetrics:
    """
    Summary metrics for a single lane run in BEV space.

    All distances and speeds are expressed in BEV units; tests or downstream
    consumers can map them to world coordinates using the homography-derived
    calibration used elsewhere in the project.
    """

    fractions: Tuple[float, ...]
    frac_positions: Tuple[float, ...]
    velocity_at_frac: Tuple[Tuple[float, float, float], ...]  # (vs, vl, |v|)
    acceleration_at_frac: Tuple[Tuple[float, float, float], ...]  # (as, al, |a|)
    total_break: float  # lateral |pos_end - pos_start|
    end_lane_speed: float
    # Track axis orientation for consistent sign handling in calibration
    primary_axis: int = 0  # 0 = x-axis is primary (along-lane), 1 = y-axis
    trajectory_reversed: bool = False  # True if trajectory was reversed for increasing s


class PostProcessor:
    """
    BEV mask processing and lane metrics computation.
    
    Supports optional trajectory interpolation for smoother velocity/acceleration
    estimates.
    """

    def __init__(
        self,
        out_size: Tuple[int, int] = (400, 800),
        dt: float = 1.0 / 30.0,
        buffer_len: int = 12,
        interpolation_mode: Literal["none", "linear", "cubic"] = "none",
        centroid_from_warped: bool = True,  # Whether to compute centroid from warped mask
    ):
        """
        Parameters
        ----------
        out_size : Tuple[int, int]
            BEV output size (width, height)
        dt : float
            Time step between frames (1/fps)
        buffer_len : int
            Maximum positions to keep in temporal buffer
        interpolation_mode : str
            Interpolation for metrics: "none", "linear", or "cubic"
        centroid_from_warped : bool
            If True, compute centroid from warped BEV mask (more accurate for BEV).
            If False, transform image centroid via homography (faster, sub-pixel).
        """
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.dt = float(dt)
        self._buffer_len = buffer_len
        self.buffer = BufferCalcs(buffer_len=buffer_len, dt=self.dt)
        self.interpolation_mode = interpolation_mode
        self.centroid_from_warped = centroid_from_warped

    def _compute_H_once(
        self,
        first_lane_mask: np.ndarray,
        homography_src_dst: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        if homography_src_dst is not None:
            src_pts, dst_pts = homography_src_dst
            return Warp.compute_homography(src_pts.astype(np.float32), dst_pts.astype(np.float32))

        # Auto from the lane mask quad
        quad = Warp.quad_from_lane_mask(first_lane_mask)
        if quad is None or quad.shape != (4, 2):
            raise RuntimeError("Could not infer lane quadrilateral from first lane mask. "
                               "Provide homography_src_dst=(src_pts, dst_pts).")

        dst_rect = Warp.default_bev_corners(self.out_size)
        return Warp.compute_homography(quad, dst_rect)

    def process_run(
        self,
        masks_by_index: Dict[int, Dict[str, np.ndarray]],
        homography_src_dst: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[Dict[int, ProcessResult], np.ndarray]:
        """
        Process a sequence of masks and compute BEV centroids and kinematics.
        
        Parameters
        ----------
        masks_by_index : Dict[int, Dict[str, np.ndarray]]
            {frame_idx: {"ball": mask, "lane": mask}}
        homography_src_dst : Optional[Tuple[np.ndarray, np.ndarray]]
            If provided, use these correspondences for homography instead of auto-detecting
            
        Returns
        -------
        results : Dict[int, ProcessResult]
            Per-frame results
        H : np.ndarray
            3x3 homography matrix used
        """
        # Reset buffer for each new episode to prevent cross-episode contamination
        self.buffer = BufferCalcs(buffer_len=self._buffer_len, dt=self.dt)
        
        if not masks_by_index:
            return {}, np.eye(3, dtype=np.float64)

        # Compute H once from the first available lane mask
        first_idx = sorted(masks_by_index.keys())[0]
        first_lane = masks_by_index[first_idx]["lane"]
        H = self._compute_H_once(first_lane, homography_src_dst)

        results: Dict[int, ProcessResult] = {}

        for idx in sorted(masks_by_index.keys()):
            ball_mask = masks_by_index[idx]["ball"]
            lane_mask = masks_by_index[idx]["lane"]

            warped_lane = Warp.warp_mask(lane_mask, H, self.out_size)
            warped_ball = Warp.warp_mask(ball_mask, H, self.out_size)

            # Compute BEV centroid
            if self.centroid_from_warped:
                # Compute directly from warped mask - more accurate for BEV metrics
                bev_centroid = Warp.centroid_from_mask(warped_ball)
            else:
                # Transform image centroid via homography - faster, preserves sub-pixel
                img_centroid = Warp.centroid_from_mask(ball_mask)
                bev_centroid = Warp.image_to_bev_point(img_centroid, H) if img_centroid else None

            # Update temporal buffer with frame index for proper dt calculation
            self.buffer.add(bev_centroid, frame_idx=idx)
            vel = self.buffer.latest_velocity()
            acc = self.buffer.latest_acceleration()

            results[idx] = ProcessResult(
                bev_centroid=bev_centroid,
                velocity=vel,
                acceleration=acc,
                warped_lane=warped_lane,
                warped_ball=warped_ball
            )

        return results, H

    # ------------------------------------------------------------------
    # Higher-level lane metrics for test evaluation
    # ------------------------------------------------------------------
    def _apply_interpolation(
        self,
        t: np.ndarray,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply interpolation to trajectory data based on interpolation_mode.
        
        Parameters
        ----------
        t : np.ndarray
            Time values in seconds
        data : np.ndarray
            Data to interpolate (N, 2) positions
            
        Returns
        -------
        t_interp, data_interp : Tuple[np.ndarray, np.ndarray]
            Interpolated time and data values
        """
        if self.interpolation_mode == "none" or len(t) < 3:
            return t, data
        
        # Create denser time grid (3x more points)
        n_interp = len(t) * 3
        t_interp = np.linspace(t[0], t[-1], n_interp)
        
        if self.interpolation_mode == "linear":
            interp_func = interpolate.interp1d(
                t, data, kind='linear', axis=0,
                bounds_error=False, fill_value=(data[0], data[-1])
            )
            data_interp = interp_func(t_interp)
            
        elif self.interpolation_mode == "cubic":
            if len(t) < 4:
                # Fall back to linear if not enough points
                interp_func = interpolate.interp1d(
                    t, data, kind='linear', axis=0,
                    bounds_error=False, fill_value=(data[0], data[-1])
                )
                data_interp = interp_func(t_interp)
            else:
                # Cubic spline with natural boundary conditions
                interp_func = interpolate.CubicSpline(
                    t, data, axis=0, bc_type='natural', extrapolate=False
                )
                data_interp = interp_func(t_interp)
                # Handle NaN from extrapolate=False
                nan_mask = np.isnan(data_interp)
                if nan_mask.any():
                    for i in range(data_interp.shape[1]):
                        col = data_interp[:, i]
                        col[np.isnan(col) & (t_interp < t[0])] = data[0, i]
                        col[np.isnan(col) & (t_interp > t[-1])] = data[-1, i]
        else:
            return t, data
            
        return t_interp, data_interp

    def compute_lane_metrics(
        self,
        results_by_index: Dict[int, ProcessResult],
        fractions: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    ) -> LaneMetrics:
        """
        Compute velocity/acceleration at fixed fractions along the lane,
        total break (lateral offset from start to end), and end-of-lane speed.

        We treat the dominant BEV axis of motion (x or y) as the lane axis and
        the other axis as lateral.
        
        Parameters
        ----------
        results_by_index : Dict[int, ProcessResult]
            Per-frame processing results
        fractions : Tuple[float, ...]
            Lane fractions at which to report metrics
            
        Returns
        -------
        LaneMetrics
            Computed metrics
        """
        if not results_by_index:
            raise ValueError("results_by_index is empty")

        # Collect BEV centroids in temporal order, skipping missing ones
        idxs = sorted(results_by_index.keys())
        traj: List[Tuple[float, float]] = []
        frame_times: List[float] = []
        
        for i in idxs:
            c = results_by_index[i].bev_centroid
            if c is not None:
                traj.append(c)
                frame_times.append(float(i) * self.dt)

        if len(traj) < 2:
            raise ValueError("Not enough valid centroids to compute metrics")

        bev = np.array(traj, dtype=np.float64)
        t_raw = np.array(frame_times, dtype=np.float64)
        
        # Apply interpolation if enabled
        t, bev_interp = self._apply_interpolation(t_raw, bev)

        # Validate interpolation result
        if not np.all(np.isfinite(bev_interp)):
            raise ValueError(
                f"Non-finite values after interpolation. "
                f"NaN: {np.sum(np.isnan(bev_interp))}, Inf: {np.sum(np.isinf(bev_interp))}"
            )

        xs = bev_interp[:, 0]
        ys = bev_interp[:, 1]

        # Choose primary axis as the one with larger dynamic range
        range_x = float(xs.max() - xs.min())
        range_y = float(ys.max() - ys.min())
        primary_axis = 0 if range_x >= range_y else 1

        s_raw = xs if primary_axis == 0 else ys  # along-lane coordinate
        l_raw = ys if primary_axis == 0 else xs  # lateral coordinate

        # If trajectory runs in negative direction, REVERSE all arrays
        # This is cleaner than negating s and tracking a sign flag
        trajectory_reversed = False
        if s_raw[-1] < s_raw[0]:
            s = s_raw[::-1].copy()
            l = l_raw[::-1].copy()
            t = t[::-1].copy()
            trajectory_reversed = True
        else:
            s = s_raw.copy()
            l = l_raw.copy()

        lane_len = float(s[-1] - s[0])
        if lane_len <= 0.0:
            raise ValueError("Non-positive lane length in BEV coordinates")

        # Compute time step for gradient calculation
        # CRITICAL: Use abs() because if trajectory was reversed, t is also reversed,
        # making t[-1] - t[0] negative. We need positive dt for correct velocity signs.
        if len(t) > 1:
            dt_for_gradient = abs(t[-1] - t[0]) / (len(t) - 1)
        else:
            dt_for_gradient = self.dt
        
        # Defensive check: dt must be positive and reasonable
        if dt_for_gradient <= 0.0:
            raise ValueError(f"Invalid dt_for_gradient: {dt_for_gradient}. Time array may be malformed.")
        
        # Finite-difference velocities and accelerations in BEV space
        # After potential reversal, s is guaranteed to be increasing, so vs should be positive
        vs = np.gradient(s, dt_for_gradient)
        vl = np.gradient(l, dt_for_gradient)
        a_s = np.gradient(vs, dt_for_gradient)
        a_l = np.gradient(vl, dt_for_gradient)
        
        # Validate derivatives
        if not (np.all(np.isfinite(vs)) and np.all(np.isfinite(vl))):
            raise ValueError(
                f"Non-finite velocity values. vs NaN: {np.sum(np.isnan(vs))}, "
                f"vl NaN: {np.sum(np.isnan(vl))}"
            )
        
        if not (np.all(np.isfinite(a_s)) and np.all(np.isfinite(a_l))):
            raise ValueError(
                f"Non-finite acceleration values. a_s NaN: {np.sum(np.isnan(a_s))}, "
                f"a_l NaN: {np.sum(np.isnan(a_l))}"
            )
        
        # Sanity check: since s is guaranteed increasing (after potential reversal),
        # the mean velocity along s should be positive. Warn if not.
        mean_vs = float(np.mean(vs))
        if mean_vs < 0:
            import warnings
            warnings.warn(
                f"Mean along-lane velocity is negative ({mean_vs:.4f}) despite trajectory "
                f"normalization. This may indicate a coordinate system issue."
            )

        frac_list = []
        frac_positions = []
        vel_at_frac = []
        acc_at_frac = []

        for f in fractions:
            f_clamped = float(max(0.0, min(1.0, f)))
            target_s = s[0] + f_clamped * lane_len
            idx_closest = int(np.argmin(np.abs(s - target_s)))

            vs_f = float(vs[idx_closest])
            vl_f = float(vl[idx_closest])
            as_f = float(a_s[idx_closest])
            al_f = float(a_l[idx_closest])

            speed_f = float(np.hypot(vs_f, vl_f))
            acc_mag_f = float(np.hypot(as_f, al_f))

            frac_list.append(f_clamped)
            frac_positions.append(float(s[idx_closest]))
            vel_at_frac.append((vs_f, vl_f, speed_f))
            acc_at_frac.append((as_f, al_f, acc_mag_f))

        # Total break: lateral deviation from start to end
        total_break = float(abs(l[-1] - l[0]))
        # End-of-lane speed from last sample
        end_speed = float(np.hypot(vs[-1], vl[-1]))

        return LaneMetrics(
            fractions=tuple(frac_list),
            frac_positions=tuple(frac_positions),
            velocity_at_frac=tuple(vel_at_frac),
            acceleration_at_frac=tuple(acc_at_frac),
            total_break=total_break,
            end_lane_speed=end_speed,
            primary_axis=primary_axis,
            trajectory_reversed=trajectory_reversed,
        )
