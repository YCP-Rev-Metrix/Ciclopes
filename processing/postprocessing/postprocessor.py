from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Literal
from scipy import interpolate


# ---------- Geometry / Warping ----------

class Warp:
    pass

    @staticmethod
    def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
        H, status = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        if H is None:
            raise RuntimeError("cv2.findHomography failed. Check your correspondences.")
        return H

    @staticmethod
    def warp_mask(mask: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask.squeeze()
        return cv2.warpPerspective(mask, H, out_size, flags=cv2.INTER_NEAREST)

    @staticmethod
    def image_to_bev_point(pt_xy: Tuple[float, float], H: np.ndarray) -> Optional[Tuple[float, float]]:
        x, y = pt_xy
        vec = np.array([x, y, 1.0], dtype=np.float64).reshape(3, 1)
        dst = H @ vec
        w = float(dst[2, 0])
        if abs(w) < 1e-9:
            return None
        return float(dst[0, 0] / w), float(dst[1, 0] / w)

    @staticmethod
    def centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float]]:
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

        # Binary cleanup to stabilize corners a bit
        bin_mask = (lane_mask > 0).astype(np.uint8) * 255
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        lane_cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(lane_cnt, True)

        # try polygonal approximation
        approx = cv2.approxPolyDP(lane_cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return Warp._order_corners_clockwise(quad)

        # fallback: oriented rectangle
        rect = cv2.minAreaRect(lane_cnt)  # ((cx, cy), (w, h), angle)
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

class BufferCalcs:
    pass

    def __init__(self, buffer_len: int = 12, dt: float = 1.0 / 30.0):
        self.buffer_len = int(buffer_len)
        self.dt = float(dt)
        self.positions: List[Tuple[float, float]] = []
        self.velocities: List[Tuple[float, float]] = []
        self.accelerations: List[Tuple[float, float]] = []

    def add(self, pos: Optional[Tuple[float, float]]) -> None:
        if pos is None:
            return  # drop missing detections; you could also repeat-last if you prefer

        # Validate position is finite
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            raise ValueError(f"Non-finite position detected: {pos}")

        self.positions.append(pos)
        if len(self.positions) > self.buffer_len:
            self.positions.pop(0)

        # velocity
        if len(self.positions) >= 2:
            x2, y2 = self.positions[-1]
            x1, y1 = self.positions[-2]
            vx = (x2 - x1) / self.dt
            vy = (y2 - y1) / self.dt
            self.velocities.append((vx, vy))
            if len(self.velocities) > self.buffer_len:
                self.velocities.pop(0)

        # acceleration
        if len(self.velocities) >= 2:
            vx2, vy2 = self.velocities[-1]
            vx1, vy1 = self.velocities[-2]
            ax = (vx2 - vx1) / self.dt
            ay = (vy2 - vy1) / self.dt
            self.accelerations.append((ax, ay))
            if len(self.accelerations) > self.buffer_len:
                self.accelerations.pop(0)

    def latest_position(self) -> Optional[Tuple[float, float]]:
        return self.positions[-1] if self.positions else None

    def latest_velocity(self) -> Optional[Tuple[float, float]]:
        return self.velocities[-1] if self.velocities else None

    def latest_acceleration(self) -> Optional[Tuple[float, float]]:
        return self.accelerations[-1] if self.accelerations else None

    def trajectory(self) -> List[Tuple[float, float]]:
        return list(self.positions)


# ---------- High-level orchestration ----------

@dataclass
class ProcessResult:
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
    velocity_at_frac: Tuple[Tuple[float, float, float], ...]  # (vx, vy, |v|)
    acceleration_at_frac: Tuple[Tuple[float, float, float], ...]  # (ax, ay, |a|)
    total_break: float  # lateral |pos_end - pos_start|
    end_lane_speed: float


class PostProcessor:
    pass

    def __init__(self,
                 out_size: Tuple[int, int] = (400, 800),
                 dt: float = 1.0 / 30.0,
                 buffer_len: int = 12,
                 interpolation_mode: Literal["none", "linear", "cubic"] = "none"):
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.dt = float(dt)
        self.buffer = BufferCalcs(buffer_len=buffer_len, dt=self.dt)
        self.interpolation_mode = interpolation_mode

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
        if not masks_by_index:
            return {}, np.eye(3, dtype=np.float64)

        # Compute H once from the first available lane mask (or provided correspondences)
        first_idx = sorted(masks_by_index.keys())[0]
        first_lane = masks_by_index[first_idx]["lane"]
        H = self._compute_H_once(first_lane, homography_src_dst)

        results: Dict[int, ProcessResult] = {}

        for idx in sorted(masks_by_index.keys()):
            ball_mask = masks_by_index[idx]["ball"]
            lane_mask = masks_by_index[idx]["lane"]

            warped_lane = Warp.warp_mask(lane_mask, H, self.out_size)
            warped_ball = Warp.warp_mask(ball_mask, H, self.out_size)

            # centroid in IMAGE -> map via H (faster than recomputing centroid in warped space)
            img_centroid = Warp.centroid_from_mask(ball_mask)
            bev_centroid = Warp.image_to_bev_point(img_centroid, H) if img_centroid else None

            # update temporal buffer and fetch kinematics
            self.buffer.add(bev_centroid)
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
            Time values (or frame indices)
        data : np.ndarray
            Data to interpolate (positions, velocities, etc.)
            
        Returns
        -------
        t_interp, data_interp : Tuple[np.ndarray, np.ndarray]
            Interpolated time and data values
        """
        if self.interpolation_mode == "none" or len(t) < 3:
            return t, data
        
        # Create denser time grid for interpolation
        t_interp = np.linspace(t[0], t[-1], len(t) * 3)
        
        if self.interpolation_mode == "linear":
            # Piecewise linear interpolation - NO extrapolation
            interp_func = interpolate.interp1d(
                t, data, kind='linear', axis=0,
                bounds_error=False, fill_value=(data[0], data[-1])  # Clamp to endpoints
            )
            data_interp = interp_func(t_interp)
        elif self.interpolation_mode == "cubic":
            # Cubic spline interpolation
            if len(t) < 4:
                # Fall back to linear if not enough points for cubic
                interp_func = interpolate.interp1d(
                    t, data, kind='linear', axis=0,
                    bounds_error=False, fill_value=(data[0], data[-1])
                )
                data_interp = interp_func(t_interp)
            else:
                # Use cubic spline with natural boundary conditions (no extrapolation)
                interp_func = interpolate.CubicSpline(
                    t, data, axis=0, bc_type='natural', extrapolate=False
                )
                data_interp = interp_func(t_interp)
                # CubicSpline with extrapolate=False returns NaN outside bounds,
                # so clip to valid range
                data_interp = np.where(
                    np.isnan(data_interp),
                    np.where(t_interp[:, None] < t[0], data[0], data[-1]),
                    data_interp
                )
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
        the other axis as lateral. This is consistent with how tests currently
        recover a 1D coordinate along the lane from BEV centroids.
        
        Interpolation can be applied based on self.interpolation_mode:
        - "none": Use raw trajectory with np.gradient
        - "linear": Piecewise linear interpolation
        - "cubic": Cubic spline interpolation
        """
        if not results_by_index:
            raise ValueError("results_by_index is empty")

        # Collect BEV centroids in temporal order, skipping missing ones.
        idxs = sorted(results_by_index.keys())
        traj: List[Tuple[float, float]] = []
        frame_times: List[float] = []
        for i in idxs:
            c = results_by_index[i].bev_centroid
            if c is not None:
                traj.append(c)
                # Convert frame index to seconds using dt to avoid mixing units
                frame_times.append(float(i) * self.dt)

        if len(traj) < 2:
            raise ValueError("Not enough valid centroids to compute metrics")

        bev = np.array(traj, dtype=np.float64)
        t_raw = np.array(frame_times, dtype=np.float64)
        
        # Apply interpolation if enabled
        t, bev_interp = self._apply_interpolation(t_raw, bev)

        # FAIL FAST: Check for any invalid values BEFORE they propagate
        if not np.all(np.isfinite(bev_interp)):
            raise ValueError(
                f"Non-finite values detected after interpolation. "
                f"NaN count: {np.sum(np.isnan(bev_interp))}, "
                f"Inf count: {np.sum(np.isinf(bev_interp))}"
            )
        
        if not np.all(np.isfinite(t)):
            raise ValueError("Non-finite time values detected")
        
        xs = bev_interp[:, 0]
        ys = bev_interp[:, 1]

        # Choose primary axis as the one with larger dynamic range.
        range_x = float(xs.max() - xs.min())
        range_y = float(ys.max() - ys.min())
        primary_axis = 0 if range_x >= range_y else 1

        s = xs if primary_axis == 0 else ys  # along-lane coordinate
        l = ys if primary_axis == 0 else xs  # lateral coordinate

        # Ensure s increases along the lane for simpler fraction logic.
        if s[-1] < s[0]:
            s = -s

        lane_len = float(s[-1] - s[0])
        if lane_len <= 0.0:
            raise ValueError("Non-positive lane length in BEV coordinates")

        # Compute uniform dt after interpolation
        # After interpolation, t should be uniformly spaced
        dt_uniform = float(np.mean(np.diff(t)))
        
        # Finite-difference velocities and accelerations in BEV space
        # Use uniform dt for cleaner gradients
        vs = np.gradient(s, dt_uniform)
        vl = np.gradient(l, dt_uniform)
        ax = np.gradient(vs, dt_uniform)
        ay = np.gradient(vl, dt_uniform)
        
        # FAIL FAST: Check derivatives for validity
        if not np.all(np.isfinite(vs)) or not np.all(np.isfinite(vl)):
            raise ValueError(
                f"Non-finite velocity values. "
                f"vs: NaN={np.sum(np.isnan(vs))}, Inf={np.sum(np.isinf(vs))}; "
                f"vl: NaN={np.sum(np.isnan(vl))}, Inf={np.sum(np.isinf(vl))}"
            )
        
        if not np.all(np.isfinite(ax)) or not np.all(np.isfinite(ay)):
            raise ValueError(
                f"Non-finite acceleration values. "
                f"ax: NaN={np.sum(np.isnan(ax))}, Inf={np.sum(np.isinf(ax))}; "
                f"ay: NaN={np.sum(np.isnan(ay))}, Inf={np.sum(np.isinf(ay))}"
            )

        frac_list = []
        frac_positions = []
        vel_at_frac = []
        acc_at_frac = []

        for f in fractions:
            f_clamped = float(max(0.0, min(1.0, f)))
            target_s = s[0] + f_clamped * lane_len
            idx_closest = int(np.argmin(np.abs(s - target_s)))

            vx_f = float(vs[idx_closest])
            vy_f = float(vl[idx_closest])
            ax_f = float(ax[idx_closest])
            ay_f = float(ay[idx_closest])

            speed_f = float(np.hypot(vx_f, vy_f))
            acc_mag_f = float(np.hypot(ax_f, ay_f))

            frac_list.append(f_clamped)
            frac_positions.append(float(s[idx_closest]))
            vel_at_frac.append((vx_f, vy_f, speed_f))
            acc_at_frac.append((ax_f, ay_f, acc_mag_f))

        # Total break: lateral deviation from start to end.
        total_break = float(abs(l[-1] - l[0]))
        # End-of-lane speed from last sample.
        end_speed = float(np.hypot(vs[-1], vl[-1]))

        return LaneMetrics(
            fractions=tuple(frac_list),
            frac_positions=tuple(frac_positions),
            velocity_at_frac=tuple(vel_at_frac),
            acceleration_at_frac=tuple(acc_at_frac),
            total_break=total_break,
            end_lane_speed=end_speed,
        )
