from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


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


class PostProcessor:
    pass

    def __init__(self,
                 out_size: Tuple[int, int] = (400, 800),
                 dt: float = 1.0 / 30.0,
                 buffer_len: int = 12):
        self.out_size = (int(out_size[0]), int(out_size[1]))
        self.buffer = BufferCalcs(buffer_len=buffer_len, dt=dt)

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
