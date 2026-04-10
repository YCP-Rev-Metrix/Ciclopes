from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from core.InferenceEngine.LaneBallInference import LaneBallInference
from core.InferenceEngine.Sam3DBodyInference import Sam3DBodyInference
from core.LaneBalls.Extrapolation import append_departure_point, trim_raw_detections
from core.LaneBalls.Interprolation import interpolate_ball_positions
from core.LaneBalls.Kinematics import compute_kinematics_per_quarter
from core.LaneBalls.Postprocessing import run_lane_ball_postprocessing
from core.LaneBalls.Preprocessing import extract_frame_segmentation

logger = logging.getLogger("ciclopes.inference_engine")


class InferenceEngine:
    """
    Inference orchestrator.

    Manages the YOLO segmentation model (ball / lane / pins) and
    SAM 3D Body (skeleton estimation). Supports single-GPU (default) or
    multi-GPU mode where each model gets its own device.

    Call `forward()` to run YOLO segmentation on the first RGB frame.
    Call `forward_sam3d_body()` to run 3D skeleton estimation on a list of frames.
    """

    def __init__(self, multi_gpu: bool = False) -> None:
        self.initialized_at = datetime.now(timezone.utc)
        self.multi_gpu = multi_gpu

        if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.device_lane_ball = torch.device("cuda:0")
            self.device_sam3d = torch.device("cuda:1")
            logger.info(
                "Multi-GPU mode: LaneBall → %s, SAM3D Body → %s",
                self.device_lane_ball, self.device_sam3d,
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device_lane_ball = device
            self.device_sam3d = device
            if multi_gpu:
                logger.warning(
                    "MULTI_GPU=true but only %d GPU(s) detected — falling back to single device %s",
                    torch.cuda.device_count(), device,
                )
            else:
                logger.info("Single-GPU mode: both models on %s", device)

        # For backwards compat (status endpoint, etc.)
        self.device = self.device_lane_ball

        # ── Load active models ────────────────────────────────────────────────
        self.lane_ball = LaneBallInference(device=str(self.device_lane_ball))
        self.sam3d_body = Sam3DBodyInference(device=str(self.device_sam3d))

        # Thread pool for running sync model inference in async context.
        # Keep 2 workers so both models can run concurrently.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")

        logger.info("InferenceEngine ready")

    # ── Async forward pass ────────────────────────────────────────────────────

    async def forward(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """
        Run YOLO segmentation on the first frame of a list of RGB frames.

        Args:
            frames: List of numpy arrays in RGB format, each (H, W, 3).

        Returns:
            {
                "segmentation": first-frame segmentation masks grouped by class,
            }
        """
        if not frames:
            return {"segmentation": {}}

        loop = asyncio.get_running_loop()

        seg_future = loop.run_in_executor(
            self._executor, self._run_segmentation_first_frame, frames[0]
        )

        seg_result = await seg_future

        return {
            "segmentation": seg_result,
        }

    async def forward_lane_ball(
        self,
        frames_rgb: list[np.ndarray],
        fps: float,
        start_frame: int,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        Full lane-ball pipeline:
          1) YOLO seg on all frames (batched)
          2) first trapezoid homography search from start_frame
          3) ball contact-point projection to lane meters
          4) kinematics per quarter
        """
        if not frames_rgb:
            return {
                "positions": [],
                "kinematics": {"quarters": []},
                "is_trapezoid": False,
                "homography_frame": None,
                "health": {"error": "No frames provided"},
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_lane_ball_pipeline_sync,
            frames_rgb,
            float(fps),
            int(start_frame),
            int(batch_size),
        )

    async def forward_sam3d_body(
        self,
        frames_rgb: list[np.ndarray],
        batch_size: int = 4,
    ) -> list[list[dict[str, Any]]]:
        """
        Run SAM 3D Body skeleton estimation on every frame using batched GPU inference.

        Args:
            frames_rgb: list of (H, W, 3) uint8 RGB numpy arrays.
            batch_size: frames per GPU forward pass (set via SAM3D_BODY_BATCH_SIZE).

        Returns:
            List (one entry per frame) of lists (one entry per detected person)
            of dicts with keys: joint_id (int), x/y/z (float).
        """
        if not frames_rgb:
            return []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._run_sam3d_body_sync,
            frames_rgb,
            batch_size,
        )

    def _run_sam3d_body_sync(
        self, frames_rgb: list[np.ndarray], batch_size: int = 4
    ) -> list[list[dict[str, Any]]]:
        t0 = time.perf_counter()

        batch_results = self.sam3d_body.infer_batch(frames_rgb, batch_size=batch_size)

        all_frames: list[list[dict[str, Any]]] = []
        for frame_persons in batch_results:
            frame_joints: list[dict[str, Any]] = []
            for person in frame_persons:
                skeleton = Sam3DBodyInference.extract_skeleton(person)
                for joint in skeleton.joints:
                    frame_joints.append(
                        {
                            "joint_id": joint.joint_id,
                            "x": joint.x,
                            "y": joint.y,
                            "z": joint.z,
                        }
                    )
            all_frames.append(frame_joints)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "SAM3D Body: %d frames, batch_size=%d, %.1f ms total",
            len(frames_rgb), batch_size, elapsed_ms,
        )
        return all_frames

    # ── Internal: YOLO seg on the first frame only ────────────────────────────

    def _run_segmentation_first_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Run YOLO segmentation on a single frame and return structured masks.
        """
        raw_results = self.lane_ball.infer(frame)
        return LaneBallInference.extract_masks(raw_results)

    def _run_lane_ball_pipeline_sync(
        self, frames_rgb: list[np.ndarray], fps: float, start_frame: int,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        inference_t0 = time.perf_counter()
        # Chunk frames to respect batch_size and avoid VRAM spikes
        raw_results = []
        for i in range(0, len(frames_rgb), batch_size):
            chunk = frames_rgb[i : i + batch_size]
            raw_results.extend(self.lane_ball.infer_batch(chunk))
        inference_ms = (time.perf_counter() - inference_t0) * 1000.0

        segmentations_by_frame = {
            idx: extract_frame_segmentation(res) for idx, res in enumerate(raw_results)
        }

        total_ball_masks = sum(len(s.ball_masks) for s in segmentations_by_frame.values())
        total_lane_masks = sum(len(s.lane_masks) for s in segmentations_by_frame.values())
        logger.info(
            "Segmentation extraction: %d frames, %d total ball masks, %d total lane masks",
            len(segmentations_by_frame), total_ball_masks, total_lane_masks,
        )

        post_t0 = time.perf_counter()
        try:
            post = run_lane_ball_postprocessing(
                segmentations_by_frame=segmentations_by_frame,
                fps=fps,
                start_frame=start_frame,
                frames_bgr=[
                    np.ascontiguousarray(frame[:, :, ::-1]) for frame in frames_rgb
                ],
            )
        except Exception as exc:
            logger.exception("Postprocessing failed — returning empty results")
            post_ms = (time.perf_counter() - post_t0) * 1000.0
            return {
                "positions": [],
                "kinematics": {"quarters": []},
                "is_trapezoid": False,
                "homography_frame": None,
                "homography_src_corners": [],
                "homography_dst_corners_m": [],
                "homography_matrix": [],
                "health": {
                    "inference_ms": round(inference_ms, 2),
                    "postprocess_ms": round(post_ms, 2),
                    "frames_scanned_for_h": 0,
                    "frames_with_lane": 0,
                    "frames_with_ball": 0,
                    "lane_polygon_count_at_h": 0,
                    "homography_determinant": 0.0,
                    "homography_condition_number": 0.0,
                    "mean_lane_coverage_ratio": 0.0,
                },
            }
        post_ms = (time.perf_counter() - post_t0) * 1000.0

        # ── Trim → Interpolate → Departure → Kinematics ─────────────────
        raw_positions = post.ball_positions.ball_positions
        clean_positions = trim_raw_detections(raw_positions)
        smooth_positions = interpolate_ball_positions(clean_positions, fps)
        final_positions = append_departure_point(smooth_positions, fps)
        kin = compute_kinematics_per_quarter(final_positions)

        return {
            "positions": [
                {
                    "frame_index": p.frame_index,
                    "timestamp_s": p.timestamp_s,
                    "x_m": p.x_m,
                    "y_m": p.y_m,
                }
                for p in final_positions
            ],
            "kinematics": {
                "quarters": [
                    {
                        "quarter": q.quarter,
                        "start_m": q.start_m,
                        "end_m": q.end_m,
                        "mean_speed_mps": q.mean_speed_mps,
                        "mean_acceleration_mps2": q.mean_acceleration_mps2,
                        "sample_count": q.sample_count,
                    }
                    for q in kin.quarters
                ]
            },
            "is_trapezoid": bool(post.homography_selection.is_trapezoid),
            "homography_frame": int(post.homography_selection.frame_index),
            "homography_src_corners": post.homography_selection.src_corners.astype(float).tolist(),
            "homography_dst_corners_m": post.homography_selection.dst_corners.astype(float).tolist(),
            "homography_matrix": post.homography_selection.homography.astype(float).tolist(),
            "health": {
                "frames_scanned_for_h": post.health.frames_scanned_for_h,
                "frames_with_lane": post.health.frames_with_lane,
                "frames_with_ball": post.health.frames_with_ball,
                "lane_polygon_count_at_h": post.health.lane_polygon_count_at_h,
                "homography_determinant": post.health.homography_determinant,
                "homography_condition_number": post.health.homography_condition_number,
                "mean_lane_coverage_ratio": post.health.mean_lane_coverage_ratio,
                "inference_ms": round(inference_ms, 2),
                "postprocess_ms": round(post_ms, 2),
            },
        }

    # ── Status / health ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return engine health info including device and VRAM usage."""
        info: dict[str, Any] = {
            "multi_gpu": self.multi_gpu,
            "device_lane_ball": str(self.device_lane_ball),
            "device_sam3d": str(self.device_sam3d),
            "initialized_at": self.initialized_at.isoformat(),
            "cuda_available": torch.cuda.is_available(),
        }

        # Append VRAM stats when running on CUDA
        if torch.cuda.is_available():
            info["vram_lane_ball_allocated_mb"] = round(
                torch.cuda.memory_allocated(self.device_lane_ball) / 1024 / 1024, 1
            )
            info["vram_lane_ball_reserved_mb"] = round(
                torch.cuda.memory_reserved(self.device_lane_ball) / 1024 / 1024, 1
            )
            if self.device_lane_ball != self.device_sam3d:
                info["vram_sam3d_allocated_mb"] = round(
                    torch.cuda.memory_allocated(self.device_sam3d) / 1024 / 1024, 1
                )
                info["vram_sam3d_reserved_mb"] = round(
                    torch.cuda.memory_reserved(self.device_sam3d) / 1024 / 1024, 1
                )

        return info
