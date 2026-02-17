from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from core.InferenceEngine.LaneBallInference import LaneBallInference

logger = logging.getLogger("ciclopes.inference_engine")


class InferenceEngine:
    """
    Inference orchestrator.

    Manages the YOLO segmentation model (ball / lane / pins) on a single GPU.
    SAM 3D Body is temporarily disabled while gated checkpoint access is pending.

    Call `forward()` to run YOLO segmentation on the first RGB frame.
    """

    def __init__(self) -> None:
        self.initialized_at = datetime.now(timezone.utc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Initializing InferenceEngine on device=%s", self.device)

        # ── Load active models ────────────────────────────────────────────────
        self.lane_ball = LaneBallInference(device=str(self.device))
        # TEMPORARILY DISABLED:
        # self.sam3d_body = Sam3DBodyInference(device=str(self.device))

        # Thread pool for running sync model inference in async context.
        # Keep 2 workers for parity with earlier dual-model setup.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")

        logger.info("InferenceEngine ready — LaneBall model loaded on %s", self.device)

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

    # ── Internal: YOLO seg on the first frame only ────────────────────────────

    def _run_segmentation_first_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Run YOLO segmentation on a single frame and return structured masks.
        """
        raw_results = self.lane_ball.infer(frame)
        return LaneBallInference.extract_masks(raw_results)

    # ── Status / health ───────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return engine health info including device and VRAM usage."""
        info: dict[str, Any] = {
            "device": str(self.device),
            "initialized_at": self.initialized_at.isoformat(),
            "cuda_available": torch.cuda.is_available(),
        }

        # Append VRAM stats when running on CUDA
        if torch.cuda.is_available():
            info["vram_allocated_mb"] = round(
                torch.cuda.memory_allocated(self.device) / 1024 / 1024, 1
            )
            info["vram_reserved_mb"] = round(
                torch.cuda.memory_reserved(self.device) / 1024 / 1024, 1
            )

        return info
