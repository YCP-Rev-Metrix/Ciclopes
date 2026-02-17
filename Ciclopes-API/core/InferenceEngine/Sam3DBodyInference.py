from __future__ import annotations

import importlib as _il
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf
_body_models = _il.import_module("core.4DBody.models")
JointObj = _body_models.JointObj
Skeleton = _body_models.Skeleton

logger = logging.getLogger("ciclopes.sam3d_body_inference")

# ── HuggingFace repo for the DinoV3-backed SAM 3D Body model ─────────────────
HF_REPO_ID = os.getenv("SAM3D_BODY_HF_REPO_ID", "facebook/sam-3d-body-dinov3")

# ── Optional local weight directory (populated by HF cache or manual copy) ────
_API_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_WEIGHTS_DIR = _API_ROOT / "core" / "weights" / "sam3d_body"


class Sam3DBodyInference:
    """
    SAM 3D Body wrapper for per-frame skeleton estimation.

    On init, tries to load weights from the local weights directory.
    If not found, downloads from HuggingFace and caches for future runs.

    The model outputs 70 MHR keypoints per detected person, each with
    3D coordinates (x, y, z). These are converted into our Skeleton /
    JointObj dataclass format for downstream use.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

        model, model_cfg = self._load_model(device)
        self.model = model
        self.model_cfg = model_cfg

        # Build estimator — no human detector, no SAM2 segmentor, no FOV estimator.
        # Without a detector, it defaults to using the full image as the bounding box.
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=self.model,
            model_cfg=self.model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        logger.info("SAM 3D Body estimator ready on device=%s", device)

    # ── Model loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(device: str):
        """
        Try local weights first, fall back to HuggingFace download.

        Local layout expected:
            core/weights/sam3d_body/
                model.ckpt
                model_config.yaml
                assets/
                    mhr_model.pt
        """
        ckpt_path = _LOCAL_WEIGHTS_DIR / "model.ckpt"
        mhr_path = _LOCAL_WEIGHTS_DIR / "assets" / "mhr_model.pt"

        if ckpt_path.exists() and mhr_path.exists():
            logger.info("Loading SAM 3D Body from local weights: %s", _LOCAL_WEIGHTS_DIR)
            return load_sam_3d_body(
                checkpoint_path=str(ckpt_path),
                device=device,
                mhr_path=str(mhr_path),
            )

        # Download from HuggingFace (caches in ~/.cache/huggingface/)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        logger.info(
            "Local weights not found at %s — downloading from HuggingFace: %s",
            _LOCAL_WEIGHTS_DIR,
            HF_REPO_ID,
        )
        try:
            return load_sam_3d_body_hf(repo_id=HF_REPO_ID, token=hf_token, device=device)
        except Exception as exc:
            # Avoid hard dependency on huggingface_hub internals when local weights are used.
            if exc.__class__.__name__ == "GatedRepoError":
                raise RuntimeError(
                    "Cannot access gated Hugging Face repo "
                    f"'{HF_REPO_ID}'. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
                    "to a token with approved access, or place local weights at "
                    f"'{_LOCAL_WEIGHTS_DIR}'."
                ) from exc
            raise

    # ── Per-frame inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def infer_frame(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """
        Run 3D body estimation on a single RGB frame.

        Args:
            image_rgb: numpy array in RGB format, shape (H, W, 3).

        Returns:
            List of result dicts (one per detected person). Each contains:
                pred_keypoints_3d  — (70, 3) float array
                pred_keypoints_2d  — (70, 2) float array
                pred_cam_t         — (3,) camera translation
                bbox               — (4,) detection box
                ... and more (see sam_3d_body_estimator.py)
        """
        return self.estimator.process_one_image(
            img=image_rgb,
            inference_type="body",
        )

    # ── Convert raw output to our Skeleton model ─────────────────────────────

    @staticmethod
    def extract_skeleton(person_result: dict[str, Any]) -> Skeleton:
        """
        Map SAM 3D Body output for one person into our Skeleton dataclass.

        Uses pred_keypoints_3d (70, 3) — one JointObj per MHR keypoint.
        Joint IDs match the mhr70 ordering (see core/sam_3d_body/metadata/mhr70.py).
        """
        kpts_3d = person_result["pred_keypoints_3d"]  # (70, 3)

        joints = []
        for joint_id in range(kpts_3d.shape[0]):
            joints.append(
                JointObj(
                    x=float(kpts_3d[joint_id, 0]),
                    y=float(kpts_3d[joint_id, 1]),
                    z=float(kpts_3d[joint_id, 2]),
                    joint_id=joint_id,
                )
            )

        return Skeleton(joints=joints)

    # ── Convenience: full frame → Skeleton list ──────────────────────────────

    def frame_to_skeletons(self, image_rgb: np.ndarray) -> list[Skeleton]:
        """
        End-to-end: RGB frame → list of Skeleton objects (one per person).
        """
        raw_results = self.infer_frame(image_rgb)
        return [self.extract_skeleton(r) for r in raw_results]
