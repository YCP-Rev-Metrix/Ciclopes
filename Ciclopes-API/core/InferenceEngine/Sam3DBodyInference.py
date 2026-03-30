from __future__ import annotations

import importlib as _il
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf
from core.sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from core.sam_3d_body.data.utils.prepare_batch import prepare_batch
from core.sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor

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

    # ── Batched inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def infer_batch(self, frames_rgb: list[np.ndarray], batch_size: int = 4) -> list[list[dict[str, Any]]]:
        """
        Run 3D body estimation on multiple frames in GPU batches.

        Each frame is treated as containing 1 person (full-image bbox, no detector).
        Frames are stacked along the batch dimension and forwarded through the
        backbone + decoder in a single pass per chunk.

        Args:
            frames_rgb: list of (H, W, 3) uint8 RGB numpy arrays.
            batch_size: number of frames per GPU forward pass.

        Returns:
            List (one per input frame) of result dicts.
        """
        if not frames_rgb:
            return []

        transform = self.estimator.transform
        all_results: list[list[dict[str, Any]]] = []

        for chunk_start in range(0, len(frames_rgb), batch_size):
            chunk = frames_rgb[chunk_start : chunk_start + batch_size]
            batches = []
            frame_sizes = []

            for frame in chunk:
                h, w = frame.shape[:2]
                frame_sizes.append((h, w))
                bbox = np.array([0, 0, w, h]).reshape(1, 4)
                single = prepare_batch(frame, transform, bbox)
                batches.append(single)

            # Stack along batch dimension:
            # each single has img shape [1, 1, C, H_crop, W_crop]
            # stacking → [N, 1, C, H_crop, W_crop]
            stacked = {}
            for key in ["img", "img_size", "ori_img_size", "bbox_center", "bbox_scale",
                        "bbox", "affine_trans", "mask", "mask_score"]:
                if key in batches[0]:
                    stacked[key] = torch.cat([b[key] for b in batches], dim=0)
            stacked["person_valid"] = torch.ones((len(chunk), 1))

            # Camera intrinsics — use first frame's default (all same resolution in a video)
            h0, w0 = frame_sizes[0]
            focal = (h0**2 + w0**2) ** 0.5
            n = len(chunk)
            cam_int = torch.tensor([[focal, 0, w0 / 2.0],
                                    [0, focal, h0 / 2.0],
                                    [0, 0, 1]]).to(stacked["img"])
            # Shape [N, 3, 3] — model adds person dim internally
            stacked["cam_int"] = cam_int.unsqueeze(0).expand(n, -1, -1).contiguous()

            # img_ori is only needed for "full" inference (hand re-crop), not "body"
            stacked["img_ori"] = batches[0]["img_ori"]

            stacked = recursive_to(stacked, self.device)
            self.model._initialize_batch(stacked)

            pose_output = self.model.forward_step(stacked, decoder_type="body")

            out = pose_output["mhr"]
            out = recursive_to(out, "cpu")
            out = recursive_to(out, "numpy")

            for idx in range(len(chunk)):
                all_results.append([{
                    "bbox": stacked["bbox"][idx, 0].cpu().numpy(),
                    "focal_length": out["focal_length"][idx],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][idx],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][idx],
                    "pred_cam_t": out["pred_cam_t"][idx],
                }])

            torch.cuda.empty_cache()

        return all_results

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
