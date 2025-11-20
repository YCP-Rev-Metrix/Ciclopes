from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Iterable

import cv2
import numpy as np


@dataclass
class InferenceConfig:
    """Configuration for YOLO segmentation inference."""

    weights_path: Optional[Path] = None
    imgsz: int = 2048
    device: Optional[str] = None  # "cuda", "cpu", "0", etc.


class YOLOSegPreprocessor:
    """
    Lightweight preprocessor that runs YOLOv11-seg `.predict()` on a sequence
    of RGB frames and produces per-frame masks for the postprocessor.

    The interface is intentionally similar to the helpers used in
    `processing/postprocessing/postprocessor_test.py`, but instead of reading
    pre-generated YOLO label polygons it executes the model directly.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        *,
        model=None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            Inference configuration. If None, sensible defaults are used:
            - weights: `yolo11s-seg.pt` in the project root (fine-tuned weights
              can be supplied here later by the caller)
            - imgsz: 2048
            - device: auto-detected ("cuda" if available, else "cpu")
        model:
            Optional pre-constructed ultralytics `YOLO` model. If provided,
            this is used directly (useful for tests / dependency injection).
            If None, the model is created from `config.weights_path` using
            `YOLO(weights_path)` – exactly like the trainer.
        """
        self.project_root = Path(__file__).resolve().parents[2]

        if config is None:
            default_weights = self.project_root / "best.pt"
            config = InferenceConfig(weights_path=default_weights)

        if config.weights_path is None:
            config.weights_path = self.project_root / "best.pt"

        self.config = config

        # Device auto-detection if not provided
        if self.config.device is None:
            try:
                import torch

                self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                # Fallback if torch is not available for any reason
                self.config.device = "cpu"

        # YOLO model – this MUST use `.predict()` exactly as in the trainer.
        if model is not None:
            self._model = model
        else:
            from ultralytics import YOLO

            self._model = YOLO(str(self.config.weights_path))

        # Class-name mapping from training (0: lane, 1: ball)
        # We keep this flexible in case a different YAML is used.
        self.names = getattr(self._model, "names", {0: "lane", 1: "ball"})
        self._lane_cls_id = self._resolve_class_id("lane")
        self._ball_cls_id = self._resolve_class_id("ball")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_on_indices(
        self,
        images_dir: Path,
        frame_indices: Iterable[int],
    ) -> Tuple[Dict[int, Dict[str, np.ndarray]], Tuple[int, int]]:
        """
        Run YOLOv11-seg `.predict()` on each frame index in an episode and
        return a masks dictionary compatible with `PostProcessor.process_run`.

        Parameters
        ----------
        images_dir:
            Directory containing RGB frames named like `_{idx:04d}.jpg`.
        frame_indices:
            Iterable of integer frame indices (e.g. range(start_idx, end_idx+1)).

        Returns
        -------
        masks_by_index, (width, height)
            `masks_by_index[idx] = {"ball": mask, "lane": mask}` with masks as
            `uint8` arrays in the original image size. Width/height are taken
            from the first successfully processed frame.
        """
        masks_by_index: Dict[int, Dict[str, np.ndarray]] = {}
        width: Optional[int] = None
        height: Optional[int] = None

        for idx in frame_indices:
            img_path = images_dir / f"_{idx:04d}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            lane_mask, ball_mask = self._infer_single(img_path)

            if width is None or height is None:
                height, width = lane_mask.shape[:2]

            masks_by_index[idx] = {"ball": ball_mask, "lane": lane_mask}

        if width is None or height is None:
            raise RuntimeError("No frames were processed; check frame_indices.")

        return masks_by_index, (width, height)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_class_id(self, target_name: str) -> int:
        """
        Resolve a semantic class name (e.g., 'lane', 'ball') to a YOLO class id.
        Falls back to the conventional ids if names are unavailable.
        """
        norm_target = target_name.strip().lower()

        # Fast path: names is a dict[int, str]
        if isinstance(self.names, dict):
            for cls_id, name in self.names.items():
                if self._normalize_class_name(name) == norm_target:
                    return int(cls_id)

        # Default mapping as used in the repo_to_yolo_seg pipeline
        return 0 if norm_target == "lane" else 1

    @staticmethod
    def _normalize_class_name(raw: str) -> str:
        """Mirror `normalize_class_name` from `repo_to_yolo_seg.py`."""
        if raw is None:
            return ""
        name = str(raw).split(",")[0].strip().lower()
        if name == "sphere":
            name = "ball"
        return name

    def _infer_single(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run YOLOv11-seg `.predict()` on a single image and extract lane/ball
        binary masks in the original image size.
        """
        # This is intentionally `.predict()` to mirror the trainer's usage.
        results = self._model.predict(
            source=str(img_path),
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )

        if not results:
            raise RuntimeError(f"YOLO predict returned no results for {img_path}")

        r = results[0]
        orig_h, orig_w = getattr(r, "orig_shape", (None, None))
        if orig_h is None or orig_w is None:
            # Fallback: read image size directly
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image at {img_path}")
            orig_h, orig_w = img.shape[:2]

        lane_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        ball_mask = np.zeros_like(lane_mask)

        masks = getattr(r, "masks", None)
        boxes = getattr(r, "boxes", None)
        if masks is None or boxes is None:
            # No detections – return empty masks
            return lane_mask, ball_mask

        mask_data = getattr(masks, "data", None)
        cls_tensor = getattr(boxes, "cls", None)

        if mask_data is None or cls_tensor is None:
            return lane_mask, ball_mask

        try:
            import torch  # noqa: F401

            mask_arr = mask_data.clone().detach().cpu().numpy()
            cls_ids = cls_tensor.clone().detach().cpu().numpy().astype(int).tolist()
        except Exception:
            # Very defensive: fall back to np.array in case tensors behave differently
            mask_arr = np.array(mask_data)
            cls_ids = np.array(cls_tensor).astype(int).tolist()

        if mask_arr.ndim != 3 or len(cls_ids) != mask_arr.shape[0]:
            # Shape mismatch; treat as no detections.
            return lane_mask, ball_mask

        n_det, mh, mw = mask_arr.shape

        # For each detection, upsample mask to original size and OR into the
        # appropriate class mask.
        for det_idx in range(n_det):
            cls_id = int(cls_ids[det_idx])
            raw_mask = (mask_arr[det_idx] > 0.5).astype(np.uint8) * 255
            up_mask = cv2.resize(
                raw_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            if cls_id == self._lane_cls_id:
                lane_mask = np.maximum(lane_mask, up_mask)
            elif cls_id == self._ball_cls_id:
                ball_mask = np.maximum(ball_mask, up_mask)

        return lane_mask, ball_mask

