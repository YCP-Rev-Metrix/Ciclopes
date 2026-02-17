from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from core.LaneBalls.Preprocessing import InferencePreprocessConfig, normalize_model_names

logger = logging.getLogger("ciclopes.lane_ball_inference")

# Weight path relative to Ciclopes-API root.
_API_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHT_PATH = str(_API_ROOT / "core" / "weights" / "best_v2_26n.pt")

# Class mapping matching yolo26n-seg training.
CLASS_NAMES = {0: "ball", 1: "lane", 2: "pins"}


class LaneBallInference:
    """
    YOLO segmentation model for ball, lane, and pins.
    """

    def __init__(self, device: str = "cuda") -> None:
        logger.info("Loading YOLO seg model from %s", _DEFAULT_WEIGHT_PATH)

        self.model = YOLO(_DEFAULT_WEIGHT_PATH)
        self.model.to(device)
        self.device = device
        self.preprocess_cfg = InferencePreprocessConfig()
        self.model_names = normalize_model_names(getattr(self.model, "names", None))

        logger.info("YOLO seg model loaded on device=%s", device)

    def infer(self, image: np.ndarray | str) -> list:
        """
        Run segmentation on a single image.
        """
        return self.model.predict(
            image,
            verbose=False,
            imgsz=self.preprocess_cfg.imgsz,
            conf=self.preprocess_cfg.conf,
            iou=self.preprocess_cfg.iou,
            device=self.device,
            retina_masks=True,
        )

    def infer_batch(self, images: list[np.ndarray]) -> list:
        """
        Run segmentation on a batch of RGB frames.
        """
        if not images:
            return []
        return self.model.predict(
            source=images,
            verbose=False,
            imgsz=self.preprocess_cfg.imgsz,
            conf=self.preprocess_cfg.conf,
            iou=self.preprocess_cfg.iou,
            device=self.device,
            retina_masks=True,
        )

    @staticmethod
    def extract_masks(results: list) -> dict[str, list[dict[str, Any]]]:
        """
        Convert YOLO Results into a JSON-friendly dict grouped by class.
        """
        output: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CLASS_NAMES.values()
        }

        if not results:
            return output

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return output

        boxes = result.boxes
        masks = result.masks

        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            detection: dict[str, Any] = {
                "confidence": round(float(boxes.conf[idx].item()), 4),
                "bbox": [round(float(v), 2) for v in boxes.xyxy[idx].tolist()],
            }

            if masks is not None and idx < len(masks.data):
                detection["mask_shape"] = list(masks.data[idx].shape)

            output.setdefault(cls_name, []).append(detection)

        return output
