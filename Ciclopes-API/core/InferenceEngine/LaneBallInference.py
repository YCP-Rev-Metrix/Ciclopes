from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger("ciclopes.lane_ball_inference")

# ── Weight path (relative to Ciclopes-API root) ──────────────────────────────
_API_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WEIGHT_PATH = str(_API_ROOT / "core" / "weights" / "best_v2_26n.pt")

# ── Class mapping matching the YOLO seg training config ───────────────────────
CLASS_NAMES = {0: "ball", 1: "lane", 2: "pins"}


class LaneBallInference:
    """
    YOLO v26n segmentation model for bowling lane, ball, and pin detection.

    Loads from a local .pt checkpoint and runs instance-segmentation inference.
    The model was trained via pipeline_v2 with yolo26n-seg on 3 classes:
        0 → ball
        1 → lane
        2 → pins
    """

    def __init__(self, device: str = "cuda") -> None:
        logger.info("Loading YOLO seg model from %s", _DEFAULT_WEIGHT_PATH)

        self.model = YOLO(_DEFAULT_WEIGHT_PATH)
        self.model.to(device)
        self.device = device

        logger.info("YOLO seg model loaded on device=%s", device)

    # ── Raw inference ─────────────────────────────────────────────────────────

    def infer(self, image: np.ndarray | str) -> list:
        """
        Run segmentation on a single image.

        Args:
            image: RGB numpy array (H, W, 3) or file path string.

        Returns:
            List of ultralytics Results objects.
        """
        return self.model.predict(image, verbose=False)

    # ── Structured mask extraction for JSON serialization ─────────────────────

    @staticmethod
    def extract_masks(results: list) -> dict[str, list[dict[str, Any]]]:
        """
        Convert YOLO Results into a JSON-friendly dict grouped by class.

        Returns:
            {
                "ball":  [{"confidence": 0.95, "bbox": [x1,y1,x2,y2], "mask_shape": [H,W]}, ...],
                "lane":  [...],
                "pins":  [...]
            }

        Each key may have zero or more detections.
        """
        output: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CLASS_NAMES.values()
        }

        if not results:
            return output

        result = results[0]  # single-image prediction

        # No detections at all
        if result.boxes is None or len(result.boxes) == 0:
            return output

        boxes = result.boxes
        masks = result.masks  # may be None if model found boxes but no masks

        for idx in range(len(boxes)):
            cls_id = int(boxes.cls[idx].item())
            cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

            detection: dict[str, Any] = {
                "confidence": round(float(boxes.conf[idx].item()), 4),
                "bbox": [round(float(v), 2) for v in boxes.xyxy[idx].tolist()],
            }

            # Attach mask shape if masks are available
            if masks is not None and idx < len(masks.data):
                mask_tensor = masks.data[idx]  # (H, W) binary mask
                detection["mask_shape"] = list(mask_tensor.shape)

            output.setdefault(cls_name, []).append(detection)

        return output
