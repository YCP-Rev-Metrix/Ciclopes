from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np


# Matches pipeline_v2 yolo26n-seg class semantics.
YOLO_CLASS_NAME_BY_ID: Dict[int, str] = {0: "ball", 1: "lane", 2: "pins"}


@dataclass(frozen=True)
class InferencePreprocessConfig:
    imgsz: int = 1024
    conf: float = 0.10
    iou: float = 0.50


@dataclass(frozen=True)
class FrameSegmentation:
    ball_masks: List[np.ndarray]
    lane_masks: List[np.ndarray]
    pins_masks: List[np.ndarray]
    frame_shape: tuple[int, int]


def to_model_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR frame to RGB for model inference."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H,W,3), got {frame_bgr.shape}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def normalize_model_names(raw_names: Mapping[Any, Any] | None) -> Dict[int, str]:
    if not isinstance(raw_names, Mapping):
        return dict(YOLO_CLASS_NAME_BY_ID)

    out: Dict[int, str] = {}
    for key, value in raw_names.items():
        try:
            cls_id = int(key)
        except Exception:
            continue
        name = str(value).strip().lower()
        out[cls_id] = name
    return out or dict(YOLO_CLASS_NAME_BY_ID)


def extract_frame_segmentation(result: Any) -> FrameSegmentation:
    """
    Convert one Ultralytics Results object to binary masks by class.
    """
    orig_h, orig_w = getattr(result, "orig_shape", (0, 0))
    if not orig_h or not orig_w:
        raise ValueError("Missing valid orig_shape on YOLO result")

    ball_masks: List[np.ndarray] = []
    lane_masks: List[np.ndarray] = []
    pins_masks: List[np.ndarray] = []

    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None or masks is None or len(boxes) == 0:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

    cls_tensor = getattr(boxes, "cls", None)
    data_tensor = getattr(masks, "data", None)
    if cls_tensor is None or data_tensor is None:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

    mask_arr = data_tensor.detach().cpu().numpy()
    cls_ids = cls_tensor.detach().cpu().numpy().astype(int)

    if mask_arr.ndim != 3:
        return FrameSegmentation(
            ball_masks=ball_masks,
            lane_masks=lane_masks,
            pins_masks=pins_masks,
            frame_shape=(int(orig_h), int(orig_w)),
        )

    for i in range(min(mask_arr.shape[0], cls_ids.shape[0])):
        cls_id = int(cls_ids[i])
        mask = (mask_arr[i] > 0.5).astype(np.uint8)
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if cls_id == 0:
            ball_masks.append(mask)
        elif cls_id == 1:
            lane_masks.append(mask)
        elif cls_id == 2:
            pins_masks.append(mask)

    return FrameSegmentation(
        ball_masks=ball_masks,
        lane_masks=lane_masks,
        pins_masks=pins_masks,
        frame_shape=(int(orig_h), int(orig_w)),
    )
