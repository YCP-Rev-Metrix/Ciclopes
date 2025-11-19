from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from processing.preprocessing.preprocessor import YOLOSegPreprocessor, InferenceConfig


class _DummyMasks:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _DummyBoxes:
    def __init__(self, cls_ids: np.ndarray) -> None:
        self.cls = cls_ids


class _DummyResult:
    def __init__(self, masks: _DummyMasks, boxes: _DummyBoxes, orig_shape) -> None:
        self.masks = masks
        self.boxes = boxes
        self.orig_shape = orig_shape


class _DummyYOLO:
    """
    Minimal stand‑in for ultralytics.YOLO that records calls to `.predict()`
    and returns synthetic segmentation masks for lane (class 0) and ball (1).
    """

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.names = {0: "lane", 1: "ball"}

    def predict(self, source: str, imgsz: int, device: str, verbose: bool):
        self.calls.append(source)
        # Small synthetic canvas
        h, w = 16, 32

        # Lane mask: wide horizontal stripe
        lane = np.zeros((h, w), dtype=np.float32)
        lane[8:12, 2:30] = 1.0

        # Ball mask: small blob that depends on frame index (to mimic motion)
        ball = np.zeros((h, w), dtype=np.float32)
        stem = Path(source).stem  # e.g., "_0030"
        try:
            idx = int(stem.strip("_"))
        except ValueError:
            idx = 0
        cx = 4 + (idx % 8)
        ball[4:8, cx : cx + 4] = 1.0

        mask_stack = np.stack([lane, ball], axis=0)  # [2, h, w]
        masks = _DummyMasks(mask_stack)
        boxes = _DummyBoxes(np.array([0.0, 1.0], dtype=np.float32))
        return [_DummyResult(masks, boxes, (h, w))]


def test_yoloseg_preprocessor_runs_predict_and_builds_masks(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create two empty dummy RGB frames
    for idx in (0, 1):
        img = np.zeros((16, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"_{idx:04d}.jpg"), img)

    dummy_model = _DummyYOLO()
    cfg = InferenceConfig(weights_path=tmp_path / "dummy.pt", imgsz=64, device="cpu")
    pre = YOLOSegPreprocessor(config=cfg, model=dummy_model)

    masks_by_index, (width, height) = pre.run_on_indices(img_dir, [0, 1])

    # One call per frame, using the expected paths
    assert len(dummy_model.calls) == 2
    assert all(Path(c).parent == img_dir for c in dummy_model.calls)

    # Masks should exist for both frames and both classes, at original size
    assert (width, height) == (32, 16)
    for idx in (0, 1):
        assert idx in masks_by_index
        lane = masks_by_index[idx]["lane"]
        ball = masks_by_index[idx]["ball"]
        assert lane.shape == (16, 32)
        assert ball.shape == (16, 32)
        # Lane and ball masks are non‑empty
        assert lane.max() in (0, 255)
        assert ball.max() in (0, 255)


