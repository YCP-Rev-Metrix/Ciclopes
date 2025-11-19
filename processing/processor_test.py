from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from processing.preprocessing.preprocessor import YOLOSegPreprocessor, InferenceConfig
from processing.postprocessing.postprocessor import PostProcessor
from processing.processor import Processor


class _TrajectoryDummyYOLO:
    """
    Dummy YOLO model that yields a ball following a simple curved trajectory
    across frames so that lane metrics (velocity, break, end speed) are
    well-defined without requiring a real network.
    """

    def __init__(self) -> None:
        self.names = {0: "lane", 1: "ball"}

    def predict(self, source: str, imgsz: int, device: str, verbose: bool):
        h, w = 40, 80
        lane = np.zeros((h, w), dtype=np.float32)
        lane[5:35, 10:70] = 1.0  # long rectangular lane

        stem = Path(source).stem  # "_0000" style
        try:
            idx = int(stem.strip("_"))
        except ValueError:
            idx = 0

        # Along-lane coordinate increases with idx, lateral coordinate curves.
        t = float(idx)
        y_center = 8 + 4 * t  # down-lane (primary axis)
        x_center = 30 + 2 * t  # lateral drift to create "break"

        ball = np.zeros((h, w), dtype=np.float32)
        y0 = max(0, int(y_center - 2))
        y1 = min(h, int(y_center + 2))
        x0 = max(0, int(x_center - 2))
        x1 = min(w, int(x_center + 2))
        ball[y0:y1, x0:x1] = 1.0

        mask_stack = np.stack([lane, ball], axis=0)

        class _Masks:
            def __init__(self, data):
                self.data = data

        class _Boxes:
            def __init__(self, cls_ids):
                self.cls = cls_ids

        class _Result:
            def __init__(self, masks, boxes, orig_shape):
                self.masks = masks
                self.boxes = boxes
                self.orig_shape = orig_shape

        masks = _Masks(mask_stack)
        boxes = _Boxes(np.array([0.0, 1.0], dtype=np.float32))
        return [_Result(masks, boxes, (h, w))]


def _make_dummy_frames(img_dir: Path, num_frames: int) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_frames):
        canvas = np.zeros((40, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"_{idx:04d}.jpg"), canvas)


def test_full_processor_wrapper_computes_lane_metrics(tmp_path):
    images_dir = tmp_path / "episode_images"
    num_frames = 6
    _make_dummy_frames(images_dir, num_frames)

    dummy_model = _TrajectoryDummyYOLO()
    infer_cfg = InferenceConfig(weights_path=tmp_path / "dummy.pt", imgsz=80, device="cpu")
    pre = YOLOSegPreprocessor(config=infer_cfg, model=dummy_model)
    post = PostProcessor(out_size=(400, 800), dt=1.0, buffer_len=64)

    proc = Processor(preprocessor=pre, postprocessor=post)
    episode = proc.run_episode_from_indices(images_dir, range(num_frames))

    # Basic structural checks
    assert episode.homography.shape == (3, 3)
    assert len(episode.results_by_index) == num_frames

    metrics = episode.lane_metrics
    # Fractions default to 4 positions along the lane
    assert len(metrics.fractions) == 4
    assert len(metrics.velocity_at_frac) == 4
    assert len(metrics.acceleration_at_frac) == 4

    # With the synthetic trajectory, we should have non-zero end speed
    assert metrics.end_lane_speed > 0.0
    # And some lateral break over the run
    assert metrics.total_break > 0.0


