from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from processing.preprocessing.preprocessor import YOLOSegPreprocessor, InferenceConfig
from processing.postprocessing.postprocessor import PostProcessor, ProcessResult, LaneMetrics


@dataclass
class EpisodeRunResult:
    """Container for a full pre+post-processed episode."""

    results_by_index: Dict[int, ProcessResult]
    homography: np.ndarray
    lane_metrics: LaneMetrics
    raw_masks_by_index: Dict[int, Dict[str, np.ndarray]]


class Processor:
    """
    Thin wrapper that ties together the YOLO-based preprocessor and the BEV
    postprocessor for end-to-end testing on a single episode.
    """

    def __init__(
        self,
        preprocessor: YOLOSegPreprocessor | None = None,
        postprocessor: PostProcessor | None = None,
        *,
        infer_config: InferenceConfig | None = None,
    ) -> None:
        # Allow dependency injection for tests; otherwise construct defaults.
        self.pre = preprocessor or YOLOSegPreprocessor(config=infer_config)
        self.post = postprocessor or PostProcessor()

    def run_episode_from_indices(
        self,
        images_dir: Path,
        frame_indices: Iterable[int],
        homography_src_dst: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> EpisodeRunResult:
        """
        Execute YOLOv11-seg inference on each frame in an episode, feed the
        resulting masks through the postprocessor, and compute lane metrics.
        
        Parameters
        ----------
        images_dir:
            Directory containing RGB frames named like `_{idx:04d}.jpg`.
        frame_indices:
            Iterable of integer frame indices (e.g. range(start_idx, end_idx+1)).
        homography_src_dst:
            Optional tuple of (src_pts, dst_pts) to use a fixed homography.
            If None, homography is computed from the first lane mask.
        """
        masks_by_index, _ = self.pre.run_on_indices(images_dir, frame_indices)
        results_by_index, H = self.post.process_run(masks_by_index, homography_src_dst=homography_src_dst)
        lane_metrics = self.post.compute_lane_metrics(results_by_index)
        return EpisodeRunResult(
            results_by_index=results_by_index,
            homography=H,
            lane_metrics=lane_metrics,
            raw_masks_by_index=masks_by_index,
        )

