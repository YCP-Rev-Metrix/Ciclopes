from core.VideoUtil.models import SplitVideo, VideoFrame
from typing import List, Optional, Union
import logging
import torch
import cv2
import numpy as np

logger = logging.getLogger("ciclopes.frame_split")

# ── Defaults for OOM protection ─────────────────────────────────────────────
DEFAULT_MAX_FRAMES = 600        # ~20 s at 30 fps
DEFAULT_MAX_DIMENSION = 1024    # inference resolution cap


def _resize_if_needed(frame: np.ndarray, max_dim: int) -> np.ndarray:
    """Down-scale so the longest edge is <= max_dim. No-op if already small."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def split_video_into_frames(
    video_path: str,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_dimension: int = DEFAULT_MAX_DIMENSION,
) -> SplitVideo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at path: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Compute step to sub-sample if the video has more frames than the cap
    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    effective_fps = fps / step if step > 1 else fps

    logger.info(
        "Video %s: %dx%d @ %.1f fps, %d total frames → step=%d (max_frames=%d, max_dim=%d)",
        video_path, width, height, fps, total_frames, step, max_frames, max_dimension,
    )

    split_video = SplitVideo(fps=effective_fps, width=width, height=height)

    frame_index = 0
    kept = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0:
            frame = _resize_if_needed(frame, max_dimension)
            timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            split_video.add_frame(
                VideoFrame(frame_index=kept, timestamp=timestamp_s, image=frame)
            )
            kept += 1
            if kept >= max_frames:
                break

        frame_index += 1

    cap.release()
    logger.info("Extracted %d frames (resized to max_dim=%d)", kept, max_dimension)
    return split_video

def _frames_to_tensor_batch(
    frames_bgr_uint8: List[np.ndarray],
    *,
    rgb: bool = True,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    if not frames_bgr_uint8:
        raise ValueError("frames_bgr_uint8 is empty")

    arr = np.stack(frames_bgr_uint8, axis=0)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (H, W, 3); got stacked shape {arr.shape}")

    if rgb:
        arr = arr[..., ::-1].copy()

    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    if device is not None:
        t = t.to(device=device)
    if normalize:
        t = t.float().div_(255.0)
    return t


def batch_split_video(
    video: SplitVideo,
    batch_size: int,
    *,
    rgb: bool = True,
    normalize: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    drop_last: bool = True,
) -> List[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    frames = video.frames
    if not frames:
        return []

    images = [vf.image for vf in frames]

    if drop_last:
        num_batches = len(images) // batch_size
    else:
        num_batches = (len(images) + batch_size - 1) // batch_size

    batches: List[torch.Tensor] = []
    for i in range(num_batches):
        start = i * batch_size

        end = min(start + batch_size, len(images))

        batch_imgs = images[start:end]
        if drop_last and len(batch_imgs) < batch_size:
            break

        batches.append(
            _frames_to_tensor_batch(
                batch_imgs,
                rgb=rgb,
                normalize=normalize,
                device=device,
            )
        )

    return batches