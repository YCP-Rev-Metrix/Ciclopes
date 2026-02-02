from core.VideoUtil.models import SplitVideo, VideoFrame
from typing import List, Optional, Union
import torch
import cv2
import numpy as np

def split_video_into_frames(video_path: str) -> SplitVideo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at path: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    split_video = SplitVideo(fps=fps, width=width, height=height)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        split_video.add_frame(
            VideoFrame(frame_index=frame_index, timestamp=timestamp_s, image=frame)
        )
        frame_index += 1

    cap.release()
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