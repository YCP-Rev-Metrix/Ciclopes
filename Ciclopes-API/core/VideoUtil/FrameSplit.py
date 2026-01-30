from core.VideoUtil.models import SplitVideo, VideoFrame

import cv2

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