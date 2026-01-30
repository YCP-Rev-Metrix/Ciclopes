from core.VideoUtil.models import SplitVideo
from core.VideoUtil.FrameSplit import split_video_into_frames

def query_video(video_path: str) -> SplitVideo:
    return split_video_into_frames(video_path)