from core.VideoUtil.models import SplitVideo, VideoFrame

def split_video_into_frames(video, fps: int, width: int, height: int) -> SplitVideo:
    split_video = SplitVideo(fps=fps, width=width, height=height)
    #TODO
    # 1. Split video into frames
    # 2. loop through frames, build VideoFrame and append to SplitVideo + hopefully timestamp exists
    return split_video