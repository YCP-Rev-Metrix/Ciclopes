from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass

@dataclass
class SplitVideo:
    frames: List[VideoFrame] = []
    fps: float
    width: int
    height: int

    def add_frame(self, frame: VideoFrame):
        self.frames.append(frame)

@dataclass
class VideoFrame:
    frame_index: int
    timestamp: float
    image: Any

