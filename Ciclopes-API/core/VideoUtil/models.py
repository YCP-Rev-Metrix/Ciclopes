from __future__ import annotations
from typing import Optional, List, Any
from dataclasses import dataclass, field

@dataclass
class SplitVideo:
    fps: float
    width: int
    height: int
    frames: List[VideoFrame] = field(default_factory=list)

    def add_frame(self, frame: VideoFrame):
        self.frames.append(frame)

@dataclass
class VideoFrame:
    frame_index: int
    timestamp: float
    image: Any

