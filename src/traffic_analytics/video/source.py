from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Immutable metadata about a video."""
    width: int
    height: int
    fps: float
    total_frames: int  # <= 0 for live streams


class VideoSource:
    """Reads frames from a video file. Knows nothing about detection or tracking."""

    def __init__(self, path: Union[str, Path]):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.info = VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        """Yield BGR frames one at a time (memory stays constant)."""
        count = 0
        while max_frames is None or count < max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame
            count += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()