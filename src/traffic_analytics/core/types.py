from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detections:
    """Detector output for one frame. Library-agnostic (numpy only).

    xyxy: (N, 4) float32, pixel coords (x1, y1, x2, y2)
    confidence: (N,) float32
    class_id: (N,) int
    """

    xyxy: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    confidence: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    class_id: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    def __len__(self) -> int:
        return len(self.xyxy)