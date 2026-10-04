from __future__ import annotations

from typing import Protocol

import numpy as np

from traffic_analytics.core.types import Detections


class Detector(Protocol):
    """Contract: frame in, detections out. Nothing else."""

    def detect(self, frame: np.ndarray) -> Detections: ...