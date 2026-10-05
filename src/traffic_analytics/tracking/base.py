from __future__ import annotations

from typing import Protocol

from traffic_analytics.core.types import Detections, Tracks


class Tracker(Protocol):
    """Contract: detections of the current frame in, identified tracks out.

    A tracker is stateful: it remembers previous frames between calls.
    """

    def update(self, detections: Detections) -> Tracks: ...