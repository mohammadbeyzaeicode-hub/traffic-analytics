import warnings
import numpy as np
import supervision as sv
from traffic_analytics.tracking.base import Tracker
from traffic_analytics.core.types import Detections, Tracks


class ByteTrackTracker(Tracker):
    """ByteTrack (supervision implementation) behind our own types.

    activation_threshold: a NEW track starts only at confidence >= this + 0.1.
        Weaker detections (> 0.1) can still extend existing tracks.
    lost_track_buffer: frames to keep a lost track alive (scaled by fps/30).
    matching_threshold: matching strictness (higher = more permissive).
    """
    def __init__(
         self,
        fps: float = 30.0,
        activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        matching_threshold: float = 0.8,
    ):
        with warnings.catch_warnings():
            # sv.ByteTrack is deprecated upstream; replacing it is a one-file change.
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=matching_threshold,
                frame_rate=int(round(fps)),
            )
        
    def update(self, detections: Detections):
        sv_dets = sv.Detections(
            xyxy=detections.xyxy.reshape(-1, 4),
            confidence=detections.confidence,
            class_id=detections.class_id,
        )
        out = self._tracker.update_with_detections(sv_dets)
        if len(out) == 0 or out.tracker_id is None:
            return Tracks()
        return Tracks(
            xyxy=out.xyxy.astype(np.float32),
            confidence=out.confidence.astype(np.float32),
            class_id=out.class_id.astype(int),
            track_id=out.tracker_id.astype(int),
        )
        
