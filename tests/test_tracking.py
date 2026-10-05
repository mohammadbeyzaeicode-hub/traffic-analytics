import numpy as np

from traffic_analytics.core.types import Detections, Tracks
from traffic_analytics.tracking import ByteTrackTracker


def det(boxes, conf):
    n = len(boxes)
    return Detections(
        xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.full(n, conf, dtype=np.float32),
        class_id=np.full(n, 2, dtype=int),
    )


def moving_box(frame, x0=100, speed=5, y=100, size=60):
    x = x0 + frame * speed
    return [x, y, x + size, y + size]


def test_same_object_keeps_same_id():
    tracker = ByteTrackTracker(fps=30)
    ids = []
    for f in range(20):
        tracks = tracker.update(det([moving_box(f)], 0.9))
        if len(tracks):
            ids.append(int(tracks.track_id[0]))
    assert len(ids) >= 15
    assert len(set(ids)) == 1


def test_two_objects_get_different_ids():
    tracker = ByteTrackTracker(fps=30)
    for f in range(10):
        tracks = tracker.update(det([moving_box(f, y=100), moving_box(f, y=400)], 0.9))
    assert len(tracks) == 2
    assert len(set(tracks.track_id.tolist())) == 2


def test_empty_frame_returns_valid_tracks():
    tracker = ByteTrackTracker(fps=30)
    tracks = tracker.update(Detections())
    assert isinstance(tracks, Tracks)
    assert len(tracks) == 0


def test_short_gap_keeps_id():
    """Object missing for 3 frames (e.g. occluded) should get the same ID back."""
    tracker = ByteTrackTracker(fps=30, lost_track_buffer=30)
    ids = []
    for f in range(20):
        if 8 <= f < 11:
            tracker.update(Detections())
            continue
        tracks = tracker.update(det([moving_box(f)], 0.9))
        if len(tracks):
            ids.append(int(tracks.track_id[0]))
    assert len(set(ids)) == 1


def count_tracks(conf, activation_threshold, frames=20):
    tracker = ByteTrackTracker(fps=30, activation_threshold=activation_threshold)
    return sum(len(tracker.update(det([moving_box(f)], conf))) for f in range(frames))


def test_new_track_needs_confidence_above_threshold_plus_0_1():
    """Real rule (verified in supervision source): a NEW track starts only if
    confidence >= activation_threshold + 0.1."""
    assert count_tracks(conf=0.18, activation_threshold=0.25) == 0
    assert count_tracks(conf=0.30, activation_threshold=0.25) == 0   # 0.30 < 0.35
    assert count_tracks(conf=0.40, activation_threshold=0.25) > 0    # 0.40 >= 0.35


def test_lower_activation_threshold_recovers_weak_object():
    assert count_tracks(conf=0.30, activation_threshold=0.15) > 0    # 0.30 >= 0.25


def test_weak_detection_extends_existing_track():
    """Strong start, then confidence drops below threshold: track should survive."""
    tracker = ByteTrackTracker(fps=30, activation_threshold=0.25)
    ids = []
    for f in range(20):
        conf = 0.9 if f < 8 else 0.18
        tracks = tracker.update(det([moving_box(f)], conf))
        if len(tracks):
            ids.append(int(tracks.track_id[0]))
    assert len(ids) >= 15 and len(set(ids)) == 1
