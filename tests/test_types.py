import numpy as np

from traffic_analytics.core.types import Detections


def test_default_is_empty_but_valid():
    d = Detections()
    assert len(d) == 0
    assert d.xyxy.shape == (0, 4)


def test_instances_do_not_share_arrays():
    a, b = Detections(), Detections()
    assert a.xyxy is not b.xyxy


def test_len_matches_boxes():
    d = Detections(
        xyxy=np.zeros((3, 4), dtype=np.float32),
        confidence=np.ones(3, dtype=np.float32),
        class_id=np.array([2, 2, 7]),
    )
    assert len(d) == 3