import cv2
import numpy as np
import pytest

from traffic_analytics.video import VideoSource


@pytest.fixture
def tiny_video(tmp_path):
    """Create a 20-frame 64x48 synthetic video, so tests don't need real data."""
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(20):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def test_info(tiny_video):
    with VideoSource(tiny_video) as src:
        assert (src.info.width, src.info.height) == (64, 48)
        assert src.info.fps == pytest.approx(10)
        assert src.info.total_frames == 20


def test_frames_shape_and_count(tiny_video):
    with VideoSource(tiny_video) as src:
        frames = list(src.frames())
    assert len(frames) == 20
    assert frames[0].shape == (48, 64, 3)  # (height, width, channels)


def test_max_frames(tiny_video):
    with VideoSource(tiny_video) as src:
        assert len(list(src.frames(max_frames=5))) == 5


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "nope.mp4")