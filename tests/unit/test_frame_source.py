"""Unit tests for Frame and FrameSource implementations."""

import numpy as np
import pytest

from dh360_hi.vision.frame import (
    Frame,
    SyntheticFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
)


def test_frame_dataclass_properties() -> None:
    """Verify Frame dataclass properties and dimensions."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(
        image=img,
        frame_id=42,
        timestamp=100.5,
        source_info="test_cam",
    )

    assert frame.frame_id == 42
    assert frame.timestamp == 100.5
    assert frame.source_info == "test_cam"
    assert frame.width == 640
    assert frame.height == 480
    assert frame.resolution == (640, 480)
    assert frame.image.shape == (480, 640, 3)


def test_synthetic_frame_source_lifecycle() -> None:
    """Test SyntheticFrameSource context manager, reading, and iteration."""
    source = SyntheticFrameSource(width=320, height=240, fps=25.0, num_frames=5)
    assert not source.is_opened
    assert source.resolution == (320, 240)
    assert source.fps == 25.0

    with source:
        assert source.is_opened
        frames = list(source)
        assert len(frames) == 5

        for i, frame in enumerate(frames):
            assert frame.frame_id == i
            assert frame.resolution == (320, 240)
            assert frame.source_info == "synthetic_source"
            assert isinstance(frame.timestamp, float)

    assert not source.is_opened


def test_synthetic_frame_source_manual_read() -> None:
    """Test manual open/read/release calls on SyntheticFrameSource."""
    source = SyntheticFrameSource(width=160, height=120, num_frames=2)
    source.open()
    assert source.is_opened

    f1 = source.read()
    assert f1 is not None
    assert f1.frame_id == 0

    f2 = source.read()
    assert f2 is not None
    assert f2.frame_id == 1

    f3 = source.read()
    assert f3 is None

    source.release()
    assert not source.is_opened


def test_videofile_missing_file_raises_error(tmp_path: pytest.TempPathFactory) -> None:
    """Verify VideoFileFrameSource raises FileNotFoundError when path is invalid."""
    non_existent = tmp_path / "does_not_exist.mp4"
    source = VideoFileFrameSource(file_path=non_existent)
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        source.open()


def test_webcam_frame_source_init() -> None:
    """Verify WebcamFrameSource initialization and configuration values."""
    # Custom values
    source = WebcamFrameSource(
        camera_index=99,
        target_width=1280,
        target_height=720,
        target_fps=30.0,
        prefer_mjpeg=False,
    )
    assert source.camera_index == 99
    assert source.target_width == 1280
    assert source.target_height == 720
    assert source.target_fps == 30.0
    assert source.prefer_mjpeg is False
    assert not source.is_opened

    # Default values: 640x480 MJPEG @ 30.0 FPS
    default_source = WebcamFrameSource(camera_index=0)
    assert default_source.camera_index == 0
    assert default_source.target_width == 640
    assert default_source.target_height == 480
    assert default_source.target_fps == 30.0
    assert default_source.prefer_mjpeg is True
    assert default_source.capture_fps == 30.0
    assert default_source.is_live is True
