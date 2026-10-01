"""Hardware-independent frame representations and capture source abstractions."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """Represents a single captured image frame with associated metadata."""

    image: np.ndarray
    frame_id: int
    timestamp: float  # Monotonic timestamp (seconds)
    source_info: str = ""

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        """Frame resolution as (width, height)."""
        return self.width, self.height


class FrameSource(ABC):
    """Abstract interface for video and camera capture sources."""

    @abstractmethod
    def open(self) -> None:
        """Open and initialize the frame source."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Read the next frame from the source.

        Returns:
            Frame object if successful, None if end of stream or read failed.
        """

    @abstractmethod
    def release(self) -> None:
        """Release underlying camera or stream resources."""

    @property
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if the frame source is currently open."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Report stream resolution as (width, height)."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Report stream frames-per-second."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.release()

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames sequentially until stream termination."""
        while self.is_opened:
            frame = self.read()
            if frame is None:
                break
            yield frame


class VideoFileFrameSource(FrameSource):
    """Prerecorded video file frame source."""

    def __init__(self, file_path: str | Path, loop: bool = False) -> None:
        self.file_path = Path(file_path)
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._fps = 30.0
        self._resolution = (0, 0)

    def open(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.file_path}")

        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.file_path}")

        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(raw_fps) if raw_fps > 0 else 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._resolution = (width, height)
        self._frame_id = 0

    def read(self) -> Frame | None:
        if self._cap is None or not self._cap.isOpened():
            return None

        success, image = self._cap.read()
        if not success or image is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = self._cap.read()
                if not success or image is None:
                    return None
            else:
                return None

        frame = Frame(
            image=image,
            frame_id=self._frame_id,
            timestamp=time.monotonic(),
            source_info=str(self.file_path.name),
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> float:
        return self._fps


class WebcamFrameSource(FrameSource):
    """Webcam frame source for local camera devices."""

    def __init__(
        self,
        camera_index: int = 0,
        target_width: int | None = None,
        target_height: int | None = None,
        target_fps: float | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._fps = target_fps or 30.0
        self._resolution = (0, 0)

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self.camera_index}")

        if self.target_width and self.target_height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._resolution = (width, height)
        if raw_fps > 0:
            self._fps = float(raw_fps)
        self._frame_id = 0

    def read(self) -> Frame | None:
        if self._cap is None or not self._cap.isOpened():
            return None

        success, image = self._cap.read()
        if not success or image is None:
            return None

        frame = Frame(
            image=image,
            frame_id=self._frame_id,
            timestamp=time.monotonic(),
            source_info=f"webcam:{self.camera_index}",
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> float:
        return self._fps


class SyntheticFrameSource(FrameSource):
    """Deterministic synthetic frame generator for testing and headless benchmarks."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        num_frames: int = 100,
    ) -> None:
        self._width = width
        self._height = height
        self._fps = fps
        self._num_frames = num_frames
        self._frame_id = 0
        self._is_opened = False

    def open(self) -> None:
        self._is_opened = True
        self._frame_id = 0

    def read(self) -> Frame | None:
        if not self._is_opened or self._frame_id >= self._num_frames:
            return None

        # Create a deterministic synthetic frame with geometric shapes
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[:, :] = (30, 30, 30)  # Dark background

        # Draw a moving reference pattern across frames
        center_x = int((self._width // 2) + 50 * np.sin(self._frame_id * 0.1))
        center_y = int((self._height // 2) + 30 * np.cos(self._frame_id * 0.1))
        cv2.circle(image, (center_x, center_y), 40, (0, 200, 200), -1)
        cv2.putText(
            image,
            f"Frame: {self._frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        frame = Frame(
            image=image,
            frame_id=self._frame_id,
            timestamp=time.monotonic(),
            source_info="synthetic_source",
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        self._is_opened = False

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def fps(self) -> float:
        return self._fps
