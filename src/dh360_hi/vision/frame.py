"""Hardware-independent frame representations and capture source abstractions."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
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

    @property
    def capture_fps(self) -> float:
        """Report measured capture frames-per-second."""
        return self.fps

    @property
    def is_live(self) -> bool:
        """Whether this source is an ongoing live capture stream (e.g. webcam)."""
        return False

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
            timestamp=time.time(),
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
        target_width: int | None = 640,
        target_height: int | None = 480,
        target_fps: float | None = 30.0,
        prefer_mjpeg: bool = True,
        auto_discover: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.prefer_mjpeg = prefer_mjpeg
        self.auto_discover = auto_discover
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._fps = target_fps or 30.0
        self._resolution = (0, 0)
        self._read_timestamps: deque[float] = deque(maxlen=30)
        self._consecutive_read_failures = 0
        self._last_reopen_attempt: float = 0.0

        # Dedicated background capture worker for decoupled, high-throughput (>=15 FPS) reads
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._latest_image: np.ndarray | None = None

    def _open_device(self, index: int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None

        # On Linux V4L2 systems, disable dynamic exposure framerate throttling to sustain full FPS
        if sys.platform.startswith("linux"):
            try:
                subprocess.run(
                    [
                        "v4l2-ctl",
                        "-d",
                        f"/dev/video{index}",
                        "--set-ctrl=exposure_dynamic_framerate=0",
                    ],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        if self.prefer_mjpeg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))

        if self.target_width and self.target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        if self.target_fps:
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Optimize latency: keep buffer size minimal to avoid processing stale frames
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Confirm that a frame can actually be captured from this device (allow up to 5 reads for UVC sync)
        success = False
        for _ in range(5):
            success, _ = cap.read()
            if success:
                break
            time.sleep(0.03)

        if not success:
            cap.release()
            return None

        return cap

    def _discover_candidate_indices(self) -> list[int]:
        """Discover candidate camera device indices, prioritizing physical camera nodes and filtering out platform codecs/ISPs."""
        candidates = [self.camera_index]
        if self.auto_discover:
            if sys.platform.startswith("linux"):
                v4l_indices: list[int] = []
                v4l_dir = Path("/sys/class/video4linux")
                ignored_keywords = ("bcm2835", "isp", "codec", "hevc", "unicam", "dummy")
                if v4l_dir.exists():
                    for entry in sorted(v4l_dir.glob("video*")):
                        try:
                            num = int(entry.name.replace("video", ""))
                            dev_path = Path(f"/dev/video{num}")
                            if not dev_path.exists():
                                continue
                            # Inspect sysfs name to skip hardware decoders/encoders and ISPs
                            name_file = entry / "name"
                            if name_file.exists():
                                dev_name = name_file.read_text().strip().lower()
                                if any(kw in dev_name for kw in ignored_keywords):
                                    continue
                            v4l_indices.append(num)
                        except Exception:
                            pass
                if v4l_indices:
                    if self.camera_index in v4l_indices:
                        candidates = [self.camera_index] + [i for i in v4l_indices if i != self.camera_index]
                    else:
                        candidates = list(v4l_indices)
                else:
                    for idx in [0, 1, 2]:
                        if idx not in candidates:
                            candidates.append(idx)
            else:
                for idx in [0, 1, 2]:
                    if idx not in candidates:
                        candidates.append(idx)
        return candidates

    def _recover_device(self) -> bool:
        """Attempt to recover from a capture error or USB device re-enumeration (thread-safe)."""
        if not self._device_lock.acquire(blocking=False):
            return False
        try:
            old_cap = self._cap
            self._cap = None
            if old_cap is not None:
                try:
                    old_cap.release()
                except Exception:
                    pass

            candidates = self._discover_candidate_indices()
            new_cap = None
            new_index = self.camera_index
            for idx in candidates:
                new_cap = self._open_device(idx)
                if new_cap is not None:
                    new_index = idx
                    break

            if new_cap is not None:
                self.camera_index = new_index
                self._cap = new_cap
                self._consecutive_read_failures = 0
                width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
                self._resolution = (width, height)
                if raw_fps > 0:
                    self._fps = float(raw_fps)
                return True
            return False
        finally:
            self._device_lock.release()

    def open(self) -> None:
        candidates = self._discover_candidate_indices()

        cap = None
        selected_index = self.camera_index
        for _attempt in range(3):
            for idx in candidates:
                cap = self._open_device(idx)
                if cap is not None:
                    selected_index = idx
                    break
            if cap is not None:
                break
            time.sleep(0.1)

        if cap is None:
            raise RuntimeError(
                f"Failed to open webcam at index {self.camera_index} (scanned candidates {candidates})"
            )

        self.camera_index = selected_index
        self._cap = cap

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)

        self._resolution = (width, height)
        if raw_fps > 0:
            self._fps = float(raw_fps)
        self._frame_id = 0
        self._read_timestamps.clear()
        self._consecutive_read_failures = 0

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_worker,
            name=f"WebcamCapture-{self.camera_index}",
            daemon=True,
        )
        self._thread.start()

    def _capture_worker(self) -> None:
        """Background thread continuously pulling frames to prevent capture stalls."""
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                if self.auto_discover:
                    now = time.monotonic()
                    if now - self._last_reopen_attempt >= 1.5:
                        self._last_reopen_attempt = now
                        try:
                            self._recover_device()
                        except Exception:
                            pass
                time.sleep(0.05)
                continue

            success, image = cap.read()
            if success and image is not None:
                now = time.monotonic()
                with self._frame_lock:
                    self._latest_image = image
                self._read_timestamps.append(now)
                self._consecutive_read_failures = 0
            else:
                self._consecutive_read_failures += 1
                if self._consecutive_read_failures >= 3:
                    with self._frame_lock:
                        self._latest_image = None
                now = time.monotonic()
                if (
                    self.auto_discover
                    and self._consecutive_read_failures >= 5
                    and (now - self._last_reopen_attempt >= 1.5)
                ):
                    self._last_reopen_attempt = now
                    try:
                        self._recover_device()
                    except Exception:
                        pass
                time.sleep(0.02)

    def read(self) -> Frame | None:
        now = time.monotonic()

        # Check freshness: if no frame was read in the last 2.0s, worker is stalled or camera dropped
        if not self._read_timestamps or (now - self._read_timestamps[-1] > 2.0):
            with self._frame_lock:
                self._latest_image = None

        # Retrieve latest image from background worker
        with self._frame_lock:
            image = self._latest_image

        # If background thread hasn't pulled first frame yet, wait briefly
        if image is None:
            for _ in range(5):
                time.sleep(0.01)
                with self._frame_lock:
                    image = self._latest_image
                if image is not None:
                    break

        if image is None:
            return None

        frame = Frame(
            image=image.copy(),
            frame_id=self._frame_id,
            timestamp=time.time(),
            source_info=f"webcam:{self.camera_index}",
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_image = None
        self._read_timestamps.clear()

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def capture_fps(self) -> float:
        """Measured rolling capture frame rate from frame read timestamps."""
        if len(self._read_timestamps) >= 2:
            span = self._read_timestamps[-1] - self._read_timestamps[0]
            if span > 0:
                return float(len(self._read_timestamps) - 1) / span
        return self._fps

    @property
    def is_live(self) -> bool:
        """Webcam is an ongoing live capture stream."""
        return True


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
            timestamp=time.time(),
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


class ResilientWebcamFrameSource(FrameSource):
    """Resilient frame source providing continuous fallback while auto-reconnecting to hardware."""

    def __init__(
        self,
        camera_index: int = 0,
        target_width: int = 640,
        target_height: int = 480,
        target_fps: float = 30.0,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self._synthetic = SyntheticFrameSource(
            width=target_width, height=target_height, fps=target_fps, num_frames=10_000_000
        )
        self._webcam: WebcamFrameSource | None = None
        self._last_reconnect_attempt = 0.0
        self._is_opened = False
        self._lock = threading.Lock()

    def open(self) -> None:
        self._is_opened = True
        self._synthetic.open()
        self._try_connect_webcam()

    def _try_connect_webcam(self) -> bool:
        try:
            ws = WebcamFrameSource(
                camera_index=self.camera_index,
                target_width=self.target_width,
                target_height=self.target_height,
                target_fps=self.target_fps,
                auto_discover=True,
            )
            ws.open()
            with self._lock:
                self._webcam = ws
            logger.info(
                "ResilientWebcamFrameSource acquired physical camera on index %d",
                self.camera_index,
            )
            return True
        except Exception:
            return False

    def read(self) -> Frame | None:
        if not self._is_opened:
            return None

        now = time.monotonic()
        with self._lock:
            if self._webcam is not None and self._webcam.is_opened:
                frame = self._webcam.read()
                if frame is not None:
                    return frame
                if getattr(self._webcam, "_consecutive_read_failures", 0) > 5:
                    try:
                        self._webcam.release()
                    except Exception:
                        pass
                    self._webcam = None

        if now - self._last_reconnect_attempt >= 2.0:
            self._last_reconnect_attempt = now
            if self._try_connect_webcam():
                with self._lock:
                    if self._webcam is not None and self._webcam.is_opened:
                        frame = self._webcam.read()
                        if frame is not None:
                            return frame

        return self._synthetic.read()

    def release(self) -> None:
        self._is_opened = False
        with self._lock:
            if self._webcam is not None:
                try:
                    self._webcam.release()
                except Exception:
                    pass
                self._webcam = None
        self._synthetic.release()

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def resolution(self) -> tuple[int, int]:
        with self._lock:
            if self._webcam is not None and self._webcam.is_opened:
                return self._webcam.resolution
        return self.target_width, self.target_height

    @property
    def fps(self) -> float:
        with self._lock:
            if self._webcam is not None and self._webcam.is_opened:
                return self._webcam.fps
        return self.target_fps

    @property
    def capture_fps(self) -> float:
        with self._lock:
            if self._webcam is not None and self._webcam.is_opened:
                return self._webcam.capture_fps
        return self.target_fps
