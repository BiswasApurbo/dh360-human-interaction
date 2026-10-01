"""Benchmarking and latency measurement utilities with monotonic timing."""

from __future__ import annotations

import platform
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

MAC_DISCLAIMER = (
    "Development benchmark — MacBook M4 Air; not representative of Raspberry Pi performance."
)


@dataclass(frozen=True)
class LatencyStats:
    """Statistical summary of execution latencies."""

    count: int
    total_time_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    average_fps: float


@dataclass(frozen=True)
class ConfidenceStats:
    """Statistical summary of face detection confidence scores."""

    total_detections: int
    frames_with_detections: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    p50_confidence: float


class StageTimer:
    """Context manager for timing individual pipeline stages with monotonic clock."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._elapsed_ms: float = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self._elapsed_ms


@dataclass
class PipelineBenchmark:
    """Tracks frame processing throughput, rolling FPS, and per-stage latency distributions."""

    window_size: int = 30
    _frame_count: int = 0
    _start_time: float | None = None
    _last_frame_time: float | None = None
    _frame_timestamps: deque[float] = field(default_factory=deque)
    _overall_latencies_ms: list[float] = field(default_factory=list)
    _stage_latencies_ms: dict[str, list[float]] = field(default_factory=dict)
    _all_confidences: list[float] = field(default_factory=list)
    _frames_with_faces: int = 0

    def start(self) -> None:
        """Mark start of benchmarking run."""
        self._start_time = time.perf_counter()
        self._frame_count = 0
        self._frame_timestamps.clear()
        self._overall_latencies_ms.clear()
        self._stage_latencies_ms.clear()
        self._all_confidences.clear()
        self._frames_with_faces = 0

    def record_frame(
        self,
        total_latency_ms: float | None = None,
        stage_latencies_ms: Mapping[str, float] | None = None,
        detections_count: int = 0,
        confidences: Sequence[float] | None = None,
    ) -> None:
        """Record the completion of a single frame, stage timings, and detection scores."""
        now = time.perf_counter()
        if self._start_time is None:
            self._start_time = now

        self._frame_count += 1
        self._last_frame_time = now

        # Maintain rolling timestamps
        self._frame_timestamps.append(now)
        while len(self._frame_timestamps) > self.window_size:
            self._frame_timestamps.popleft()

        # Record overall latency
        if total_latency_ms is not None:
            self._overall_latencies_ms.append(total_latency_ms)

        # Record per-stage latencies
        if stage_latencies_ms:
            for stage, latency in stage_latencies_ms.items():
                if stage not in self._stage_latencies_ms:
                    self._stage_latencies_ms[stage] = []
                self._stage_latencies_ms[stage].append(latency)

        # Record detection metrics
        if detections_count > 0:
            self._frames_with_faces += 1

        if confidences:
            self._all_confidences.extend(confidences)

    @property
    def frame_count(self) -> int:
        """Total number of frames recorded."""
        return self._frame_count

    @property
    def elapsed_time_s(self) -> float:
        """Total elapsed time since start in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._last_frame_time or time.perf_counter()
        return max(end - self._start_time, 1e-9)

    @property
    def average_fps(self) -> float:
        """Average FPS over entire run."""
        if self._frame_count == 0 or self._start_time is None:
            return 0.0
        return self._frame_count / self.elapsed_time_s

    @property
    def rolling_fps(self) -> float:
        """Rolling FPS calculated over the recent sliding window."""
        if len(self._frame_timestamps) < 2:
            return self.average_fps
        window_duration = self._frame_timestamps[-1] - self._frame_timestamps[0]
        if window_duration <= 0:
            return 0.0
        return (len(self._frame_timestamps) - 1) / window_duration

    @staticmethod
    def _compute_stats(latencies: list[float], total_time_s: float) -> LatencyStats:
        """Calculate statistical metrics from a list of latencies in ms."""
        if not latencies:
            return LatencyStats(
                count=0,
                total_time_ms=0.0,
                mean_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                p50_ms=0.0,
                p90_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                average_fps=0.0,
            )

        arr = np.array(latencies, dtype=np.float64)
        count = len(arr)
        total_time_ms = float(np.sum(arr))
        fps = (count / total_time_s) if total_time_s > 0 else 0.0

        return LatencyStats(
            count=count,
            total_time_ms=total_time_ms,
            mean_ms=float(np.mean(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            average_fps=fps,
        )

    def get_overall_stats(self) -> LatencyStats:
        """Compute latency statistics for the overall pipeline."""
        return self._compute_stats(self._overall_latencies_ms, self.elapsed_time_s)

    def get_stage_stats(self, stage_name: str) -> LatencyStats:
        """Compute latency statistics for a specific stage."""
        latencies = self._stage_latencies_ms.get(stage_name, [])
        return self._compute_stats(latencies, self.elapsed_time_s)

    def get_confidence_stats(self) -> ConfidenceStats:
        """Compute statistical summary of detection confidence scores."""
        if not self._all_confidences:
            return ConfidenceStats(
                total_detections=0,
                frames_with_detections=0,
                mean_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                p50_confidence=0.0,
            )

        arr = np.array(self._all_confidences, dtype=np.float64)
        return ConfidenceStats(
            total_detections=len(arr),
            frames_with_detections=self._frames_with_faces,
            mean_confidence=float(np.mean(arr)),
            min_confidence=float(np.min(arr)),
            max_confidence=float(np.max(arr)),
            p50_confidence=float(np.percentile(arr, 50)),
        )

    def get_all_stage_names(self) -> list[str]:
        """List all tracked pipeline stages."""
        return sorted(self._stage_latencies_ms.keys())

    def format_summary(
        self,
        device_label: str = "MacBook M4 Air",
        notes: str = "",
    ) -> str:
        """Produce a formatted benchmark report."""
        overall = self.get_overall_stats()
        conf_stats = self.get_confidence_stats()
        system_info = f"{platform.system()} {platform.machine()}"

        lines = [
            "================================================================================",
            "DH360 HUMAN INTERACTION — VISION BENCHMARK REPORT",
            "================================================================================",
            f"Environment / Host:       {device_label} [{system_info}]",
            f"Disclaimer:               {MAC_DISCLAIMER}",
            "--------------------------------------------------------------------------------",
            f"Frames Processed:         {self.frame_count}",
            f"Total Elapsed Time:       {self.elapsed_time_s:.3f} s",
            f"Average Throughput:       {self.average_fps:.2f} FPS",
            f"Final Rolling FPS (N={self.window_size}): {self.rolling_fps:.2f} FPS",
            "--------------------------------------------------------------------------------",
            "Overall Latency Distribution (ms):",
            f"  Mean:  {overall.mean_ms:6.2f} ms",
            f"  Min:   {overall.min_ms:6.2f} ms",
            f"  p50:   {overall.p50_ms:6.2f} ms",
            f"  p90:   {overall.p90_ms:6.2f} ms",
            f"  p95:   {overall.p95_ms:6.2f} ms",
            f"  p99:   {overall.p99_ms:6.2f} ms",
            f"  Max:   {overall.max_ms:6.2f} ms",
        ]

        if conf_stats.total_detections > 0:
            lines.append(
                "--------------------------------------------------------------------------------"
            )
            lines.append("Face Detection Metrics:")
            lines.append(f"  Total Faces Detected:   {conf_stats.total_detections}")
            rate = (conf_stats.frames_with_detections / max(self.frame_count, 1)) * 100
            lines.append(
                f"  Frames With Faces:      {conf_stats.frames_with_detections}/{self.frame_count} "
                f"({rate:.1f}%)"
            )
            lines.append(
                f"  Confidence (Mean/p50):  {conf_stats.mean_confidence:.4f} / "
                f"{conf_stats.p50_confidence:.4f}"
            )
            lines.append(
                f"  Confidence (Min/Max):   {conf_stats.min_confidence:.4f} / "
                f"{conf_stats.max_confidence:.4f}"
            )

        stage_names = self.get_all_stage_names()
        if stage_names:
            lines.append(
                "--------------------------------------------------------------------------------"
            )
            lines.append("Per-Stage Latency Breakdown (ms):")
            for stage in stage_names:
                stats = self.get_stage_stats(stage)
                lines.append(
                    f"  [{stage:<20}]  mean: {stats.mean_ms:5.2f} ms | "
                    f"p50: {stats.p50_ms:5.2f} ms | "
                    f"p95: {stats.p95_ms:5.2f} ms | "
                    f"p99: {stats.p99_ms:5.2f} ms"
                )

        if notes:
            lines.append(
                "--------------------------------------------------------------------------------"
            )
            lines.append(f"Notes:\n{notes}")

        lines.append(
            "================================================================================"
        )
        return "\n".join(lines)
