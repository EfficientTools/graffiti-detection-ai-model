"""Repeatable runtime benchmarks for graffiti detectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from statistics import fmean, median, pstdev
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Protocol, Sequence


class DetectorProtocol(Protocol):
    """Minimum detector contract required by the benchmark runner."""

    def predict(self, source: Any, **predict_kwargs: Any) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class InferenceBenchmark:
    """Summary of repeated, single-image detector measurements."""

    samples: int
    measured_runs: int
    measured_inferences: int
    total_detections: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    latency_stdev_ms: float
    throughput_fps: float

    def to_dict(self) -> Dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def benchmark_detector(
    detector: DetectorProtocol,
    sources: Iterable[Any],
    *,
    warmup_runs: int = 1,
    measured_runs: int = 3,
    predict_kwargs: Mapping[str, Any] | None = None,
    timer: Callable[[], float] = perf_counter,
) -> InferenceBenchmark:
    """Measure single-image latency and throughput across repeated passes.

    Warm-up predictions are excluded so model initialization does not distort
    steady-state latency. Each measured source is timed independently.
    """
    source_list = list(sources)
    if not source_list:
        raise ValueError("At least one benchmark source is required.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be zero or greater.")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be greater than zero.")

    kwargs = dict(predict_kwargs or {})
    for _ in range(warmup_runs):
        for source in source_list:
            detector.predict(source, **kwargs)

    latencies_ms: List[float] = []
    total_detections = 0
    for _ in range(measured_runs):
        for source in source_list:
            started_at = timer()
            detections = detector.predict(source, **kwargs)
            elapsed_ms = (timer() - started_at) * 1_000
            if elapsed_ms < 0:
                raise ValueError("Benchmark timer moved backwards.")
            latencies_ms.append(elapsed_ms)
            total_detections += len(detections)

    measured_inferences = len(latencies_ms)
    total_seconds = sum(latencies_ms) / 1_000
    ordered_latencies = sorted(latencies_ms)
    p95_index = max(0, ceil(0.95 * measured_inferences) - 1)

    return InferenceBenchmark(
        samples=len(source_list),
        measured_runs=measured_runs,
        measured_inferences=measured_inferences,
        total_detections=total_detections,
        mean_latency_ms=fmean(latencies_ms),
        median_latency_ms=median(latencies_ms),
        p95_latency_ms=ordered_latencies[p95_index],
        latency_stdev_ms=pstdev(latencies_ms),
        throughput_fps=measured_inferences / total_seconds if total_seconds > 0 else 0.0,
    )
