"""Tests for repeatable detector performance measurements."""

from collections import deque

import pytest

from graffiti_detection import InferenceBenchmark, benchmark_detector


class StubDetector:
    def __init__(self):
        self.calls = []

    def predict(self, source, **predict_kwargs):
        self.calls.append((source, predict_kwargs))
        return [{"confidence": 0.9}]


def test_benchmark_reports_latency_throughput_and_stability():
    detector = StubDetector()
    times = deque([0.00, 0.01, 0.01, 0.03, 0.03, 0.06, 0.06, 0.10])

    result = benchmark_detector(
        detector,
        ["street-a.jpg", "street-b.jpg"],
        warmup_runs=1,
        measured_runs=2,
        predict_kwargs={"conf_threshold": 0.4},
        timer=times.popleft,
    )

    assert isinstance(result, InferenceBenchmark)
    assert result.samples == 2
    assert result.measured_runs == 2
    assert result.measured_inferences == 4
    assert result.total_detections == 4
    assert result.mean_latency_ms == pytest.approx(25.0)
    assert result.median_latency_ms == pytest.approx(25.0)
    assert result.p95_latency_ms == pytest.approx(40.0)
    assert result.latency_stdev_ms == pytest.approx(11.1803, rel=1e-4)
    assert result.throughput_fps == pytest.approx(40.0)
    assert len(detector.calls) == 6
    assert all(call[1] == {"conf_threshold": 0.4} for call in detector.calls)
    assert result.to_dict()["measured_inferences"] == 4


@pytest.mark.parametrize(
    ("sources", "warmup_runs", "measured_runs", "message"),
    [
        ([], 1, 3, "At least one benchmark source"),
        (["street.jpg"], -1, 3, "warmup_runs"),
        (["street.jpg"], 1, 0, "measured_runs"),
    ],
)
def test_benchmark_rejects_invalid_configuration(
    sources, warmup_runs, measured_runs, message
):
    with pytest.raises(ValueError, match=message):
        benchmark_detector(
            StubDetector(),
            sources,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )


def test_benchmark_rejects_timer_regression():
    times = deque([1.0, 0.5])

    with pytest.raises(ValueError, match="timer moved backwards"):
        benchmark_detector(
            StubDetector(),
            ["street.jpg"],
            warmup_runs=0,
            measured_runs=1,
            timer=times.popleft,
        )
