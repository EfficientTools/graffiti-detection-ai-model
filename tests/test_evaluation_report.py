"""Tests for reproducible model evidence and release gates."""

from datetime import datetime, timezone

import pytest

from graffiti_detection.evaluation.report import (
    EvaluationGates,
    artifact_identity,
    build_evaluation_report,
    normalize_metrics,
    normalize_numeric_mapping,
)


def test_normalizes_ultralytics_metrics_and_numeric_speed():
    metrics = normalize_metrics(
        {
            "metrics/mAP50(B)": 0.82,
            "metrics/mAP50-95(B)": 0.61,
            "metrics/precision(B)": 0.79,
            "metrics/recall(B)": 0.75,
            "ignored": 12,
        }
    )
    speed = normalize_numeric_mapping({"inference": 14, "invalid": object()})

    assert metrics == {
        "map50": 0.82,
        "map50_95": 0.61,
        "precision": 0.79,
        "recall": 0.75,
    }
    assert speed == {"inference": 14.0}


def test_quality_gates_report_regressions_and_missing_evidence():
    gates = EvaluationGates(
        min_map50=0.85,
        min_precision=0.75,
        min_recall=0.80,
        max_inference_ms=20,
    )

    failures = gates.assess(
        {"map50": 0.82, "precision": 0.79},
        {"inference": 24.5},
    )

    assert failures == [
        "map50 0.8200 is below required minimum 0.8500",
        "recall is unavailable; required minimum is 0.8000",
        "inference latency 24.50 ms exceeds required maximum 20.00 ms",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_map50": -0.1},
        {"min_precision": 1.1},
        {"max_inference_ms": 0},
    ],
)
def test_quality_gates_reject_invalid_thresholds(kwargs):
    with pytest.raises(ValueError):
        EvaluationGates(**kwargs)


def test_builds_versioned_report_with_artifact_fingerprints(tmp_path):
    model = tmp_path / "best.pt"
    dataset = tmp_path / "dataset.yaml"
    model.write_bytes(b"model")
    dataset.write_text("names: [graffiti]\n")

    report = build_evaluation_report(
        model_path=model,
        data_path=dataset,
        metrics={"map50": 0.9, "precision": 0.85, "recall": 0.8},
        speed_ms={"inference": 12.0},
        evaluation_config={"split": "test"},
        gates=EvaluationGates(min_map50=0.85, max_inference_ms=15),
        environment={"python": "3.13.0"},
        generated_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == 1
    assert report["generated_at"] == "2026-07-18T00:00:00+00:00"
    assert report["model"]["sha256"] == artifact_identity(model)["sha256"]
    assert report["dataset"]["size_bytes"] == len(dataset.read_bytes())
    assert report["quality_gates"]["passed"] is True
    assert report["quality_gates"]["failures"] == []
