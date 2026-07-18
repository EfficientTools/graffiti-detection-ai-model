"""Reproducible evaluation reports and release quality gates."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

METRIC_KEYS = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


@dataclass(frozen=True)
class EvaluationGates:
    """Optional minimum quality and maximum latency requirements."""

    min_map50: Optional[float] = None
    min_map50_95: Optional[float] = None
    min_precision: Optional[float] = None
    min_recall: Optional[float] = None
    max_inference_ms: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("min_map50", "min_map50_95", "min_precision", "min_recall"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.max_inference_ms is not None and self.max_inference_ms <= 0:
            raise ValueError("max_inference_ms must be greater than zero.")

    @property
    def is_configured(self) -> bool:
        return any(value is not None for value in asdict(self).values())

    def assess(
        self,
        metrics: Mapping[str, float],
        speed_ms: Mapping[str, float],
    ) -> list[str]:
        """Return human-readable failures for unmet or missing requirements."""
        failures = []
        minimums = {
            "map50": self.min_map50,
            "map50_95": self.min_map50_95,
            "precision": self.min_precision,
            "recall": self.min_recall,
        }
        for metric_name, minimum in minimums.items():
            if minimum is None:
                continue
            actual = metrics.get(metric_name)
            if actual is None:
                failures.append(f"{metric_name} is unavailable; required minimum is {minimum:.4f}")
            elif actual < minimum:
                failures.append(
                    f"{metric_name} {actual:.4f} is below required minimum {minimum:.4f}"
                )

        if self.max_inference_ms is not None:
            actual_latency = speed_ms.get("inference")
            if actual_latency is None:
                failures.append(
                    "inference latency is unavailable; required maximum is "
                    f"{self.max_inference_ms:.2f} ms"
                )
            elif actual_latency > self.max_inference_ms:
                failures.append(
                    f"inference latency {actual_latency:.2f} ms exceeds required maximum "
                    f"{self.max_inference_ms:.2f} ms"
                )
        return failures

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


def normalize_metrics(raw_metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Map Ultralytics metric names to a stable public report schema."""
    return {
        name: float(raw_metrics[source_name])
        for name, source_name in METRIC_KEYS.items()
        if source_name in raw_metrics
    }


def normalize_numeric_mapping(values: Mapping[str, Any]) -> Dict[str, float]:
    """Convert numeric result mappings to JSON-safe floats."""
    normalized = {}
    for name, value in values.items():
        try:
            normalized[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def artifact_identity(path: Path) -> Dict[str, Any]:
    """Describe a file so benchmark inputs can be reproduced exactly."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation artifact not found: {resolved}")

    digest = hashlib.sha256()
    with resolved.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_evaluation_report(
    *,
    model_path: Path,
    data_path: Path,
    metrics: Mapping[str, float],
    speed_ms: Mapping[str, float],
    evaluation_config: Mapping[str, Any],
    gates: EvaluationGates,
    environment: Mapping[str, str],
    generated_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Build a versioned report containing evidence, inputs, and gate results."""
    failures = gates.assess(metrics, speed_ms)
    timestamp = generated_at or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "model": artifact_identity(model_path),
        "dataset": artifact_identity(data_path),
        "evaluation": dict(evaluation_config),
        "environment": dict(environment),
        "metrics": dict(metrics),
        "speed_ms_per_image": dict(speed_ms),
        "quality_gates": {
            "configured": gates.to_dict(),
            "passed": not failures,
            "failures": failures,
        },
    }
