"""Evaluation utilities exposed under the public package namespace."""

from graffiti_detection.evaluation.metrics import (
    DetectionMetrics,
    calculate_ap,
    calculate_confusion_matrix,
    calculate_iou,
    calculate_map,
    calculate_precision_recall_f1,
    non_max_suppression,
)
from graffiti_detection.evaluation.report import (
    EvaluationGates,
    artifact_identity,
    build_evaluation_report,
    normalize_metrics,
    normalize_numeric_mapping,
)

__all__ = [
    "DetectionMetrics",
    "calculate_ap",
    "calculate_confusion_matrix",
    "calculate_iou",
    "calculate_map",
    "calculate_precision_recall_f1",
    "non_max_suppression",
    "EvaluationGates",
    "artifact_identity",
    "build_evaluation_report",
    "normalize_metrics",
    "normalize_numeric_mapping",
]
