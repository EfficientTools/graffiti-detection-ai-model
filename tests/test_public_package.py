"""Tests for the public package namespace."""

import numpy as np

from graffiti_detection import (
    GraffitiDetector,
    InferenceBenchmark,
    __version__,
    benchmark_detector,
)
from graffiti_detection.data.preprocessing import preprocess_image
from graffiti_detection.evaluation.metrics import calculate_iou


def test_public_detector_class_is_importable():
    assert GraffitiDetector.__name__ == "GraffitiDetector"
    assert InferenceBenchmark.__name__ == "InferenceBenchmark"
    assert benchmark_detector.__name__ == "benchmark_detector"
    assert __version__ == "1.4.0"


def test_public_metrics_namespace_works():
    box = np.array([0, 0, 100, 100])
    assert calculate_iou(box, box) == 1.0


def test_public_preprocessing_namespace_works():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    processed = preprocess_image(image, target_size=64)
    assert processed.shape == (1, 3, 64, 64)
