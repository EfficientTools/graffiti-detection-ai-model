"""Tests for the high-level detector contract."""

import pytest

from graffiti_detection import GraffitiDetector


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_detector_rejects_invalid_confidence_thresholds(threshold, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", lambda _path: object())

    with pytest.raises(ValueError, match="between 0 and 1"):
        GraffitiDetector("model.pt", conf_threshold=threshold)


def test_detector_returns_empty_list_when_backend_returns_no_results():
    class EmptyModel:
        def predict(self, **_kwargs):
            return []

    detector = GraffitiDetector.__new__(GraffitiDetector)
    detector.conf_threshold = 0.25
    detector.device = None
    detector.model = EmptyModel()

    assert detector.predict("street.jpg") == []
