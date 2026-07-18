"""API contract tests that do not require model weights."""

from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import graffiti_detection.api as graffiti_detector


def make_image_bytes() -> bytes:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


def make_model():
    box = Mock()
    box.conf = 0.9
    box.xyxy = [np.array([1.0, 2.0, 10.0, 12.0])]
    result = Mock()
    result.boxes = [box]
    result.plot.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    model = Mock(return_value=[result])
    return model


def api_client():
    model = make_model()
    yolo = patch.object(graffiti_detector, "YOLO", return_value=model)
    return yolo, TestClient(graffiti_detector.app)


def test_health_endpoint_reports_package_version():
    yolo, client = api_client()
    with yolo, client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == graffiti_detector.__version__


def test_detect_rejects_invalid_images_as_client_errors():
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect",
            files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_detect_rejects_non_image_media_types():
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect",
            files={"file": ("image.txt", make_image_bytes(), "text/plain")},
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported image type"


def test_detect_rejects_oversized_uploads(monkeypatch):
    monkeypatch.setattr(graffiti_detector, "MAX_UPLOAD_BYTES", 8)
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect",
            files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Image file is too large"


def test_detect_validates_confidence_threshold():
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect?conf_threshold=1.1",
            files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")},
        )

    assert response.status_code == 422


def test_detect_returns_normalized_response_contract():
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect",
            files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["detections"] == 1
    assert response.json()["bounding_boxes"] == [[1.0, 2.0, 10.0, 12.0]]
    assert response.json()["alert_triggered"] is True


def test_batch_rejects_too_many_images(monkeypatch):
    monkeypatch.setattr(graffiti_detector, "MAX_BATCH_SIZE", 1)
    image = make_image_bytes()
    yolo, client = api_client()
    with yolo, client:
        response = client.post(
            "/detect/batch",
            files=[
                ("files", ("first.jpg", image, "image/jpeg")),
                ("files", ("second.jpg", image, "image/jpeg")),
            ],
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Batch contains more than 1 images"


def test_detect_hides_internal_inference_errors():
    yolo, client = api_client()
    with yolo, client:
        graffiti_detector.model.side_effect = RuntimeError("private implementation detail")
        response = client.post(
            "/detect",
            files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Detection failed"
