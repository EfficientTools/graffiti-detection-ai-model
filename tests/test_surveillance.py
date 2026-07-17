"""Tests for multi-camera surveillance configuration behavior."""

from unittest.mock import patch

from scripts.multi_camera_surveillance import MultiCameraSurveillance


@patch('scripts.multi_camera_surveillance.YOLO')
def test_surveillance_selects_only_enabled_cameras(mock_yolo):
    cameras = {
        'enabled': {'source': 0, 'enabled': True},
        'disabled': {'source': 1, 'enabled': False},
    }
    surveillance = MultiCameraSurveillance('model.pt', cameras)

    enabled_cameras = surveillance._enabled_cameras()

    assert [camera_id for camera_id, _ in enabled_cameras] == ['enabled']
