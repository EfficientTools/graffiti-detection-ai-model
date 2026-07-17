"""Repository-level integration checks."""

import json
from pathlib import Path

import yaml


def test_runtime_yaml_configs_are_valid():
    for config_path in (
        Path('configs/dataset.yaml'),
        Path('configs/training.yaml'),
        Path('configs/street_scenarios.yaml'),
    ):
        with config_path.open(encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
        assert isinstance(config, dict)
        assert config


def test_alert_example_has_supported_channels():
    config = json.loads(Path('configs/alerts_example.json').read_text())

    assert set(config) == {
        'email',
        'sms',
        'webhook',
        'discord',
        'slack',
        'push_notification',
    }
    assert all('enabled' in channel for channel in config.values())


def test_camera_example_has_required_fields():
    config = json.loads(Path('configs/cameras_example.json').read_text())

    assert config
    for camera in config.values():
        assert {'source', 'conf_threshold', 'enabled'} <= camera.keys()


def test_runtime_directories_are_present():
    required_directories = (
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test',
        'models',
        'outputs/crops',
        'outputs/logs',
        'outputs/metrics',
        'outputs/predictions',
        'outputs/visualizations',
    )

    assert all(Path(directory).is_dir() for directory in required_directories)


def test_documented_scripts_are_present():
    scripts = (
        'train.py',
        'evaluate.py',
        'inference.py',
        'prepare_dataset.py',
        'multi_camera_surveillance.py',
        'real_time_dashboard.py',
        'incident_logger.py',
    )

    assert all((Path('scripts') / script).is_file() for script in scripts)
