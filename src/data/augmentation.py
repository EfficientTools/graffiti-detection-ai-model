"""Compatibility imports for the public augmentation module."""

from graffiti_detection.data.augmentation import (
    AUGMENTATION_PRESETS,
    get_augmentation_by_preset,
    get_inference_transform,
    get_mosaic_augmentation,
    get_street_scene_augmentation,
    get_test_augmentation,
    get_training_augmentation,
    get_validation_augmentation,
)

__all__ = [
    "AUGMENTATION_PRESETS",
    "get_augmentation_by_preset",
    "get_inference_transform",
    "get_mosaic_augmentation",
    "get_street_scene_augmentation",
    "get_test_augmentation",
    "get_training_augmentation",
    "get_validation_augmentation",
]
