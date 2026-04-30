"""
Data augmentation pipeline for graffiti detection.
Uses Albumentations library for efficient and flexible augmentation.
"""

import math
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
from typing import Optional


def _variance_to_std_range(noise_var_limit: tuple) -> tuple:
    """Convert old pixel variance limits to Albumentations 2.x std_range."""
    low, high = noise_var_limit
    return (
        min(max(math.sqrt(low) / 255.0, 0.0), 1.0),
        min(max(math.sqrt(high) / 255.0, 0.0), 1.0),
    )


def get_training_augmentation(
    img_size: int = 640,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    blur_limit: int = 7,
    noise_var_limit: tuple = (10.0, 50.0)
) -> A.Compose:
    """
    Get augmentation pipeline for training.
    
    Includes geometric and color augmentations suitable for object detection.
    All transformations preserve bounding boxes.
    
    Args:
        img_size: Target image size
        brightness_limit: Range for brightness adjustment
        contrast_limit: Range for contrast adjustment
        blur_limit: Maximum kernel size for blur
        noise_var_limit: Variance range for Gaussian noise
        
    Returns:
        Albumentations composition pipeline
    """
    train_transform = A.Compose([
        # Resize to target size
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=(114, 114, 114),
            p=1.0
        ),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.8, 1.2),
            rotate=(-15, 15),
            border_mode=0,
            fill=(114, 114, 114),
            p=0.5
        ),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        
        # Quality degradations (simulate real-world conditions)
        A.OneOf([
            A.Blur(blur_limit=blur_limit, p=1.0),
            A.GaussNoise(std_range=_variance_to_std_range(noise_var_limit), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Weather and lighting conditions
        A.OneOf([
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=5,
                brightness_coefficient=0.7,
                rain_type="default",
                p=1.0
            ),
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=1.0
            ),
        ], p=0.2),
        
        # Image quality
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.ImageCompression(quality_range=(75, 100), p=1.0),
        ], p=0.2),
        
        # Advanced augmentations
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.3
        ),
        
        # Normalize (commented out - done in dataset class)
        # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_area=0,
        min_visibility=0.1,
        label_fields=['class_labels']
    ))
    
    return train_transform


def get_validation_augmentation(img_size: int = 640) -> A.Compose:
    """
    Get augmentation pipeline for validation/inference.
    Only includes resizing and padding, no random augmentations.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations composition pipeline
    """
    val_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=(114, 114, 114),
            p=1.0
        ),
        # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_area=0,
        min_visibility=0,
        label_fields=['class_labels']
    ))
    
    return val_transform


def get_test_augmentation(img_size: int = 640) -> Optional[A.Compose]:
    """
    Get augmentation pipeline for test-time augmentation (TTA).
    Applies multiple augmentations and averages predictions.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations composition pipeline
    """
    # For now, same as validation
    # Can be extended with TTA strategies like multi-scale, flips, etc.
    return get_validation_augmentation(img_size)


def get_mosaic_augmentation(img_size: int = 640) -> A.Compose:
    """
    Get mosaic augmentation (combines 4 images into one).
    This is typically handled by YOLO's built-in augmentation.
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations composition pipeline
    """
    # Note: Mosaic is better handled by YOLO's native implementation
    # This is a placeholder for custom mosaic if needed
    return get_training_augmentation(img_size)


def get_inference_transform(img_size: int = 640) -> A.Compose:
    """
    Get transformation pipeline for inference (no augmentation).
    
    Args:
        img_size: Target image size
        
    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=(114, 114, 114),
            p=1.0
        ),
    ])


# Preset configurations
AUGMENTATION_PRESETS = {
    'light': {
        'hsv': 0.1,
        'blur': 3,
        'noise': (5.0, 25.0),
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
        'blur_limit': 3,
        'noise_var_limit': (5.0, 25.0)
    },
    'medium': {
        'hsv': 0.2,
        'blur': 7,
        'noise': (10.0, 50.0),
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'blur_limit': 7,
        'noise_var_limit': (10.0, 50.0)
    },
    'heavy': {
        'hsv': 0.3,
        'blur': 11,
        'noise': (20.0, 80.0),
        'brightness_limit': 0.3,
        'contrast_limit': 0.3,
        'blur_limit': 11,
        'noise_var_limit': (20.0, 80.0)
    },
    'street': {
        'hsv': 0.25,
        'blur': 9,
        'noise': (15.0, 60.0),
        'brightness_limit': 0.35,
        'contrast_limit': 0.35,
        'blur_limit': 9,
        'noise_var_limit': (15.0, 60.0)
    }
}


def get_street_scene_augmentation(img_size: int = 640) -> A.Compose:
    """
    Get augmentation pipeline optimised for public street scenarios.

    Covers typical urban conditions:
    - varying sunlight, shadows from buildings, street furniture occlusion
    - rain, fog, and mixed lighting (day/night)
    - wide-angle camera distortions and motion blur
    - graffiti viewed from different distances and angles

    Args:
        img_size: Target image size

    Returns:
        Albumentations composition pipeline
    """
    street_transform = A.Compose([
        # Resize
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=(114, 114, 114),
            p=1.0
        ),

        # Street-specific geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=(-0.15, 0.15),
            scale=(0.7, 1.3),
            rotate=(-10, 10),
            border_mode=0,
            fill=(114, 114, 114),
            p=0.6
        ),
        A.Perspective(scale=(0.05, 0.12), p=0.4),

        # Varying urban lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.35,
                contrast_limit=0.35,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=25,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),

        # Dynamic range / exposure simulation (night-time / strong sunlight)
        A.RandomToneCurve(scale=0.2, p=0.2),

        # Shadows cast by buildings and street furniture
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=6,
            p=0.4
        ),

        # Weather conditions common on streets
        A.OneOf([
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=5,
                brightness_coefficient=0.7,
                rain_type="default",
                p=1.0
            ),
            A.RandomFog(fog_coef_range=(0.15, 0.45), p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_range=(0.0, 1.0),
                num_flare_circles_range=(3, 6),
                src_radius=200,
                p=1.0
            ),
        ], p=0.35),

        # Camera quality degradations
        A.OneOf([
            A.MotionBlur(blur_limit=9, p=1.0),
            A.Blur(blur_limit=7, p=1.0),
            A.GaussNoise(std_range=_variance_to_std_range((15.0, 60.0)), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.6), p=1.0),
        ], p=0.35),

        # Compression / low-quality CCTV feed simulation
        A.ImageCompression(quality_range=(50, 95), p=0.3),

        # Partial occlusion (poles, fences, passing objects)
        A.CoarseDropout(
            num_holes_range=(1, 10),
            hole_height_range=(12, 48),
            hole_width_range=(12, 48),
            fill=0,
            p=0.35
        ),

    ], bbox_params=A.BboxParams(
        format='yolo',
        min_area=0,
        min_visibility=0.15,
        label_fields=['class_labels']
    ))

    return street_transform


def get_augmentation_by_preset(preset: str = 'medium', img_size: int = 640) -> A.Compose:
    """
    Get augmentation pipeline by preset name.
    
    Args:
        preset: Preset name ('light', 'medium', 'heavy')
        img_size: Target image size
        
    Returns:
        Albumentations composition pipeline
    """
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(AUGMENTATION_PRESETS.keys())}")

    # The 'street' preset uses a dedicated pipeline with urban-specific transforms
    if preset == 'street':
        return get_street_scene_augmentation(img_size=img_size)

    params = {
        k: v
        for k, v in AUGMENTATION_PRESETS[preset].items()
        if k in {"brightness_limit", "contrast_limit", "blur_limit", "noise_var_limit"}
    }
    return get_training_augmentation(img_size=img_size, **params)
