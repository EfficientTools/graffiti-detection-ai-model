"""
Data augmentation pipeline for graffiti detection.
Uses Albumentations library for efficient and flexible augmentation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional


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
            value=(114, 114, 114),
            p=1.0
        ),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            value=(114, 114, 114),
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
            A.GaussNoise(var_limit=noise_var_limit, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Weather and lighting conditions
        A.OneOf([
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=5,
                brightness_coefficient=0.7,
                rain_type=None,
                p=1.0
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=1.0
            ),
        ], p=0.2),
        
        # Image quality
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
        ], p=0.2),
        
        # Advanced augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
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
            value=(114, 114, 114),
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
            value=(114, 114, 114),
            p=1.0
        ),
    ])


# Preset configurations
AUGMENTATION_PRESETS = {
    'light': {
        'brightness_limit': 0.1,
        'contrast_limit': 0.1,
        'blur_limit': 3,
        'noise_var_limit': (5.0, 25.0)
    },
    'medium': {
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'blur_limit': 7,
        'noise_var_limit': (10.0, 50.0)
    },
    'heavy': {
        'brightness_limit': 0.3,
        'contrast_limit': 0.3,
        'blur_limit': 11,
        'noise_var_limit': (20.0, 80.0)
    }
}


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
    
    params = AUGMENTATION_PRESETS[preset]
    return get_training_augmentation(img_size=img_size, **params)
