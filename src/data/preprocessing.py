"""
Preprocessing utilities for graffiti detection.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    
    Args:
        img: Input image
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scale_fill: Stretch to new_shape
        scaleup: Allow scaling up
        stride: Stride multiple constraint
        
    Returns:
        Tuple of (resized image, ratio, padding)
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


def normalize_image(img: np.ndarray, mean: Optional[Tuple] = None, std: Optional[Tuple] = None) -> np.ndarray:
    """
    Normalize image to [0, 1] or using mean and std.
    
    Args:
        img: Input image (H, W, C) in range [0, 255]
        mean: Mean values for normalization
        std: Std values for normalization
        
    Returns:
        Normalized image
    """
    img = img.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
    
    return img


def denormalize_image(img: np.ndarray, mean: Optional[Tuple] = None, std: Optional[Tuple] = None) -> np.ndarray:
    """
    Denormalize image back to [0, 255] range.
    
    Args:
        img: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image in [0, 255]
    """
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        img = img * std + mean
    
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def preprocess_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for inference.
    
    Args:
        img: Input image (H, W, C) in BGR format
        target_size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Tuple of (preprocessed image, metadata dict)
    """
    original_shape = img.shape
    
    # Convert to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Letterbox resize
    img, ratio, padding = letterbox(img, new_shape=target_size, auto=False)
    
    # Normalize
    if normalize:
        img = img.astype(np.float32) / 255.0
    
    # HWC to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    metadata = {
        'original_shape': original_shape,
        'ratio': ratio,
        'padding': padding
    }
    
    return img, metadata


def postprocess_boxes(
    boxes: np.ndarray,
    original_shape: Tuple[int, int],
    ratio: Tuple[float, float],
    padding: Tuple[float, float]
) -> np.ndarray:
    """
    Convert boxes back to original image coordinates.
    
    Args:
        boxes: Predicted boxes in format [x1, y1, x2, y2]
        original_shape: Original image shape (H, W)
        ratio: Scaling ratio (w_ratio, h_ratio)
        padding: Padding (dw, dh)
        
    Returns:
        Boxes in original image coordinates
    """
    boxes = boxes.copy()
    
    # Remove padding
    boxes[:, [0, 2]] -= padding[0]  # x padding
    boxes[:, [1, 3]] -= padding[1]  # y padding
    
    # Scale to original size
    boxes[:, [0, 2]] /= ratio[0]
    boxes[:, [1, 3]] /= ratio[1]
    
    # Clip to image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_shape[0])
    
    return boxes


def resize_image(img: np.ndarray, size: Tuple[int, int], keep_ratio: bool = True) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        img: Input image
        size: Target size (width, height)
        keep_ratio: Keep aspect ratio
        
    Returns:
        Resized image
    """
    if keep_ratio:
        img, _, _ = letterbox(img, new_shape=size[::-1])
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    
    return img


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        img: Input image (H, W, C)
        
    Returns:
        Grayscale image (H, W)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img: Input image
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Image with enhanced contrast
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    
    if len(img.shape) == 3:
        # Apply to each channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        img = clahe.apply(img)
    
    return img
