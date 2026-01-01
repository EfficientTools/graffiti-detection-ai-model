"""
Visualization utilities for graffiti detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import random


# Color palette for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
]


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[int]] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (H, W, C) in RGB format
        boxes: Bounding boxes in format [x1, y1, x2, y2]
        labels: Class labels for each box
        scores: Confidence scores for each box
        class_names: Dictionary mapping class IDs to names
        thickness: Line thickness for boxes
        font_scale: Font scale for text
        
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Get class label and score
        label = labels[i] if labels is not None else 0
        score = scores[i] if scores is not None else None
        
        # Get color for this class
        color = COLORS[label % len(COLORS)]
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        if class_names and label in class_names:
            class_name = class_names[label]
        else:
            class_name = f"Class {label}"
        
        if score is not None:
            text = f"{class_name}: {score:.2f}"
        else:
            text = class_name
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    return img


def visualize_detection(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[int]] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Visualize detection results using matplotlib.
    
    Args:
        image: Input image (H, W, C) in RGB format
        boxes: Bounding boxes
        labels: Class labels
        scores: Confidence scores
        class_names: Dictionary mapping class IDs to names
        save_path: Path to save the visualization
        show: Whether to display the plot
        figsize: Figure size
    """
    img_with_boxes = draw_boxes(image, boxes, labels, scores, class_names)
    
    plt.figure(figsize=figsize)
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.title('Graffiti Detection Results')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, len(history), figsize=figsize)
    
    if len(history) == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved training history to {save_path}")
    
    plt.show()


def visualize_dataset_samples(
    images: List[np.ndarray],
    boxes_list: List[np.ndarray],
    labels_list: Optional[List[List[int]]] = None,
    class_names: Optional[Dict[int, str]] = None,
    n_samples: int = 6,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize multiple dataset samples in a grid.
    
    Args:
        images: List of images
        boxes_list: List of bounding boxes for each image
        labels_list: List of labels for each image
        class_names: Dictionary mapping class IDs to names
        n_samples: Number of samples to display
        save_path: Path to save the visualization
        figsize: Figure size
    """
    n_samples = min(n_samples, len(images))
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx in range(n_samples):
        img = images[idx]
        boxes = boxes_list[idx]
        labels = labels_list[idx] if labels_list else None
        
        img_with_boxes = draw_boxes(img, boxes, labels, class_names=class_names)
        
        axes[idx].imshow(img_with_boxes)
        axes[idx].axis('off')
        axes[idx].set_title(f'Sample {idx + 1}')
    
    # Hide extra subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved dataset visualization to {save_path}")
    
    plt.show()


def draw_yolo_labels(
    image: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Draw YOLO format labels (normalized coordinates) on image.
    
    Args:
        image: Input image (H, W, C)
        labels: YOLO format labels [class_id, x_center, y_center, width, height]
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    boxes = []
    label_ids = []
    
    for label in labels:
        class_id, x_center, y_center, width, height = label
        
        # Convert from YOLO format to pixel coordinates
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        boxes.append([x1, y1, x2, y2])
        label_ids.append(int(class_id))
    
    if boxes:
        boxes = np.array(boxes)
        return draw_boxes(img, boxes, label_ids, class_names=class_names)
    
    return img


def create_mosaic_visualization(
    images: List[np.ndarray],
    n_cols: int = 4,
    border: int = 10,
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create a mosaic visualization of multiple images.
    
    Args:
        images: List of images (all same size)
        n_cols: Number of columns in mosaic
        border: Border width between images
        border_color: Border color
        
    Returns:
        Mosaic image
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Get image size (assume all images are same size)
    h, w = images[0].shape[:2]
    
    # Create mosaic canvas
    mosaic_h = n_rows * h + (n_rows + 1) * border
    mosaic_w = n_cols * w + (n_cols + 1) * border
    mosaic = np.full((mosaic_h, mosaic_w, 3), border_color, dtype=np.uint8)
    
    # Place images
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        y = row * h + (row + 1) * border
        x = col * w + (col + 1) * border
        
        mosaic[y:y+h, x:x+w] = img
    
    return mosaic


def save_detection_crops(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[List[int]] = None,
    scores: Optional[np.ndarray] = None,
    output_dir: str = 'outputs/crops',
    image_name: str = 'image'
) -> List[str]:
    """
    Save cropped detections to individual files.
    
    Args:
        image: Input image (H, W, C) in RGB format
        boxes: Bounding boxes
        labels: Class labels
        scores: Confidence scores
        output_dir: Output directory for crops
        image_name: Base name for saved crops
        
    Returns:
        List of paths to saved crops
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Crop image
        crop = image[y1:y2, x1:x2]
        
        # Generate filename
        label = labels[i] if labels is not None else 0
        score = scores[i] if scores is not None else 0
        filename = f"{image_name}_crop_{i}_class{label}_conf{score:.2f}.jpg"
        save_path = Path(output_dir) / filename
        
        # Convert RGB to BGR for OpenCV
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), crop_bgr)
        
        saved_paths.append(str(save_path))
    
    return saved_paths


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize the matrix
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def annotate_video(
    video_path: str,
    output_path: str,
    detection_fn,
    class_names: Optional[Dict[int, str]] = None,
    conf_threshold: float = 0.25,
    skip_frames: int = 0
) -> None:
    """
    Annotate video with detections.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        detection_fn: Function that takes image and returns (boxes, labels, scores)
        class_names: Dictionary mapping class IDs to names
        conf_threshold: Confidence threshold for detections
        skip_frames: Number of frames to skip (for faster processing)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames if specified
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            frame_idx += 1
            out.write(frame)
            continue
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        boxes, labels, scores = detection_fn(frame_rgb)
        
        # Filter by confidence
        if len(scores) > 0:
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            labels = [l for l, m in zip(labels, mask) if m]
            scores = scores[mask]
        
        # Draw boxes
        if len(boxes) > 0:
            frame_rgb = draw_boxes(frame_rgb, boxes, labels, scores, class_names)
        
        # Convert back to BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(frame_bgr)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    print(f"Saved annotated video to {output_path}")
