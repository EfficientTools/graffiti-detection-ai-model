"""
Custom dataset class for graffiti detection.
Supports YOLO format annotations and integrates with PyTorch DataLoader.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class GraffitiDataset(Dataset):
    """
    Custom PyTorch Dataset for graffiti detection.
    
    Supports YOLO format annotations:
    - Each image has a corresponding .txt file
    - Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    
    Args:
        image_paths: List of paths to images
        label_paths: List of paths to label files (YOLO format)
        img_size: Target image size (width, height)
        augmentation: Albumentations augmentation pipeline
        preprocessing: Preprocessing function
        return_labels: Whether to return labels (False for inference)
    """
    
    def __init__(
        self,
        image_paths: List[str],
        label_paths: Optional[List[str]] = None,
        img_size: Tuple[int, int] = (640, 640),
        augmentation=None,
        preprocessing=None,
        return_labels: bool = True
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.return_labels = return_labels
        
        # Validate paths
        if return_labels and label_paths is None:
            raise ValueError("label_paths must be provided when return_labels=True")
        
        if return_labels and len(image_paths) != len(label_paths):
            raise ValueError("Number of images and labels must match")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - image: Tensor of shape (C, H, W)
                - labels: Tensor of shape (N, 5) where N is number of objects
                          Format: [class_id, x_center, y_center, width, height]
                - image_path: Path to original image
                - original_shape: Original image shape (H, W, C)
        """
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape
        
        # Load labels if required
        labels = None
        if self.return_labels:
            label_path = self.label_paths[idx]
            labels = self._load_yolo_labels(label_path)
        
        # Apply augmentation
        if self.augmentation is not None and labels is not None:
            # Convert labels for albumentations
            # From YOLO format (normalized) to pixel coordinates
            h, w = image.shape[:2]
            bboxes = []
            class_labels = []
            
            for label in labels:
                class_id, x_center, y_center, width, height = label
                # Convert to [x_min, y_min, x_max, y_max] in pixels
                x_min = (x_center - width / 2) * w
                y_min = (y_center - height / 2) * h
                x_max = (x_center + width / 2) * w
                y_max = (y_center + height / 2) * h
                
                bboxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(int(class_id))
            
            # Apply augmentation
            if len(bboxes) > 0:
                augmented = self.augmentation(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                image = augmented['image']
                bboxes = augmented['bboxes']
                class_labels = augmented['class_labels']
                
                # Convert back to YOLO format
                h, w = image.shape[:2]
                labels = []
                for bbox, cls in zip(bboxes, class_labels):
                    x_min, y_min, x_max, y_max = bbox
                    x_center = ((x_min + x_max) / 2) / w
                    y_center = ((y_min + y_max) / 2) / h
                    width = (x_max - x_min) / w
                    height = (y_max - y_min) / h
                    labels.append([cls, x_center, y_center, width, height])
                
                labels = np.array(labels) if labels else np.zeros((0, 5))
            else:
                labels = np.zeros((0, 5))
        
        # Resize image
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply preprocessing
        if self.preprocessing is not None:
            image = self.preprocessing(image)
        else:
            # Default preprocessing: normalize to [0, 1] and convert to CHW format
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Convert to tensors
        image = torch.from_numpy(image).float()
        
        result = {
            'image': image,
            'image_path': image_path,
            'original_shape': original_shape
        }
        
        if self.return_labels and labels is not None:
            if isinstance(labels, list):
                labels = np.array(labels)
            result['labels'] = torch.from_numpy(labels).float()
        
        return result
    
    def _load_yolo_labels(self, label_path: str) -> np.ndarray:
        """
        Load YOLO format labels from a text file.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Array of shape (N, 5) with format [class_id, x_center, y_center, width, height]
        """
        if not os.path.exists(label_path):
            # Return empty array if no labels
            return np.zeros((0, 5))
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                labels.append([class_id, x_center, y_center, width, height])
        
        return np.array(labels) if labels else np.zeros((0, 5))
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate function for batching samples with varying number of labels.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dictionary with stacked images and concatenated labels
        """
        images = torch.stack([item['image'] for item in batch])
        image_paths = [item['image_path'] for item in batch]
        original_shapes = [item['original_shape'] for item in batch]
        
        result = {
            'images': images,
            'image_paths': image_paths,
            'original_shapes': original_shapes
        }
        
        # Handle labels if present
        if 'labels' in batch[0]:
            # Add batch index to labels
            labels = []
            for i, item in enumerate(batch):
                item_labels = item['labels']
                if len(item_labels) > 0:
                    # Add batch index as first column
                    batch_idx = torch.full((len(item_labels), 1), i)
                    item_labels = torch.cat([batch_idx, item_labels], dim=1)
                    labels.append(item_labels)
            
            # Concatenate all labels
            if labels:
                result['labels'] = torch.cat(labels, dim=0)
            else:
                result['labels'] = torch.zeros((0, 6))  # Empty labels
        
        return result


def load_image_paths_from_file(txt_file: str, data_root: Optional[str] = None) -> List[str]:
    """
    Load image paths from a text file (one path per line).
    
    Args:
        txt_file: Path to text file containing image paths
        data_root: Root directory to prepend to relative paths
        
    Returns:
        List of absolute image paths
    """
    with open(txt_file, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    # Convert to absolute paths
    if data_root:
        paths = [os.path.join(data_root, p) if not os.path.isabs(p) else p for p in paths]
    
    return paths


def get_label_path_from_image_path(image_path: str, labels_dir: Optional[str] = None) -> str:
    """
    Get corresponding label path from image path.
    Assumes labels are in a parallel directory structure or specified labels_dir.
    
    Args:
        image_path: Path to image file
        labels_dir: Directory containing labels (if different from images)
        
    Returns:
        Path to label file
    """
    path = Path(image_path)
    
    if labels_dir:
        # Use specified labels directory
        label_path = Path(labels_dir) / f"{path.stem}.txt"
    else:
        # Replace 'images' with 'labels' in path and change extension
        label_path = Path(str(path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
        label_path = label_path.with_suffix('.txt')
    
    return str(label_path)


def create_dataloaders(
    train_txt: str,
    val_txt: str,
    data_root: str,
    batch_size: int = 16,
    img_size: Tuple[int, int] = (640, 640),
    num_workers: int = 4,
    train_augmentation=None,
    val_augmentation=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_txt: Path to training images list
        val_txt: Path to validation images list
        data_root: Root directory for data
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of worker processes
        train_augmentation: Augmentation pipeline for training
        val_augmentation: Augmentation pipeline for validation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load image paths
    train_images = load_image_paths_from_file(train_txt, data_root)
    val_images = load_image_paths_from_file(val_txt, data_root)
    
    # Get corresponding label paths
    train_labels = [get_label_path_from_image_path(img) for img in train_images]
    val_labels = [get_label_path_from_image_path(img) for img in val_images]
    
    # Create datasets
    train_dataset = GraffitiDataset(
        train_images,
        train_labels,
        img_size=img_size,
        augmentation=train_augmentation,
        return_labels=True
    )
    
    val_dataset = GraffitiDataset(
        val_images,
        val_labels,
        img_size=img_size,
        augmentation=val_augmentation,
        return_labels=True
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=GraffitiDataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=GraffitiDataset.collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
