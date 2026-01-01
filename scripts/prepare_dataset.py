#!/usr/bin/env python
"""
Dataset preparation script for graffiti detection.
Organizes raw data into train/val/test splits with YOLO format.
"""

import argparse
from pathlib import Path
import shutil
import random
from typing import List, Tuple
import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare graffiti detection dataset')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to raw data directory containing images and labels'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for organized dataset'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        nargs=3,
        default=[0.8, 0.15, 0.05],
        help='Train/val/test split ratios (must sum to 1.0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate annotations before organizing'
    )
    
    return parser.parse_args()


def find_image_label_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find matching image and label file pairs.
    
    Args:
        data_dir: Directory containing images and labels
        
    Returns:
        List of (image_path, label_path) tuples
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f"**/*{ext}"))
        image_files.extend(data_dir.glob(f"**/*{ext.upper()}"))
    
    pairs = []
    missing_labels = []
    
    for img_path in image_files:
        # Look for corresponding label file
        # Try in same directory
        label_path = img_path.with_suffix('.txt')
        
        # Try in labels subdirectory
        if not label_path.exists():
            labels_dir = img_path.parent / 'labels'
            label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Try replacing 'images' with 'labels' in path
        if not label_path.exists():
            label_path = Path(str(img_path).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
            label_path = label_path.with_suffix('.txt')
        
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path)
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no corresponding labels:")
        for img_path in missing_labels[:5]:
            print(f"  {img_path}")
        if len(missing_labels) > 5:
            print(f"  ... and {len(missing_labels) - 5} more")
    
    return pairs


def validate_yolo_label(label_path: Path) -> Tuple[bool, str]:
    """
    Validate YOLO format label file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                return False, f"Line {line_num}: Expected at least 5 values, got {len(parts)}"
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Check ranges
                if not (0 <= x_center <= 1):
                    return False, f"Line {line_num}: x_center out of range [0, 1]: {x_center}"
                if not (0 <= y_center <= 1):
                    return False, f"Line {line_num}: y_center out of range [0, 1]: {y_center}"
                if not (0 <= width <= 1):
                    return False, f"Line {line_num}: width out of range [0, 1]: {width}"
                if not (0 <= height <= 1):
                    return False, f"Line {line_num}: height out of range [0, 1]: {height}"
                if class_id < 0:
                    return False, f"Line {line_num}: class_id must be >= 0: {class_id}"
                    
            except ValueError as e:
                return False, f"Line {line_num}: Invalid numeric value: {e}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error reading file: {e}"


def organize_dataset(
    pairs: List[Tuple[Path, Path]],
    output_dir: Path,
    split: List[float],
    seed: int,
    copy_files: bool = False
) -> None:
    """
    Organize dataset into train/val/test splits.
    
    Args:
        pairs: List of (image_path, label_path) tuples
        output_dir: Output directory
        split: Train/val/test split ratios
        seed: Random seed
        copy_files: Copy files instead of moving
    """
    random.seed(seed)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_ratio, val_ratio, test_ratio = split
    
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} samples ({len(train_pairs)/n*100:.1f}%)")
    print(f"  Validation: {len(val_pairs)} samples ({len(val_pairs)/n*100:.1f}%)")
    print(f"  Test: {len(test_pairs)} samples ({len(test_pairs)/n*100:.1f}%)")
    
    # Create directory structure
    for split_name in ['train', 'val', 'test']:
        (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    file_op = shutil.copy2 if copy_files else shutil.move
    
    for split_name, split_pairs in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create list files
        img_list_path = output_dir / f"{split_name}.txt"
        
        with open(img_list_path, 'w') as f:
            for img_path, label_path in split_pairs:
                # Destination paths
                img_dest = output_dir / 'images' / split_name / img_path.name
                label_dest = output_dir / 'labels' / split_name / label_path.name
                
                # Copy/move files
                file_op(str(img_path), str(img_dest))
                file_op(str(label_path), str(label_dest))
                
                # Write relative path to list file
                f.write(f"./images/{split_name}/{img_path.name}\n")
        
        print(f"  Created {img_list_path}")


def create_dataset_config(output_dir: Path, num_classes: int = 1) -> None:
    """
    Create dataset.yaml configuration file.
    
    Args:
        output_dir: Output directory
        num_classes: Number of classes
    """
    config = {
        'path': str(output_dir.absolute()),
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt',
        'nc': num_classes,
        'names': {0: 'graffiti'} if num_classes == 1 else {i: f'class_{i}' for i in range(num_classes)}
    }
    
    config_path = output_dir / 'dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nCreated dataset configuration: {config_path}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(args.split)}")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Preparing dataset from: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={args.split[0]:.2f}, val={args.split[1]:.2f}, test={args.split[2]:.2f}")
    
    # Find image-label pairs
    print("\nSearching for image-label pairs...")
    pairs = find_image_label_pairs(data_dir)
    
    if not pairs:
        raise ValueError("No valid image-label pairs found!")
    
    print(f"Found {len(pairs)} valid image-label pairs")
    
    # Validate labels if requested
    if args.validate:
        print("\nValidating labels...")
        invalid_labels = []
        
        for img_path, label_path in pairs:
            is_valid, error_msg = validate_yolo_label(label_path)
            if not is_valid:
                invalid_labels.append((label_path, error_msg))
        
        if invalid_labels:
            print(f"\nWarning: {len(invalid_labels)} invalid label files found:")
            for label_path, error_msg in invalid_labels[:5]:
                print(f"  {label_path}: {error_msg}")
            if len(invalid_labels) > 5:
                print(f"  ... and {len(invalid_labels) - 5} more")
            
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        else:
            print("All labels are valid!")
    
    # Organize dataset
    print("\nOrganizing dataset...")
    organize_dataset(pairs, output_dir, args.split, args.seed, args.copy)
    
    # Create dataset config
    create_dataset_config(output_dir)
    
    # Create .gitkeep files for empty directories
    for subdir in ['raw', 'images', 'labels']:
        gitkeep = output_dir / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()
    
    print("\n" + "="*50)
    print("Dataset preparation completed!")
    print("="*50)
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/")
    print(f"    │   ├── val/")
    print(f"    │   └── test/")
    print(f"    ├── labels/")
    print(f"    │   ├── train/")
    print(f"    │   ├── val/")
    print(f"    │   └── test/")
    print(f"    ├── train.txt")
    print(f"    ├── val.txt")
    print(f"    ├── test.txt")
    print(f"    └── dataset.yaml")
    print("\nYou can now start training with:")
    print(f"  python scripts/train.py --data {output_dir}/dataset.yaml")


if __name__ == '__main__':
    main()
