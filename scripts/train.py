#!/usr/bin/env python
"""
Training script for graffiti detection model using YOLOv8.
"""

import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train graffiti detection model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='YOLOv8 model variant'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=None,
        help='Input image size (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (0, 1, ..., or cpu)'
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained weights'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='graffiti-detection',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Allow overwriting existing project/name'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker threads'
    )
    
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images for faster training'
    )
    
    return parser.parse_args()


def load_training_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load training configuration
    print(f"Loading training configuration from {args.config}")
    config = load_training_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch'] = args.batch_size
    if args.img_size is not None:
        config['imgsz'] = args.img_size
    if args.device is not None:
        config['device'] = args.device
    if args.workers is not None:
        config['workers'] = args.workers
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available, using CPU")
        config['device'] = 'cpu'
    
    # Initialize model
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}")
        model = YOLO(args.pretrained)
    elif args.resume:
        print("Resuming training from last checkpoint")
        # Find last checkpoint
        last_checkpoint = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            model = YOLO(str(last_checkpoint))
        else:
            print(f"No checkpoint found at {last_checkpoint}, starting from scratch")
            model = YOLO(f'{args.model}.pt')
    else:
        print(f"Loading YOLOv8 {args.model} model")
        model = YOLO(f'{args.model}.pt')
    
    # Print training configuration
    print("\n" + "="*50)
    print("Training Configuration:")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch', 16)}")
    print(f"Image size: {config.get('imgsz', 640)}")
    print(f"Device: {config.get('device', 0)}")
    print(f"Workers: {config.get('workers', 8)}")
    print(f"Optimizer: {config.get('optimizer', 'AdamW')}")
    print(f"Learning rate: {config.get('lr0', 0.001)}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("="*50 + "\n")
    
    # Start training
    print("Starting training...")
    results = model.train(
        data=args.data,
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        device=config.get('device', 0),
        workers=config.get('workers', 8),
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.001),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        hsv_h=config.get('hsv_h', 0.015),
        hsv_s=config.get('hsv_s', 0.7),
        hsv_v=config.get('hsv_v', 0.4),
        degrees=config.get('degrees', 0.0),
        translate=config.get('translate', 0.1),
        scale=config.get('scale', 0.5),
        shear=config.get('shear', 0.0),
        perspective=config.get('perspective', 0.0),
        flipud=config.get('flipud', 0.0),
        fliplr=config.get('fliplr', 0.5),
        mosaic=config.get('mosaic', 1.0),
        mixup=config.get('mixup', 0.0),
        copy_paste=config.get('copy_paste', 0.0),
        conf=config.get('conf', 0.25),
        iou=config.get('iou', 0.7),
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=config.get('pretrained', True),
        verbose=config.get('verbose', True),
        save=config.get('save', True),
        save_period=config.get('save_period', -1),
        cache=args.cache,
        val=config.get('val', True),
        patience=config.get('patience', 50),
        plots=config.get('plots', True),
        amp=config.get('amp', True),
        resume=args.resume
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print(f"Last checkpoint saved to: {Path(args.project) / args.name / 'weights' / 'last.pt'}")
    print(f"Results saved to: {Path(args.project) / args.name}")
    print("="*50 + "\n")
    
    # Print final metrics
    if results:
        print("Final Metrics:")
        print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 0):.4f}")


if __name__ == '__main__':
    main()
