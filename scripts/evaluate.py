#!/usr/bin/env python
"""
Evaluation script for graffiti detection model.
"""

import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate graffiti detection model')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='configs/dataset.yaml',
        help='Path to dataset configuration file'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Dataset split to evaluate on'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for predictions'
    )
    
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.7,
        help='IoU threshold for NMS'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (0, 1, ..., or cpu)'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results in COCO JSON format'
    )
    
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save predictions in YOLO format'
    )
    
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='Save confidence scores in txt files'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default='runs/evaluate',
        help='Project directory'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Allow overwriting existing project/name'
    )
    
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check if model file exists
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")
        args.device = 'cpu'
    
    # Load model
    print(f"Loading model from {args.model}")
    model = YOLO(args.model)
    
    # Print evaluation configuration
    print("\n" + "="*50)
    print("Evaluation Configuration:")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Device: {args.device}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("="*50 + "\n")
    
    # Run validation
    print(f"Evaluating on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch_size,
        imgsz=args.img_size,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        device=args.device,
        save_json=args.save_json,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=args.plots,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    
    metrics = results.results_dict
    
    # Detection metrics
    if 'metrics/mAP50(B)' in metrics:
        print(f"mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in metrics:
        print(f"mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
    if 'metrics/precision(B)' in metrics:
        print(f"Precision: {metrics['metrics/precision(B)']:.4f}")
    if 'metrics/recall(B)' in metrics:
        print(f"Recall: {metrics['metrics/recall(B)']:.4f}")
    
    # Calculate F1 score
    if 'metrics/precision(B)' in metrics and 'metrics/recall(B)' in metrics:
        precision = metrics['metrics/precision(B)']
        recall = metrics['metrics/recall(B)']
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            print(f"F1-Score: {f1_score:.4f}")
    
    print("="*50 + "\n")
    
    # Save metrics to JSON
    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to: {metrics_file}")
    print(f"Results saved to: {output_dir}")
    
    # Per-class metrics if available
    if hasattr(results, 'ap_class_index'):
        print("\nPer-class metrics:")
        print("-" * 50)
        for i, class_idx in enumerate(results.ap_class_index):
            class_name = results.names[class_idx]
            ap50 = results.ap50[i] if hasattr(results, 'ap50') else 0
            ap = results.ap[i] if hasattr(results, 'ap') else 0
            print(f"Class '{class_name}':")
            print(f"  AP@0.5: {ap50:.4f}")
            print(f"  AP@0.5:0.95: {ap:.4f}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
