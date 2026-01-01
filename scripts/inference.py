#!/usr/bin/env python
"""
Inference script for graffiti detection model.
Supports single images, batch images, videos, and webcam.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.visualization import draw_boxes, save_detection_crops


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on graffiti detection model')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Input source: image file, directory, video file, or webcam (0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions',
        help='Output directory for predictions'
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
        default=0.45,
        help='IoU threshold for NMS'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (0, 1, ..., or cpu)'
    )
    
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=None,
        help='Filter by class IDs'
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
        '--save-crops',
        action='store_true',
        help='Save cropped detections'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in a window'
    )
    
    parser.add_argument(
        '--line-thickness',
        type=int,
        default=2,
        help='Bounding box line thickness'
    )
    
    parser.add_argument(
        '--hide-labels',
        action='store_true',
        help='Hide labels in visualization'
    )
    
    parser.add_argument(
        '--hide-conf',
        action='store_true',
        help='Hide confidence scores in visualization'
    )
    
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use FP16 half-precision inference'
    )
    
    parser.add_argument(
        '--agnostic-nms',
        action='store_true',
        help='Class-agnostic NMS'
    )
    
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='Maximum detections per image'
    )
    
    parser.add_argument(
        '--vid-stride',
        type=int,
        default=1,
        help='Video frame-rate stride'
    )
    
    return parser.parse_args()


def process_image(model, image_path, args, output_dir):
    """Process a single image."""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Run inference
    results = model.predict(
        source=image_path,
        imgsz=args.img_size,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        device=args.device,
        classes=args.classes,
        half=args.half,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det,
        verbose=False
    )[0]
    
    # Get predictions
    boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)
    
    # Get class names
    class_names = {i: name for i, name in enumerate(results.names.values())}
    
    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    if len(boxes) > 0:
        image_rgb = draw_boxes(
            image_rgb,
            boxes,
            labels.tolist(),
            scores,
            class_names,
            thickness=args.line_thickness
        )
    
    # Save image
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Save txt predictions
    if args.save_txt:
        txt_dir = output_dir / 'labels'
        txt_dir.mkdir(exist_ok=True)
        txt_path = txt_dir / f"{image_path.stem}.txt"
        
        with open(txt_path, 'w') as f:
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                # Convert to YOLO format (normalized)
                h, w = image.shape[:2]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                if args.save_conf:
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.4f}\n")
                else:
                    f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Save crops
    if args.save_crops and len(boxes) > 0:
        crops_dir = output_dir / 'crops'
        save_detection_crops(
            image_rgb,
            boxes,
            labels.tolist(),
            scores,
            output_dir=str(crops_dir),
            image_name=image_path.stem
        )
    
    # Display
    if args.show:
        cv2.imshow('Graffiti Detection', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    
    return len(boxes)


def main():
    """Main inference function."""
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
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"IoU threshold: {args.iou_threshold}\n")
    
    source_path = Path(args.source)
    
    # Check source type
    if args.source == '0' or args.source.isdigit():
        # Webcam
        print("Running inference on webcam...")
        results = model.predict(
            source=int(args.source),
            imgsz=args.img_size,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            device=args.device,
            classes=args.classes,
            half=args.half,
            agnostic_nms=args.agnostic_nms,
            max_det=args.max_det,
            show=True,
            stream=True
        )
        
        for r in results:
            pass  # Real-time display handled by show=True
            
    elif source_path.is_file():
        # Check if video or image
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        if source_path.suffix.lower() in video_extensions:
            # Video file
            print(f"Running inference on video: {source_path}")
            results = model.predict(
                source=str(source_path),
                imgsz=args.img_size,
                conf=args.conf_threshold,
                iou=args.iou_threshold,
                device=args.device,
                classes=args.classes,
                half=args.half,
                agnostic_nms=args.agnostic_nms,
                max_det=args.max_det,
                vid_stride=args.vid_stride,
                stream=True,
                save=True,
                project=str(output_dir.parent),
                name=output_dir.name
            )
            
            for r in results:
                pass
            
            print(f"Video saved to: {output_dir}")
            
        elif source_path.suffix.lower() in image_extensions:
            # Single image
            print(f"Running inference on image: {source_path}")
            num_detections = process_image(model, source_path, args, output_dir)
            print(f"Detections: {num_detections}")
            print(f"Results saved to: {output_dir / source_path.name}")
        else:
            raise ValueError(f"Unsupported file format: {source_path.suffix}")
    
    elif source_path.is_dir():
        # Directory of images
        print(f"Running inference on directory: {source_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f"*{ext}"))
            image_files.extend(source_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No images found in {source_path}")
        
        print(f"Found {len(image_files)} images")
        
        total_detections = 0
        for image_path in tqdm(image_files, desc="Processing images"):
            num_detections = process_image(model, image_path, args, output_dir)
            total_detections += num_detections
        
        print(f"\nProcessed {len(image_files)} images")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(image_files):.2f}")
        print(f"Results saved to: {output_dir}")
    
    else:
        raise ValueError(f"Invalid source: {args.source}")
    
    if args.show:
        cv2.destroyAllWindows()
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
