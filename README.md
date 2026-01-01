# Graffiti Detection AI Model

An AI-powered object detection system using YOLOv8 to identify and locate graffiti across various contexts including walls, buildings, over-bridges, vehicles, and other surfaces.

## Overview

This project implements a deep learning model to automatically detect graffiti in images, helping authorities and property managers identify and address vandalism efficiently. The model can detect different types of graffiti across multiple surface contexts.

## Features

- **Multi-context Detection**: Detects graffiti on walls, buildings, bridges, cars, trains, and other surfaces
- **Real-time Inference**: Fast detection using optimized YOLOv8 architecture
- **Comprehensive Metrics**: Evaluation with mAP, precision, recall, and F1-score
- **Data Augmentation**: Robust training with extensive augmentation techniques
- **Visualization Tools**: Annotated predictions with bounding boxes and confidence scores
- **Flexible Architecture**: Support for YOLOv8n/s/m/l/x variants based on speed vs accuracy needs

## Project Structure

```
graffiti-detection-ai-model/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset directory
│   ├── raw/                           # Raw images
│   ├── images/                        # Processed images
│   ├── labels/                        # YOLO format annotations
│   ├── train.txt                      # Training set paths
│   ├── val.txt                        # Validation set paths
│   └── test.txt                       # Test set paths
│
├── src/                               # Source code
│   ├── models/                        # Model architectures
│   ├── data/                          # Data processing
│   ├── training/                      # Training utilities
│   ├── evaluation/                    # Evaluation utilities
│   └── utils/                         # General utilities
│
├── configs/                           # Configuration files
│   ├── dataset.yaml                   # Dataset configuration
│   ├── model.yaml                     # Model architecture config
│   └── training.yaml                  # Training hyperparameters
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_visualization.ipynb         # Annotation visualization
│   └── 03_model_testing.ipynb         # Model inference testing
│
├── scripts/                           # Training/inference scripts
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── inference.py                   # Single image inference
│   └── prepare_dataset.py             # Dataset preparation
│
├── models/                            # Saved models
│   └── checkpoints/                   # Training checkpoints
│
└── outputs/                           # Training outputs
    ├── logs/                          # Training logs
    ├── visualizations/                # Prediction visualizations
    └── metrics/                       # Evaluation results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ free disk space

### Setup

1. Clone the repository:
```bash
cd /Users/pierre/Code/graffiti-detection-ai-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Data Collection

Collect images containing graffiti across various contexts:
- Urban walls and buildings
- Over-bridges and underpasses
- Vehicles (cars, trains, buses)
- Public infrastructure
- Mixed scenes

Aim for at least 1500+ diverse images for robust training.

### Annotation

Use annotation tools to label graffiti instances:
- **LabelImg**: Desktop tool for bounding box annotation
- **CVAT**: Web-based collaborative annotation
- **Roboflow**: Cloud-based annotation with auto-assist

Export annotations in **YOLO format**:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Prepare Dataset

Organize your dataset and create train/val/test splits:

```bash
python scripts/prepare_dataset.py --data_dir data/raw --split 0.8 0.15 0.05
```

## Training

### Quick Start

Train a YOLOv8 nano model (fastest):
```bash
python scripts/train.py --config configs/training.yaml --model yolov8n
```

### Advanced Training

Train with custom settings:
```bash
python scripts/train.py \
    --config configs/training.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

### Training Parameters

- `--model`: YOLOv8 variant (n/s/m/l/x) - larger = more accurate but slower
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--img-size`: Input image size (default: 640)
- `--device`: GPU device (0, 1, etc.) or 'cpu'

## Evaluation

Evaluate model performance on test set:

```bash
python scripts/evaluate.py --model models/best.pt --data configs/dataset.yaml
```

### Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: mAP averaged over IoU thresholds 0.5 to 0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Inference

### Single Image Prediction

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source path/to/image.jpg \
    --output outputs/predictions/ \
    --conf-threshold 0.25
```

### Batch Processing

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source path/to/images/ \
    --output outputs/predictions/ \
    --save-crops
```

### Real-time Webcam

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source 0 \
    --show
```

## Configuration

### Dataset Configuration (`configs/dataset.yaml`)

Defines dataset paths and class names:
```yaml
path: ./data
train: train.txt
val: val.txt
test: test.txt

names:
  0: graffiti
```

### Model Configuration (`configs/model.yaml`)

Architecture and model-specific settings for custom modifications.

### Training Configuration (`configs/training.yaml`)

Hyperparameters including learning rate, optimizer, augmentation settings, etc.

## Class Definitions

The model supports different classification approaches:

### Approach 1: Single Class (Simplest)
- `graffiti`: Any graffiti regardless of type or context

### Approach 2: Type-based Classes
- `tag`: Simple signatures
- `throw-up`: Bubble letters
- `piece`: Complex artistic graffiti
- `stencil`: Stenciled graffiti

### Approach 3: Context-based Classes
- `wall-graffiti`: Graffiti on walls
- `building-graffiti`: Graffiti on building facades
- `bridge-graffiti`: Graffiti on bridges/overpasses
- `vehicle-graffiti`: Graffiti on cars, trains, etc.

Choose the approach that best fits your use case and update `configs/dataset.yaml` accordingly.

## Model Export

### ONNX Format (for deployment)
```bash
yolo export model=models/best.pt format=onnx
```

### TensorRT (for NVIDIA GPUs)
```bash
yolo export model=models/best.pt format=engine device=0
```

### CoreML (for iOS/macOS)
```bash
yolo export model=models/best.pt format=coreml
```

## Performance Optimization

### For Speed
- Use YOLOv8n (nano) variant
- Reduce input image size (e.g., 320x320)
- Export to TensorRT for GPU inference
- Use half-precision (FP16) inference

### For Accuracy
- Use YOLOv8l or YOLOv8x variants
- Increase input image size (e.g., 1280x1280)
- Train for more epochs
- Use larger dataset with diverse examples

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Use smaller model variant
- Reduce image size
- Enable gradient accumulation

### Poor Detection Results
- Check annotation quality
- Increase dataset size and diversity
- Train for more epochs
- Adjust confidence threshold
- Review class balance

### Slow Training
- Ensure GPU is being used (`--device 0`)
- Increase batch size (if memory allows)
- Use mixed precision training
- Enable multi-GPU training

## Contributing

Contributions are welcome! Areas for improvement:
- Dataset expansion with diverse graffiti examples
- Multi-class detection for graffiti types
- Segmentation model for pixel-level detection
- Mobile deployment optimization
- Real-time video stream processing

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Ultralytics YOLOv8**: State-of-the-art object detection framework
- **PyTorch**: Deep learning framework
- **Albumentations**: Image augmentation library

## Citation

If you use this project in your research, please cite:

```bibtex
@software{graffiti_detection_2026,
  title={Graffiti Detection AI Model},
  author={Pierre},
  year={2026},
  url={https://github.com/yourusername/graffiti-detection-ai-model}
}
```

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Note**: This model is designed to assist in identifying graffiti for maintenance and urban management purposes. Always respect local laws and privacy regulations when deploying computer vision systems in public spaces.
