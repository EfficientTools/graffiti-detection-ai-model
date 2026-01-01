# 🎨 Graffiti Detection AI Model

**AI-Powered Graffiti Detection Using YOLOv8**

![Graffiti Detection](https://img.shields.io/badge/YOLOv8-Object%20Detection-blue?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> [!NOTE]
> **Production-Ready Graffiti Detection System**
> 
> Detect graffiti across multiple contexts including walls, buildings, bridges, and vehicles. Built with YOLOv8 for real-time, accurate detection.

An end-to-end deep learning system that automatically identifies and locates graffiti in images, helping authorities and property managers efficiently address urban vandalism.

**Key Features:**
- 🎯 Multi-context detection (walls, buildings, bridges, vehicles)
- ⚡ Real-time inference with optimized YOLOv8 architecture
- 📊 Complete training, evaluation, and deployment pipeline
- 🎨 Advanced data augmentation (weather, lighting, geometric transforms)
- 🛡️ Production-ready exports (ONNX, TensorRT, CoreML)

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Export & Deployment](#export--deployment)
- [Contributing](#contributing)
- [Author](#author)

## Requirements

* Python 3.8 or higher
* CUDA-capable GPU (recommended for training)
* 8GB+ RAM and 10GB+ disk space
* Annotated graffiti dataset in YOLO format

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/graffiti-detection-ai-model.git
   cd graffiti-detection-ai-model
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Prepare Your Dataset

Collect and annotate graffiti images (aim for 1500+ diverse images):

```bash
# Organize your raw images and YOLO format labels
python scripts/prepare_dataset.py --data-dir data/raw --output-dir data
```

**Supported Annotation Tools:**
- **LabelImg** - Desktop tool for bounding box annotation
- **CVAT** - Web-based collaborative annotation
- **Roboflow** - Cloud-based annotation with AI-assist

### 2. Train the Model

**Basic Training:**
```bash
python scripts/train.py --data configs/dataset.yaml --model yolov8n --epochs 100
```

**Advanced Training:**
```bash
python scripts/train.py \
    --data configs/dataset.yaml \
    --model yolov8m \
    --epochs 100 \
    --batch-size 16 \
    --device 0
```

### 3. Run Inference

**Single Image:**
```bash
python scripts/inference.py --model models/best.pt --source image.jpg
```

**Batch Images:**
```bash
python scripts/inference.py --model models/best.pt --source images_folder/
```

**Real-time Webcam:**
```bash
python scripts/inference.py --model models/best.pt --source 0 --show
```

## Training

The training pipeline supports multiple YOLOv8 variants with configurable parameters:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| YOLOv8n | ⚡⚡⚡ | ⭐⭐ | Real-time, edge devices |
| YOLOv8s | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| YOLOv8m | ⚡ | ⭐⭐⭐⭐ | High accuracy needed |
| YOLOv8l | 🐌 | ⭐⭐⭐⭐⭐ | Best accuracy |

**Key Training Features:**
- 🎨 Advanced augmentation with weather, lighting, perspective transforms
- 📊 Progress tracking via TensorBoard
- 💾 Automatic checkpointing
- 🔄 Resume training from checkpoints
- ⚡ Mixed precision (FP16) for faster training

## Inference

### Usage Examples

**Single image detection:**
```bash
python scripts/inference.py --model models/best.pt --source image.jpg --conf-threshold 0.25
```

**Save detection crops:**
```bash
python scripts/inference.py --model models/best.pt --source image.jpg --save-crops
```

**Custom confidence threshold:**
```bash
python scripts/inference.py --model models/best.pt --source image.jpg --conf-threshold 0.5
```

## Evaluation

Evaluate your trained model on the test set:

```bash
python scripts/evaluate.py --model models/best.pt --data configs/dataset.yaml
```

**Metrics Reported:**
- mAP@0.5 - Standard object detection metric
- mAP@0.5:0.95 - COCO-style average precision
- Precision/Recall - Classification accuracy
- F1-Score - Harmonic mean of precision and recall

## Export & Deployment

Export your model for different deployment scenarios:

**ONNX (Cross-platform):**
```bash
yolo export model=models/best.pt format=onnx
```

**TensorRT (NVIDIA GPUs):**
```bash
yolo export model=models/best.pt format=engine device=0
```

**CoreML (iOS/macOS):**
```bash
yolo export model=models/best.pt format=coreml
```

**Deployment Options:**
- 🌐 REST API: Deploy with FastAPI or Flask
- 📱 Mobile: TensorFlow Lite or CoreML
- 🖥️ Edge: TensorRT for NVIDIA Jetson
- ☁️ Cloud: AWS/GCP/Azure deployment

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

**Areas for Improvement:**
- Dataset expansion with diverse graffiti examples
- Multi-class detection for graffiti types
- Segmentation model for pixel-level detection
- Mobile deployment optimization
- Real-time video stream processing

## Author

**Pierre-Henry Soria** - Passionate software AI engineer building intelligent systems to solve real-world problems.

☕️ Enjoying this project? [Buy me a coffee](https://ko-fi.com/phenry) to support more AI innovations!

[![@phenrysay](https://img.shields.io/badge/x-000000?style=for-the-badge&logo=x)](https://x.com/phenrysay)
[![pH-7](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/pH-7)

## License

This project is distributed under the [MIT License](LICENSE).

## Disclaimer

This model is designed to assist in identifying graffiti for maintenance and urban management purposes. Always respect local laws and privacy regulations when deploying computer vision systems in public spaces.
