# 🎨 Graffiti Detection AI Model

**AI-Powered Graffiti Detection Using YOLOv8**

![Graffiti Detection](https://img.shields.io/badge/YOLOv8-Object%20Detection-blue?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> [!NOTE]
> **Real-Time Anti-Vandalism Detection System**
> 
> Instantly detect and alert on any sign of graffiti vandalism across walls, buildings, bridges, and vehicles. Built with YOLOv8 for immediate response and prevention.

A powerful AI-driven system that instantly identifies and alerts on graffiti vandalism, enabling rapid response to prevent urban decay. Deploy on surveillance cameras, mobile apps, or edge devices for continuous 24/7 monitoring and immediate threat detection.

**Key Features:**
- 🚨 **Instant Vandalism Alerts** - Immediate detection and notification of graffiti incidents
- ⚡ **Real-Time Processing** - Sub-second inference for rapid response (<50ms per frame)
- 📹 **24/7 Surveillance Integration** - Deploy on CCTV cameras for continuous monitoring
- 🎯 **Multi-Context Detection** - Walls, buildings, bridges, vehicles, trains, and public spaces
- 🔔 **Alert System Ready** - Integration with SMS, email, push notifications, and security systems
- 📊 **Incident Logging** - Automatic timestamping and geo-tagging of vandalism events
- 🛡️ **Edge Deployment** - Run on edge devices (NVIDIA Jetson, Raspberry Pi) for offline operation
- 🌐 **API-Ready** - RESTful API for integration with existing security infrastructure

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Real-Time Surveillance](#real-time-surveillance)
- [Alert System Integration](#alert-system-integration)
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

## Real-Time Surveillance

### Deploy on CCTV/IP Cameras

Monitor live camera feeds for immediate graffiti detection:

**RTSP Stream Monitoring:**
```bash
python scripts/inference.py \
    --model models/best.pt \
    --source rtsp://username:password@camera-ip:554/stream \
    --conf-threshold 0.3 \
    --save-detections
```

**Multiple Camera Monitoring:**
```bash
# Monitor multiple cameras simultaneously
python scripts/multi_camera_surveillance.py \
    --model models/best.pt \
    --cameras cameras_config.json \
    --alert-webhook https://your-alert-system.com/webhook
```

**Edge Device Deployment (NVIDIA Jetson):**
```bash
# Export to TensorRT for edge deployment
yolo export model=models/best.pt format=engine device=0

# Run optimized inference on Jetson
python scripts/inference.py \
    --model models/best.engine \
    --source /dev/video0 \
    --conf-threshold 0.25
```

## Alert System Integration

### Immediate Notification Setup

Configure instant alerts when graffiti is detected:

**Email Alerts:**
```bash
python scripts/inference.py \
    --model models/best.pt \
    --source rtsp://camera-ip/stream \
    --alert-email security@company.com \
    --smtp-config smtp_settings.json
```

**SMS/Push Notifications:**
```bash
python scripts/inference.py \
    --model models/best.pt \
    --source 0 \
    --alert-sms +1234567890 \
    --alert-service twilio \
    --min-confidence 0.4
```

**Webhook Integration:**
```bash
# Send detection events to your security system
python scripts/inference.py \
    --model models/best.pt \
    --source rtsp://camera-ip/stream \
    --webhook-url https://security-api.com/graffiti-alert \
    --include-image \
    --geo-tag
```

**Discord/Slack Alerts:**
```python
# Example: Discord webhook integration
from detection_alerts import DiscordAlert

alert = DiscordAlert(webhook_url="YOUR_DISCORD_WEBHOOK")
alert.send_graffiti_detection(
    image_path="detections/graffiti_001.jpg",
    confidence=0.87,
    location="Building A, Camera 3",
    timestamp="2026-01-02 14:23:15"
)
```

### Automated Response System

**Integration with Security Systems:**
- Trigger alarms when graffiti detected
- Activate additional cameras to capture perpetrators
- Log incidents with timestamp and location
- Generate daily/weekly vandalism reports
- Alert nearest security personnel via mobile app

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

**Deployment Scenarios:**

| Deployment | Use Case | Response Time | Best For |
|------------|----------|---------------|----------|
| 🌐 **REST API** | Centralized monitoring hub | ~100ms | Multiple camera integration |
| 📱 **Mobile App** | Field inspection & reporting | ~50ms | Property managers, inspectors |
| 🖥️ **Edge (Jetson)** | Standalone surveillance | <30ms | 24/7 real-time monitoring |
| ☁️ **Cloud (AWS/GCP)** | Large-scale city deployment | ~150ms | City-wide surveillance networks |
| 🏢 **On-Premise Server** | Private security systems | ~80ms | Corporate/institutional security |

**Quick Deploy Examples:**

```bash
# Deploy as REST API service
uvicorn api.graffiti_detector:app --host 0.0.0.0 --port 8000

# Docker deployment with auto-restart
docker run -d --restart always \
  --gpus all \
  -p 8000:8000 \
  graffiti-detector:latest

# Kubernetes deployment for high availability
kubectl apply -f kubernetes/graffiti-detection-deployment.yaml
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

**Areas for Contribution:**
- 🎯 Dataset expansion with diverse vandalism examples (tags, murals, scratchiti)
- 🔍 Multi-class detection for graffiti types and severity levels
- 🎨 Segmentation model for pixel-level vandalism mapping
- 📱 Mobile deployment optimization for field inspectors
- 🤖 Automated perpetrator tracking across multiple cameras
- 📊 Vandalism hotspot analytics and predictive modeling
- 🔔 Advanced alert filtering to reduce false positives
- 🌐 Multi-language support for international deployment

## Author

<div align="center">

[![Pierre-Henry Soria](https://s.gravatar.com/avatar/a210fe61253c43c869d71eaed0e90149?s=200 "Pierre-Henry Soria - Software AI Engineer")](https://pierrehenry.dev)

**Pierre-Henry Soria**

Passionate software AI engineer building intelligent systems to solve real-world problems.

☕️ Enjoying this project? [Buy me a coffee](https://ko-fi.com/phenry) to support more AI innovations!

[![@phenrysay](https://img.shields.io/badge/x-000000?style=for-the-badge&logo=x)](https://x.com/phenrysay)
[![pH-7](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/pH-7)

</div>

## License

This project is distributed under the [MIT License](LICENSE).

## Disclaimer

This model is designed to assist in identifying graffiti for maintenance and urban management purposes. Always respect local laws and privacy regulations when deploying computer vision systems in public spaces.
