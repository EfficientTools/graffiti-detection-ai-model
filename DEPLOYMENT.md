# Graffiti Detection AI - Deployment Guide

## Quick Deploy Options

### 1. Docker Deployment (Recommended)

#### Build and Run
```bash
# Build image
docker build -t graffiti-detector:latest .

# Run API service
docker run -d \
  --name graffiti-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  graffiti-detector:latest

# Test API
curl http://localhost:8000/
```

#### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Kubernetes Deployment

```bash
# Apply deployment
kubectl apply -f deployment/k8s-deployment.yaml

# Check status
kubectl get pods -l app=graffiti-detector

# Get service URL
kubectl get service graffiti-detector-service
```

### 3. Edge Device (NVIDIA Jetson)

```bash
# Export model to TensorRT
yolo export model=models/best.pt format=engine device=0

# Run optimized inference
python scripts/inference.py \
  --model models/best.engine \
  --source rtsp://camera-ip/stream \
  --conf-threshold 0.25 \
  --alert-config configs/alerts.json
```

### 4. Cloud Deployment

#### AWS (EC2 with GPU)
```bash
# Launch g4dn.xlarge instance
# Install NVIDIA drivers and Docker
# Deploy using Docker commands above
```

#### GCP (Compute Engine with GPU)
```bash
# Create instance with GPU
gcloud compute instances create graffiti-detector \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# SSH and deploy
```

## Production Configuration

### Environment Variables
```bash
export MODEL_PATH=/app/models/best.pt
export ALERT_CONFIG=/app/configs/alerts.json
export LOG_LEVEL=INFO
export API_PORT=8000
```

### Monitoring & Logging
- Logs: `outputs/logs/`
- Detections: `outputs/detections/`
- Metrics: Available via API `/stats` endpoint

### Security
- Use HTTPS in production
- Implement API authentication
- Restrict camera access
- Secure alert credentials

## Performance Optimization

### Model Selection
- **Real-time (>30 FPS)**: YOLOv8n
- **Balanced**: YOLOv8s
- **High Accuracy**: YOLOv8m/l

### Hardware Recommendations
- **Edge Devices**: NVIDIA Jetson Nano/Xavier
- **Server**: GPU with 8GB+ VRAM
- **Cloud**: AWS g4dn.xlarge or GCP n1-standard-4 with T4

## Scaling

### Load Balancing
Use nginx or cloud load balancers to distribute requests across multiple API instances.

### Multi-Camera
Each camera runs in a separate thread. For >10 cameras, consider distributed deployment.

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check GPU availability
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory
- Reduce batch size
- Use smaller model variant (yolov8n)
- Reduce image size

### Camera Connection Issues
- Verify RTSP URL format
- Check network connectivity
- Test with VLC: `vlc rtsp://camera-ip/stream`
