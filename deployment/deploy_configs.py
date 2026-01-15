"""
Deployment scripts and configurations for production environments
"""

# Docker deployment example
DOCKERFILE = """
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api.graffiti_detector:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Docker Compose for multi-service deployment
DOCKER_COMPOSE = """
version: '3.8'

services:
  graffiti-detector:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - MODEL_PATH=/app/models/best.pt
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  surveillance-monitor:
    build: .
    command: python scripts/multi_camera_surveillance.py --model models/best.pt --cameras configs/cameras.json --alert-config configs/alerts.json
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
"""

# Kubernetes deployment
KUBERNETES_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graffiti-detector
  labels:
    app: graffiti-detector
spec:
  replicas: 2
  selector:
    matchLabels:
      app: graffiti-detector
  template:
    metadata:
      labels:
        app: graffiti-detector
    spec:
      containers:
      - name: api
        image: graffiti-detector:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: models
          mountPath: /app/models
        env:
        - name: MODEL_PATH
          value: "/app/models/best.pt"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: graffiti-detector-service
spec:
  selector:
    app: graffiti-detector
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
"""

print("Deployment configurations available")
