# Deployment

The repository ships a Docker image for the FastAPI service and an optional Compose profile for multi-camera monitoring. Provide your own trained model at `models/best.pt` before starting either service.

## API Container

```bash
docker build -t graffiti-detection-ai:1.4.0 .
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -v graffiti-detection-output:/app/outputs \
  graffiti-detection-ai:1.4.0
```

The default image is portable and CPU-compatible. GPU deployments should use a PyTorch base image that matches the host CUDA runtime. Check service health at `http://localhost:8000/`.

## Docker Compose

Start the API:

```bash
docker compose up -d api
```

To add camera monitoring, create local configuration files first:

```bash
cp configs/cameras_example.json configs/cameras.json
cp configs/alerts_example.json configs/alerts.json
docker compose --profile surveillance up -d
```

The real configuration files are ignored by Git because they can contain camera credentials and alert secrets.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/best.pt` | Model loaded by the API |
| `LOG_LEVEL` | `INFO` | Application log level |
| `TZ` | `UTC` | Surveillance container timezone |

## Production Checklist

- Terminate TLS at a reverse proxy or load balancer.
- Add authentication and request limits before exposing the API publicly.
- Store camera and alert credentials in a secret manager or mounted read-only files.
- Restrict access to camera networks, models, saved detections, and logs.
- Set retention rules that comply with local privacy requirements.
- Pin the image tag and scan it before deployment.
