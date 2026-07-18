FROM python:3.11-slim

LABEL org.opencontainers.image.title="Graffiti Detection AI Model" \
      org.opencontainers.image.description="YOLOv8 graffiti detection API and surveillance tools" \
      org.opencontainers.image.source="https://github.com/EfficientTools/graffiti-detection-ai-model" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY graffiti_detection ./graffiti_detection

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[all]"

COPY api ./api
COPY scripts ./scripts
COPY configs ./configs

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p models outputs/detections outputs/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD ["curl", "--fail", "--silent", "--show-error", "http://localhost:8000/"]

CMD ["uvicorn", "api.graffiti_detector:app", "--host", "0.0.0.0", "--port", "8000"]
