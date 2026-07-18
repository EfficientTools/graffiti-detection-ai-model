"""FastAPI service for graffiti detection."""

import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

from graffiti_detection import __version__

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
logger = logging.getLogger(__name__)
model = None
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once for the lifetime of the API process."""
    global model
    model = YOLO(MODEL_PATH)
    try:
        yield
    finally:
        model = None


app = FastAPI(
    title="Graffiti Detection API",
    description="Real-time graffiti detection API",
    version=__version__,
    lifespan=lifespan,
)


class DetectionResponse(BaseModel):
    """Detection response schema"""

    detections: int
    confidence_scores: List[float]
    bounding_boxes: List[List[float]]
    timestamp: str
    alert_triggered: bool
    processing_time_ms: float


class AlertConfig(BaseModel):
    """Alert configuration"""

    min_confidence: float = 0.3
    alert_webhook: Optional[str] = None
    send_email: bool = False
    email_recipients: Optional[List[str]] = None


def decode_image(contents: bytes) -> np.ndarray:
    """Decode an uploaded image or return a client-facing validation error."""
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions are too large")
    return image


async def read_uploaded_image(file: UploadFile) -> np.ndarray:
    """Read one bounded image upload and validate its declared media type."""
    if file.content_type and file.content_type.casefold() not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image file is too large")
    return decode_image(contents)


async def predict(image: np.ndarray, confidence: float):
    """Serialize access to the shared model without blocking the event loop."""
    async with inference_lock:
        return await run_in_threadpool(model, image, conf=confidence, verbose=False)


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Graffiti Detection API",
        "version": __version__,
        "model": MODEL_PATH,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_graffiti(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0),
    trigger_alert: bool = True,
):
    """
    Detect graffiti in uploaded image

    - **file**: Image file (JPG, PNG)
    - **conf_threshold**: Confidence threshold (0.0-1.0)
    - **trigger_alert**: Whether to trigger alerts on detection
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = await read_uploaded_image(file)

        # Run detection
        start_time = perf_counter()
        results = await predict(image, conf_threshold)
        processing_time = (perf_counter() - start_time) * 1000

        # Extract detections
        boxes = results[0].boxes
        detections = len(boxes)

        confidence_scores = [float(box.conf) for box in boxes]
        bounding_boxes = [box.xyxy[0].tolist() for box in boxes]

        # Check if alert should be triggered
        alert_triggered = (
            trigger_alert and detections > 0 and max(confidence_scores, default=0) >= conf_threshold
        )

        return DetectionResponse(
            detections=detections,
            confidence_scores=confidence_scores,
            bounding_boxes=bounding_boxes,
            timestamp=datetime.now(timezone.utc).isoformat(),
            alert_triggered=alert_triggered,
            processing_time_ms=round(processing_time, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Graffiti detection failed")
        raise HTTPException(status_code=500, detail="Detection failed") from exc


@app.post("/detect/annotated")
async def detect_and_annotate(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0),
):
    """
    Detect graffiti and return annotated image

    Returns image with bounding boxes drawn
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = await read_uploaded_image(file)

        # Run detection
        results = await predict(image, conf_threshold)

        # Get annotated image
        annotated = results[0].plot()

        # Encode to JPEG
        encoded, buffer = cv2.imencode(".jpg", annotated)
        if not encoded:
            raise HTTPException(status_code=500, detail="Failed to encode result")
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Annotated graffiti detection failed")
        raise HTTPException(status_code=500, detail="Detection failed") from exc


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = Query(0.25, ge=0.0, le=1.0),
):
    """
    Detect graffiti in multiple images

    Returns detection results for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains more than {MAX_BATCH_SIZE} images",
        )

    results_list = []

    for file in files:
        try:
            image = await read_uploaded_image(file)

            # Run detection
            results = await predict(image, conf_threshold)
            boxes = results[0].boxes

            results_list.append(
                {
                    "filename": file.filename,
                    "detections": len(boxes),
                    "confidence_scores": [float(box.conf) for box in boxes],
                    "bounding_boxes": [box.xyxy[0].tolist() for box in boxes],
                }
            )

        except HTTPException as exc:
            results_list.append(
                {
                    "filename": file.filename,
                    "error": exc.detail,
                }
            )
        except Exception:
            logger.exception("Batch graffiti detection failed for %s", file.filename)
            results_list.append({"filename": file.filename, "error": "Detection failed"})

    return {"results": results_list, "total_images": len(files)}


@app.get("/stats")
async def get_stats():
    """Get model statistics and performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_path": MODEL_PATH,
        "model_type": "YOLO object detector",
        "classes": ["graffiti"],
        "input_size": 640,
        "status": "ready",
    }


@app.post("/alert/test")
async def test_alert(config: AlertConfig):
    """Test alert system configuration"""
    return {"status": "Alert configuration valid", "config": config.model_dump()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
