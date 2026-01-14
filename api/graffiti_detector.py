"""
FastAPI REST API for Real-Time Graffiti Detection
Deploy as a service for integration with security systems
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import io
from datetime import datetime
import json
from pathlib import Path

app = FastAPI(
    title="Graffiti Detection API",
    description="Real-time AI-powered graffiti detection and vandalism alert system",
    version="1.0.0"
)

# Load model (update path as needed)
MODEL_PATH = "models/best.pt"
model = None

@app.on_event("startup")
async def load_model():
    """Load YOLO model on startup"""
    global model
    model = YOLO(MODEL_PATH)
    print(f"✓ Model loaded: {MODEL_PATH}")


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


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "Graffiti Detection API",
        "version": "1.0.0",
        "model": MODEL_PATH
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_graffiti(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    trigger_alert: bool = True
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
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        start_time = datetime.now()
        results = model(image, conf=conf_threshold, verbose=False)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Extract detections
        boxes = results[0].boxes
        detections = len(boxes)
        
        confidence_scores = [float(box.conf) for box in boxes]
        bounding_boxes = [box.xyxy[0].tolist() for box in boxes]
        
        # Check if alert should be triggered
        alert_triggered = trigger_alert and detections > 0 and max(confidence_scores, default=0) >= conf_threshold
        
        return DetectionResponse(
            detections=detections,
            confidence_scores=confidence_scores,
            bounding_boxes=bounding_boxes,
            timestamp=datetime.now().isoformat(),
            alert_triggered=alert_triggered,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/annotated")
async def detect_and_annotate(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25
):
    """
    Detect graffiti and return annotated image
    
    Returns image with bounding boxes drawn
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model(image, conf=conf_threshold, verbose=False)
        
        # Get annotated image
        annotated = results[0].plot()
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', annotated)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    conf_threshold: float = 0.25
):
    """
    Detect graffiti in multiple images
    
    Returns detection results for each image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results_list = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results_list.append({
                    "filename": file.filename,
                    "error": "Invalid image"
                })
                continue
            
            # Run detection
            results = model(image, conf=conf_threshold, verbose=False)
            boxes = results[0].boxes
            
            results_list.append({
                "filename": file.filename,
                "detections": len(boxes),
                "confidence_scores": [float(box.conf) for box in boxes],
                "bounding_boxes": [box.xyxy[0].tolist() for box in boxes]
            })
            
        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results_list, "total_images": len(files)}


@app.get("/stats")
async def get_stats():
    """Get model statistics and performance metrics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "model_type": "YOLOv8",
        "classes": ["graffiti"],
        "input_size": 640,
        "status": "ready"
    }


@app.post("/alert/test")
async def test_alert(config: AlertConfig):
    """Test alert system configuration"""
    # This would integrate with your alert system
    return {
        "status": "Alert configuration valid",
        "config": config.dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
