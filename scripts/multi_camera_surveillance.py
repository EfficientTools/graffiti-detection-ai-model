#!/usr/bin/env python3
"""
Multi-Camera Surveillance System for Real-Time Graffiti Detection
Monitors multiple camera feeds simultaneously and sends instant alerts
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.alerts import AlertManager
from src.utils.visualization import draw_boxes


class CameraMonitor:
    """Monitor a single camera feed for graffiti detection"""
    
    def __init__(
        self,
        camera_id: str,
        source: str,
        model: YOLO,
        conf_threshold: float = 0.3,
        alert_queue: Queue = None
    ):
        self.camera_id = camera_id
        self.source = source
        self.model = model
        self.conf_threshold = conf_threshold
        self.alert_queue = alert_queue
        self.running = False
        self.cap = None
        
    def start(self):
        """Start monitoring the camera feed"""
        self.running = True
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera: {self.camera_id}")
            return
            
        print(f"[INFO] Started monitoring camera: {self.camera_id}")
        
        frame_count = 0
        detection_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"[WARNING] Failed to read from camera: {self.camera_id}")
                time.sleep(1)
                continue
                
            frame_count += 1
            
            # Run detection every frame (adjust for performance)
            if frame_count % 1 == 0:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Check for detections
                if len(results[0].boxes) > 0:
                    detection_count += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Draw detections
                    annotated_frame = results[0].plot()
                    
                    # Prepare alert data
                    alert_data = {
                        'camera_id': self.camera_id,
                        'timestamp': timestamp,
                        'detections': len(results[0].boxes),
                        'frame': annotated_frame,
                        'confidences': [float(box.conf) for box in results[0].boxes]
                    }
                    
                    # Send alert
                    if self.alert_queue:
                        self.alert_queue.put(alert_data)
                    
                    print(f"[ALERT] Graffiti detected on {self.camera_id} at {timestamp} "
                          f"(Confidence: {max(alert_data['confidences']):.2f})")
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
        
        self.cap.release()
        print(f"[INFO] Stopped monitoring camera: {self.camera_id} "
              f"(Total detections: {detection_count})")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False


class MultiCameraSurveillance:
    """Manage multiple camera feeds for graffiti surveillance"""
    
    def __init__(
        self,
        model_path: str,
        cameras_config: Dict,
        alert_config: Optional[Dict] = None
    ):
        self.model = YOLO(model_path)
        self.cameras_config = cameras_config
        self.alert_queue = Queue()
        self.monitors = []
        self.threads = []
        
        # Initialize alert manager
        self.alert_manager = None
        if alert_config:
            self.alert_manager = AlertManager(alert_config)
    
    def start_monitoring(self):
        """Start monitoring all cameras"""
        print(f"[INFO] Starting surveillance system with {len(self.cameras_config)} cameras")
        
        for camera_id, camera_info in self.cameras_config.items():
            monitor = CameraMonitor(
                camera_id=camera_id,
                source=camera_info['source'],
                model=self.model,
                conf_threshold=camera_info.get('conf_threshold', 0.3),
                alert_queue=self.alert_queue
            )
            
            thread = threading.Thread(target=monitor.start, daemon=True)
            thread.start()
            
            self.monitors.append(monitor)
            self.threads.append(thread)
            
            # Small delay between camera starts
            time.sleep(0.5)
        
        # Start alert processor
        alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        alert_thread.start()
        
        print("[INFO] All cameras are now active. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down surveillance system...")
            self.stop_monitoring()
    
    def _process_alerts(self):
        """Process alerts from the queue"""
        while True:
            alert_data = self.alert_queue.get()
            
            if alert_data is None:
                break
            
            # Send alerts through configured channels
            if self.alert_manager:
                self.alert_manager.send_alert(alert_data)
            
            # Save detection image
            output_dir = Path("outputs/detections")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{alert_data['camera_id']}_{alert_data['timestamp'].replace(':', '-').replace(' ', '_')}.jpg"
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), alert_data['frame'])
            
            self.alert_queue.task_done()
    
    def stop_monitoring(self):
        """Stop all camera monitors"""
        for monitor in self.monitors:
            monitor.stop()
        
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.alert_queue.put(None)
        print("[INFO] Surveillance system stopped.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Graffiti Surveillance System")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO model")
    parser.add_argument("--cameras", type=str, required=True,
                        help="Path to cameras configuration JSON file")
    parser.add_argument("--alert-config", type=str, default=None,
                        help="Path to alert configuration JSON file")
    
    args = parser.parse_args()
    
    # Load cameras configuration
    with open(args.cameras, 'r') as f:
        cameras_config = json.load(f)
    
    # Load alert configuration
    alert_config = None
    if args.alert_config:
        with open(args.alert_config, 'r') as f:
            alert_config = json.load(f)
    
    # Start surveillance system
    surveillance = MultiCameraSurveillance(
        model_path=args.model,
        cameras_config=cameras_config,
        alert_config=alert_config
    )
    
    surveillance.start_monitoring()


if __name__ == "__main__":
    main()
