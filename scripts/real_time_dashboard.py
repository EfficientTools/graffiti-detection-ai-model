#!/usr/bin/env python3
"""
Real-Time Anti-Vandalism Dashboard
Displays live detection statistics, camera feeds, and alert history
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from typing import Dict, List

import cv2
import numpy as np


class AntiVandalismDashboard:
    """Real-time dashboard for graffiti surveillance system"""
    
    def __init__(self, stats_file: str = "outputs/stats.json"):
        self.stats_file = Path(stats_file)
        self.detection_history = deque(maxlen=100)
        self.alert_history = deque(maxlen=50)
        self.camera_stats = {}
        
        # Dashboard window
        self.window_name = "🚨 Anti-Vandalism Surveillance Dashboard"
        self.width = 1400
        self.height = 900
        
        # Statistics
        self.total_detections = 0
        self.detections_last_hour = 0
        self.active_cameras = 0
        self.avg_confidence = 0.0
        self.last_detection_time = None
        
    def update_stats(self):
        """Update statistics from log file"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                    
                self.total_detections = stats.get('total_detections', 0)
                self.active_cameras = stats.get('active_cameras', 0)
                self.avg_confidence = stats.get('avg_confidence', 0.0)
                
                # Update detection history
                detections = stats.get('recent_detections', [])
                for det in detections:
                    if det not in self.detection_history:
                        self.detection_history.append(det)
                        
                # Count last hour
                now = datetime.now()
                hour_ago = now - timedelta(hours=1)
                self.detections_last_hour = sum(
                    1 for det in self.detection_history 
                    if datetime.fromisoformat(det.get('timestamp', '2000-01-01')) > hour_ago
                )
                
            except Exception as e:
                print(f"Error reading stats: {e}")
    
    def create_dashboard(self):
        """Create dashboard visualization"""
        # Create blank canvas
        dashboard = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        dashboard[:] = (30, 30, 30)  # Dark background
        
        # Header
        self._draw_header(dashboard)
        
        # Statistics panels
        self._draw_stats_panel(dashboard)
        
        # Detection timeline
        self._draw_timeline(dashboard)
        
        # Alert list
        self._draw_alert_list(dashboard)
        
        # Camera status
        self._draw_camera_status(dashboard)
        
        # Footer with timestamp
        self._draw_footer(dashboard)
        
        return dashboard
    
    def _draw_header(self, img):
        """Draw dashboard header"""
        # Title
        cv2.rectangle(img, (0, 0), (self.width, 80), (45, 45, 45), -1)
        cv2.putText(
            img,
            "ANTI-VANDALISM SURVEILLANCE SYSTEM",
            (20, 50),
            cv2.FONT_HERSHEY_BOLD,
            1.2,
            (255, 255, 255),
            2
        )
        
        # Status indicator
        status_text = "ACTIVE"
        status_color = (0, 255, 0) if self.active_cameras > 0 else (0, 0, 255)
        cv2.circle(img, (self.width - 150, 40), 15, status_color, -1)
        cv2.putText(
            img,
            status_text,
            (self.width - 120, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )
    
    def _draw_stats_panel(self, img):
        """Draw statistics panel"""
        y_start = 100
        panel_height = 150
        
        # Background
        cv2.rectangle(img, (20, y_start), (self.width - 20, y_start + panel_height), (50, 50, 50), -1)
        
        # Stats boxes
        stats = [
            ("TOTAL DETECTIONS", str(self.total_detections), (100, 200, 255)),
            ("LAST HOUR", str(self.detections_last_hour), (255, 200, 100)),
            ("ACTIVE CAMERAS", str(self.active_cameras), (100, 255, 150)),
            ("AVG CONFIDENCE", f"{self.avg_confidence:.1%}", (255, 150, 200))
        ]
        
        box_width = (self.width - 100) // 4
        
        for i, (label, value, color) in enumerate(stats):
            x = 40 + i * (box_width + 20)
            
            # Label
            cv2.putText(
                img,
                label,
                (x, y_start + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )
            
            # Value
            cv2.putText(
                img,
                value,
                (x, y_start + 90),
                cv2.FONT_HERSHEY_BOLD,
                1.5,
                color,
                2
            )
    
    def _draw_timeline(self, img):
        """Draw detection timeline"""
        y_start = 270
        panel_height = 200
        
        cv2.rectangle(img, (20, y_start), (self.width - 20, y_start + panel_height), (50, 50, 50), -1)
        cv2.putText(
            img,
            "DETECTION TIMELINE (Last 100)",
            (40, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw timeline
        if len(self.detection_history) > 0:
            timeline_y = y_start + 120
            timeline_width = self.width - 80
            
            # Base line
            cv2.line(
                img,
                (40, timeline_y),
                (40 + timeline_width, timeline_y),
                (100, 100, 100),
                2
            )
            
            # Detection markers
            for i, det in enumerate(list(self.detection_history)[-50:]):
                x = 40 + int(i * timeline_width / 50)
                confidence = det.get('confidence', 0.5)
                
                # Color based on confidence
                if confidence > 0.7:
                    color = (0, 0, 255)  # High confidence - Red
                elif confidence > 0.4:
                    color = (0, 165, 255)  # Medium - Orange
                else:
                    color = (0, 255, 255)  # Low - Yellow
                
                height = int(confidence * 60)
                cv2.line(img, (x, timeline_y), (x, timeline_y - height), color, 2)
    
    def _draw_alert_list(self, img):
        """Draw recent alerts list"""
        y_start = 490
        panel_height = 250
        
        cv2.rectangle(img, (20, y_start), (self.width // 2 - 10, y_start + panel_height), (50, 50, 50), -1)
        cv2.putText(
            img,
            "RECENT ALERTS",
            (40, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # List recent detections
        y_offset = 60
        for i, det in enumerate(list(self.detection_history)[-8:]):
            y = y_start + y_offset + i * 25
            
            timestamp = det.get('timestamp', 'N/A')
            camera = det.get('camera_id', 'Unknown')
            confidence = det.get('confidence', 0.0)
            
            # Alert icon
            alert_color = (0, 0, 255) if confidence > 0.6 else (0, 165, 255)
            cv2.circle(img, (50, y), 5, alert_color, -1)
            
            # Alert text
            alert_text = f"{timestamp[:19]} | {camera} | Conf: {confidence:.0%}"
            cv2.putText(
                img,
                alert_text,
                (70, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1
            )
    
    def _draw_camera_status(self, img):
        """Draw camera status panel"""
        y_start = 490
        panel_height = 250
        x_start = self.width // 2 + 10
        
        cv2.rectangle(img, (x_start, y_start), (self.width - 20, y_start + panel_height), (50, 50, 50), -1)
        cv2.putText(
            img,
            "CAMERA STATUS",
            (x_start + 20, y_start + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # List cameras (example)
        cameras = [
            ("Camera 1 - Building A", "ACTIVE", True),
            ("Camera 2 - Parking", "ACTIVE", True),
            ("Camera 3 - Bridge", "ACTIVE", True),
            ("Camera 4 - East Gate", "OFFLINE", False),
        ]
        
        y_offset = 60
        for i, (name, status, is_active) in enumerate(cameras):
            y = y_start + y_offset + i * 50
            
            # Status indicator
            status_color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.circle(img, (x_start + 30, y), 8, status_color, -1)
            
            # Camera name
            cv2.putText(
                img,
                name,
                (x_start + 50, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1
            )
            
            # Status
            cv2.putText(
                img,
                status,
                (x_start + 400, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                1
            )
    
    def _draw_footer(self, img):
        """Draw footer with timestamp"""
        cv2.rectangle(img, (0, self.height - 50), (self.width, self.height), (45, 45, 45), -1)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            img,
            f"Last Update: {current_time}",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        
        cv2.putText(
            img,
            "Press 'Q' to quit | 'R' to reset stats",
            (self.width - 400, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
    
    def run(self, update_interval: float = 1.0):
        """Run the dashboard"""
        print(f"🖥️  Starting Anti-Vandalism Dashboard...")
        print(f"📊 Update interval: {update_interval}s")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
        try:
            while True:
                # Update statistics
                self.update_stats()
                
                # Create dashboard
                dashboard = self.create_dashboard()
                
                # Display
                cv2.imshow(self.window_name, dashboard)
                
                # Handle key press
                key = cv2.waitKey(int(update_interval * 1000)) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    self.detection_history.clear()
                    self.alert_history.clear()
                    print("📊 Statistics reset")
                    
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Anti-Vandalism Dashboard")
    parser.add_argument("--stats-file", type=str, default="outputs/stats.json",
                        help="Path to statistics JSON file")
    parser.add_argument("--update-interval", type=float, default=1.0,
                        help="Dashboard update interval in seconds")
    
    args = parser.parse_args()
    
    dashboard = AntiVandalismDashboard(args.stats_file)
    dashboard.run(args.update_interval)


if __name__ == "__main__":
    main()
