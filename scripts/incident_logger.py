#!/usr/bin/env python3
"""
Incident Logging System for Graffiti Vandalism Detection
Maintains detailed logs of all detection events with metadata
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3


class IncidentLogger:
    """Log and track graffiti vandalism incidents"""
    
    def __init__(self, log_dir: str = "outputs/incidents"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.log_dir / "incidents.db"
        self._init_database()
        
        # JSON log file
        self.json_log = self.log_dir / "incidents.json"
        self.incidents = self._load_json_log()
        
    def _init_database(self):
        """Initialize SQLite database for incidents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                location TEXT,
                latitude REAL,
                longitude REAL,
                confidence REAL NOT NULL,
                detections INTEGER NOT NULL,
                image_path TEXT,
                video_clip_path TEXT,
                alert_sent BOOLEAN DEFAULT 0,
                alert_type TEXT,
                response_time TEXT,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON incidents(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_camera ON incidents(camera_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status ON incidents(status)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_json_log(self) -> List[Dict]:
        """Load existing JSON log"""
        if self.json_log.exists():
            with open(self.json_log, 'r') as f:
                return json.load(f)
        return []
    
    def log_incident(
        self,
        camera_id: str,
        confidence: float,
        detections: int,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        image_path: Optional[str] = None,
        video_clip_path: Optional[str] = None,
        alert_sent: bool = False,
        alert_type: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Log a graffiti vandalism incident
        
        Returns:
            Incident ID
        """
        timestamp = datetime.now().isoformat()
        
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incidents (
                timestamp, camera_id, location, latitude, longitude,
                confidence, detections, image_path, video_clip_path,
                alert_sent, alert_type, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, camera_id, location, latitude, longitude,
            confidence, detections, image_path, video_clip_path,
            alert_sent, alert_type, notes
        ))
        
        incident_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Log to JSON
        incident = {
            'id': incident_id,
            'timestamp': timestamp,
            'camera_id': camera_id,
            'location': location,
            'latitude': latitude,
            'longitude': longitude,
            'confidence': confidence,
            'detections': detections,
            'image_path': image_path,
            'video_clip_path': video_clip_path,
            'alert_sent': alert_sent,
            'alert_type': alert_type,
            'status': 'pending',
            'notes': notes
        }
        
        self.incidents.append(incident)
        self._save_json_log()
        
        print(f"🚨 Incident #{incident_id} logged: {camera_id} at {timestamp}")
        
        return incident_id
    
    def update_incident_status(
        self,
        incident_id: int,
        status: str,
        response_time: Optional[str] = None,
        notes: Optional[str] = None
    ):
        """Update incident status (e.g., 'pending', 'responded', 'resolved', 'false_positive')"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE incidents 
            SET status = ?, response_time = ?, notes = ?
            WHERE id = ?
        ''', (status, response_time, notes, incident_id))
        
        conn.commit()
        conn.close()
        
        # Update JSON
        for incident in self.incidents:
            if incident['id'] == incident_id:
                incident['status'] = status
                if response_time:
                    incident['response_time'] = response_time
                if notes:
                    incident['notes'] = notes
                break
        
        self._save_json_log()
        print(f"✓ Incident #{incident_id} updated: {status}")
    
    def get_incidents(
        self,
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict]:
        """Query incidents with filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM incidents WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        if min_confidence:
            query += " AND confidence >= ?"
            params.append(min_confidence)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_statistics(self, time_period: str = "all") -> Dict:
        """Get incident statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Time filter
        time_filter = ""
        if time_period == "today":
            time_filter = f"WHERE date(timestamp) = date('now')"
        elif time_period == "week":
            time_filter = f"WHERE timestamp >= datetime('now', '-7 days')"
        elif time_period == "month":
            time_filter = f"WHERE timestamp >= datetime('now', '-30 days')"
        
        # Total incidents
        cursor.execute(f"SELECT COUNT(*) FROM incidents {time_filter}")
        total = cursor.fetchone()[0]
        
        # By status
        cursor.execute(f"SELECT status, COUNT(*) FROM incidents {time_filter} GROUP BY status")
        by_status = dict(cursor.fetchall())
        
        # By camera
        cursor.execute(f"SELECT camera_id, COUNT(*) FROM incidents {time_filter} GROUP BY camera_id")
        by_camera = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute(f"SELECT AVG(confidence) FROM incidents {time_filter}")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        # High confidence incidents
        cursor.execute(f"SELECT COUNT(*) FROM incidents {time_filter} WHERE confidence >= 0.7")
        high_confidence = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_incidents': total,
            'by_status': by_status,
            'by_camera': by_camera,
            'avg_confidence': round(avg_confidence, 3),
            'high_confidence_count': high_confidence,
            'time_period': time_period
        }
    
    def export_to_csv(self, output_file: str, filters: Optional[Dict] = None):
        """Export incidents to CSV"""
        incidents = self.get_incidents(**(filters or {}))
        
        if not incidents:
            print("No incidents to export")
            return
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = incidents[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(incidents)
        
        print(f"📄 Exported {len(incidents)} incidents to {output_file}")
    
    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """Generate daily incident report"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        incidents = self.get_incidents(
            start_date=f"{date} 00:00:00",
            end_date=f"{date} 23:59:59"
        )
        
        stats = self.get_statistics("today")
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║        DAILY GRAFFITI VANDALISM DETECTION REPORT               ║
║                    {date}                             ║
╚════════════════════════════════════════════════════════════════╝

📊 SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Incidents: {stats['total_incidents']}
High Confidence (>70%): {stats['high_confidence_count']}
Average Confidence: {stats['avg_confidence']:.1%}

📈 BY STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for status, count in stats['by_status'].items():
            report += f"{status.upper()}: {count}\n"
        
        report += f"""
📹 BY CAMERA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for camera, count in stats['by_camera'].items():
            report += f"{camera}: {count}\n"
        
        report += f"""
🚨 RECENT INCIDENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for incident in incidents[:10]:
            report += f"#{incident['id']} | {incident['timestamp'][:19]} | {incident['camera_id']} | Conf: {incident['confidence']:.0%} | {incident['status']}\n"
        
        report += "\n" + "="*64 + "\n"
        
        return report
    
    def _save_json_log(self):
        """Save incidents to JSON file"""
        with open(self.json_log, 'w') as f:
            json.dump(self.incidents, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Incident Logger Management")
    parser.add_argument("--action", choices=['stats', 'export', 'report'], required=True)
    parser.add_argument("--period", choices=['today', 'week', 'month', 'all'], default='all')
    parser.add_argument("--output", type=str, help="Output file for export")
    
    args = parser.parse_args()
    
    logger = IncidentLogger()
    
    if args.action == 'stats':
        stats = logger.get_statistics(args.period)
        print(json.dumps(stats, indent=2))
    
    elif args.action == 'export':
        if not args.output:
            args.output = f"incidents_export_{datetime.now().strftime('%Y%m%d')}.csv"
        logger.export_to_csv(args.output)
    
    elif args.action == 'report':
        report = logger.generate_daily_report()
        print(report)


if __name__ == "__main__":
    main()
