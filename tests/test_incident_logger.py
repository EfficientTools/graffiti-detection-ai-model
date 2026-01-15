"""
Unit tests for incident logging system
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.incident_logger import IncidentLogger


class TestIncidentLogger(unittest.TestCase):
    """Test incident logging system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = IncidentLogger(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        self.assertIsNotNone(self.logger)
        self.assertTrue(self.logger.log_dir.exists())
        self.assertTrue(self.logger.db_path.exists())
    
    def test_log_incident(self):
        """Test logging an incident"""
        incident_id = self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.85,
            detections=2,
            location='Building A',
            latitude=40.7128,
            longitude=-74.0060,
            alert_sent=True,
            alert_type='email',
            notes='Test incident'
        )
        
        self.assertIsInstance(incident_id, int)
        self.assertGreater(incident_id, 0)
    
    def test_update_incident_status(self):
        """Test updating incident status"""
        # Log incident
        incident_id = self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.75,
            detections=1
        )
        
        # Update status
        self.logger.update_incident_status(
            incident_id=incident_id,
            status='resolved',
            response_time='00:15:30',
            notes='Security responded'
        )
        
        # Verify update
        incidents = self.logger.get_incidents(status='resolved')
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0]['id'], incident_id)
        self.assertEqual(incidents[0]['status'], 'resolved')
    
    def test_get_incidents_with_filters(self):
        """Test getting incidents with filters"""
        # Log multiple incidents
        self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.85,
            detections=1
        )
        self.logger.log_incident(
            camera_id='camera_2',
            confidence=0.65,
            detections=2
        )
        self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.95,
            detections=1
        )
        
        # Filter by camera
        camera1_incidents = self.logger.get_incidents(camera_id='camera_1')
        self.assertEqual(len(camera1_incidents), 2)
        
        # Filter by confidence
        high_conf_incidents = self.logger.get_incidents(min_confidence=0.8)
        self.assertGreaterEqual(len(high_conf_incidents), 2)
    
    def test_get_statistics(self):
        """Test statistics generation"""
        # Log incidents
        self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.85,
            detections=1
        )
        self.logger.log_incident(
            camera_id='camera_2',
            confidence=0.75,
            detections=2
        )
        
        stats = self.logger.get_statistics(time_period='all')
        
        self.assertIn('total_incidents', stats)
        self.assertIn('by_status', stats)
        self.assertIn('by_camera', stats)
        self.assertIn('avg_confidence', stats)
        
        self.assertEqual(stats['total_incidents'], 2)
        self.assertGreater(stats['avg_confidence'], 0)
    
    def test_export_to_csv(self):
        """Test CSV export"""
        # Log incidents
        self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.85,
            detections=1
        )
        
        # Export
        csv_path = Path(self.temp_dir) / 'export.csv'
        self.logger.export_to_csv(str(csv_path))
        
        self.assertTrue(csv_path.exists())
        
        # Verify content
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 1)  # Header + data
    
    def test_generate_daily_report(self):
        """Test daily report generation"""
        # Log incident
        self.logger.log_incident(
            camera_id='camera_1',
            confidence=0.85,
            detections=1
        )
        
        report = self.logger.generate_daily_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('DAILY GRAFFITI VANDALISM DETECTION REPORT', report)
        self.assertIn('SUMMARY', report)
        self.assertIn('camera_1', report)


class TestIncidentLoggerEdgeCases(unittest.TestCase):
    """Test edge cases in incident logger"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = IncidentLogger(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_empty_database(self):
        """Test operations on empty database"""
        incidents = self.logger.get_incidents()
        self.assertEqual(len(incidents), 0)
        
        stats = self.logger.get_statistics()
        self.assertEqual(stats['total_incidents'], 0)
    
    def test_invalid_incident_id(self):
        """Test updating non-existent incident"""
        # Should not crash
        self.logger.update_incident_status(
            incident_id=99999,
            status='resolved'
        )


if __name__ == '__main__':
    unittest.main()
