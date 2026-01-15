"""
Unit tests for alert system
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np

from src.utils.alerts import (
    AlertManager,
    EmailAlert,
    WebhookAlert,
    DiscordAlert,
    SlackAlert
)


class TestAlertManager(unittest.TestCase):
    """Test alert manager"""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization"""
        config = {
            'email': {'enabled': False},
            'sms': {'enabled': False},
            'webhook': {'enabled': False},
            'discord': {'enabled': False},
            'slack': {'enabled': False}
        }
        
        manager = AlertManager(config)
        
        self.assertIsInstance(manager, AlertManager)
        self.assertEqual(len(manager.alert_channels), 0)
    
    def test_alert_manager_with_enabled_channels(self):
        """Test alert manager with enabled channels"""
        config = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.test.com',
                'smtp_port': 587,
                'username': 'test@test.com',
                'password': 'password',
                'from_email': 'test@test.com',
                'to_emails': ['recipient@test.com']
            },
            'webhook': {
                'enabled': True,
                'url': 'https://test.com/webhook',
                'headers': {}
            }
        }
        
        manager = AlertManager(config)
        
        self.assertGreater(len(manager.alert_channels), 0)


class TestWebhookAlert(unittest.TestCase):
    """Test webhook alerts"""
    
    @patch('src.utils.alerts.requests.post')
    def test_webhook_alert_send(self, mock_post):
        """Test webhook alert sending"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        config = {
            'url': 'https://test.com/webhook',
            'headers': {'Authorization': 'Bearer test'},
            'include_image': False
        }
        
        alert = WebhookAlert(config)
        
        detection_data = {
            'camera_id': 'test_camera',
            'timestamp': '2026-01-03 12:00:00',
            'detections': 1,
            'confidences': [0.85]
        }
        
        alert.send(detection_data)
        
        self.assertTrue(mock_post.called)
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], 'https://test.com/webhook')


class TestDiscordAlert(unittest.TestCase):
    """Test Discord alerts"""
    
    @patch('src.utils.alerts.requests.post')
    def test_discord_alert_send(self, mock_post):
        """Test Discord alert sending"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        config = {
            'webhook_url': 'https://discord.com/api/webhooks/test',
            'include_image': False
        }
        
        alert = DiscordAlert(config)
        
        detection_data = {
            'camera_id': 'test_camera',
            'timestamp': '2026-01-03 12:00:00',
            'detections': 2,
            'confidences': [0.85, 0.76]
        }
        
        alert.send(detection_data)
        
        self.assertTrue(mock_post.called)


class TestSlackAlert(unittest.TestCase):
    """Test Slack alerts"""
    
    @patch('src.utils.alerts.requests.post')
    def test_slack_alert_send(self, mock_post):
        """Test Slack alert sending"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        config = {
            'webhook_url': 'https://hooks.slack.com/services/test',
            'channel': '#security',
            'include_image': False
        }
        
        alert = SlackAlert(config)
        
        detection_data = {
            'camera_id': 'test_camera',
            'timestamp': '2026-01-03 12:00:00',
            'detections': 1,
            'confidences': [0.92]
        }
        
        alert.send(detection_data)
        
        self.assertTrue(mock_post.called)


if __name__ == '__main__':
    unittest.main()
