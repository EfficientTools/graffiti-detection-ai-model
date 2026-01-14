"""
Alert Management System for Graffiti Detection
Supports email, SMS, webhooks, Discord, Slack, and push notifications
"""

import json
import smtplib
import requests
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


class AlertManager:
    """Manage different alert channels for graffiti detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_channels = []
        
        # Initialize enabled channels
        if config.get('email', {}).get('enabled'):
            self.alert_channels.append(EmailAlert(config['email']))
        
        if config.get('sms', {}).get('enabled'):
            self.alert_channels.append(SMSAlert(config['sms']))
        
        if config.get('webhook', {}).get('enabled'):
            self.alert_channels.append(WebhookAlert(config['webhook']))
        
        if config.get('discord', {}).get('enabled'):
            self.alert_channels.append(DiscordAlert(config['discord']))
        
        if config.get('slack', {}).get('enabled'):
            self.alert_channels.append(SlackAlert(config['slack']))
    
    def send_alert(self, detection_data: Dict):
        """Send alert through all enabled channels"""
        for channel in self.alert_channels:
            try:
                channel.send(detection_data)
            except Exception as e:
                print(f"[ERROR] Failed to send alert via {channel.__class__.__name__}: {e}")


class EmailAlert:
    """Send email alerts with detection images"""
    
    def __init__(self, config: Dict):
        self.smtp_server = config['smtp_server']
        self.smtp_port = config['smtp_port']
        self.username = config['username']
        self.password = config['password']
        self.from_email = config['from_email']
        self.to_emails = config['to_emails']
        self.include_image = config.get('include_image', True)
    
    def send(self, detection_data: Dict):
        """Send email alert"""
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        msg['Subject'] = f"🚨 Graffiti Detected - {detection_data['camera_id']}"
        
        # Email body
        body = f"""
        <html>
        <body>
        <h2 style="color: red;">⚠️ GRAFFITI VANDALISM DETECTED</h2>
        <p><strong>Camera:</strong> {detection_data['camera_id']}</p>
        <p><strong>Time:</strong> {detection_data['timestamp']}</p>
        <p><strong>Detections:</strong> {detection_data['detections']}</p>
        <p><strong>Confidence:</strong> {max(detection_data['confidences']):.2%}</p>
        <p><strong>Action Required:</strong> Immediate inspection and response recommended.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach image
        if self.include_image and 'frame' in detection_data:
            _, buffer = cv2.imencode('.jpg', detection_data['frame'])
            image = MIMEImage(buffer.tobytes())
            image.add_header('Content-Disposition', 'attachment', filename='detection.jpg')
            msg.attach(image)
        
        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
        
        print(f"[INFO] Email alert sent to {', '.join(self.to_emails)}")


class SMSAlert:
    """Send SMS alerts via Twilio or other providers"""
    
    def __init__(self, config: Dict):
        self.provider = config['provider']
        self.account_sid = config.get('account_sid')
        self.auth_token = config.get('auth_token')
        self.from_number = config.get('from_number')
        self.to_numbers = config['to_numbers']
    
    def send(self, detection_data: Dict):
        """Send SMS alert"""
        message = (f"🚨 GRAFFITI ALERT\n"
                  f"Camera: {detection_data['camera_id']}\n"
                  f"Time: {detection_data['timestamp']}\n"
                  f"Confidence: {max(detection_data['confidences']):.0%}")
        
        if self.provider == 'twilio':
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            
            for number in self.to_numbers:
                client.messages.create(
                    body=message,
                    from_=self.from_number,
                    to=number
                )
        
        print(f"[INFO] SMS alert sent to {len(self.to_numbers)} recipients")


class WebhookAlert:
    """Send alerts via webhook to external systems"""
    
    def __init__(self, config: Dict):
        self.url = config['url']
        self.headers = config.get('headers', {})
        self.include_image = config.get('include_image', False)
    
    def send(self, detection_data: Dict):
        """Send webhook alert"""
        payload = {
            'event': 'graffiti_detected',
            'camera_id': detection_data['camera_id'],
            'timestamp': detection_data['timestamp'],
            'detections': detection_data['detections'],
            'max_confidence': max(detection_data['confidences'])
        }
        
        if self.include_image and 'frame' in detection_data:
            _, buffer = cv2.imencode('.jpg', detection_data['frame'])
            import base64
            payload['image_base64'] = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=10
        )
        
        response.raise_for_status()
        print(f"[INFO] Webhook alert sent to {self.url}")


class DiscordAlert:
    """Send alerts to Discord channel"""
    
    def __init__(self, config: Dict):
        self.webhook_url = config['webhook_url']
        self.include_image = config.get('include_image', True)
    
    def send(self, detection_data: Dict):
        """Send Discord alert"""
        embed = {
            "title": "🚨 Graffiti Vandalism Detected!",
            "color": 0xFF0000,  # Red
            "fields": [
                {"name": "Camera", "value": detection_data['camera_id'], "inline": True},
                {"name": "Time", "value": detection_data['timestamp'], "inline": True},
                {"name": "Detections", "value": str(detection_data['detections']), "inline": True},
                {"name": "Confidence", "value": f"{max(detection_data['confidences']):.0%}", "inline": True}
            ],
            "footer": {"text": "Graffiti Detection AI - Immediate Action Required"}
        }
        
        payload = {"embeds": [embed]}
        
        # Send without image first
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()
        
        # Send image if enabled
        if self.include_image and 'frame' in detection_data:
            _, buffer = cv2.imencode('.jpg', detection_data['frame'])
            files = {'file': ('detection.jpg', buffer.tobytes(), 'image/jpeg')}
            requests.post(self.webhook_url, files=files)
        
        print(f"[INFO] Discord alert sent")


class SlackAlert:
    """Send alerts to Slack channel"""
    
    def __init__(self, config: Dict):
        self.webhook_url = config['webhook_url']
        self.channel = config.get('channel', '#security')
        self.include_image = config.get('include_image', True)
    
    def send(self, detection_data: Dict):
        """Send Slack alert"""
        payload = {
            "channel": self.channel,
            "text": "🚨 *GRAFFITI VANDALISM DETECTED*",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {"title": "Camera", "value": detection_data['camera_id'], "short": True},
                        {"title": "Time", "value": detection_data['timestamp'], "short": True},
                        {"title": "Detections", "value": str(detection_data['detections']), "short": True},
                        {"title": "Confidence", "value": f"{max(detection_data['confidences']):.0%}", "short": True}
                    ],
                    "footer": "Graffiti Detection AI",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
                }
            ]
        }
        
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()
        
        print(f"[INFO] Slack alert sent to {self.channel}")


class PushNotificationAlert:
    """Send push notifications via services like OneSignal, Firebase, etc."""
    
    def __init__(self, config: Dict):
        self.provider = config['provider']
        self.api_key = config['api_key']
        self.app_id = config.get('app_id')
    
    def send(self, detection_data: Dict):
        """Send push notification"""
        # Implementation depends on provider
        # Example for OneSignal
        if self.provider == 'onesignal':
            payload = {
                "app_id": self.app_id,
                "included_segments": ["All"],
                "headings": {"en": "Graffiti Detected!"},
                "contents": {
                    "en": f"Camera {detection_data['camera_id']} detected graffiti at {detection_data['timestamp']}"
                },
                "priority": 10
            }
            
            headers = {
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://onesignal.com/api/v1/notifications",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            print(f"[INFO] Push notification sent")
