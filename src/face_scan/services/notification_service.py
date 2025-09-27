"""
Notification service for sending alerts and notifications.
"""

import logging
from typing import Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json


class NotificationService:
    """Service for sending notifications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.smtp_server = None
        self.smtp_port = 587
        self.smtp_username = None
        self.smtp_password = None
        self.from_email = None
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, password: str, from_email: str):
        """
        Configure email settings.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = username
        self.smtp_password = password
        self.from_email = from_email
    
    def send_email(self, to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
        """
        Send email notification.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            is_html: Whether body is HTML
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            if not self.smtp_server:
                self.logger.error("Email not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def send_notification(self, message: str, notification_type: str = "info") -> bool:
        """
        Send general notification.
        
        Args:
            message: Notification message
            notification_type: Type of notification (info, warning, error)
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Log notification
            self.logger.info(f"Notification [{notification_type}]: {message}")
            
            # In a real implementation, this could send to various channels:
            # - Email
            # - SMS
            # - Push notifications
            # - Webhooks
            # - Slack/Discord
            # - etc.
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False
    
    def send_face_detection_alert(self, image_path: str, faces_detected: int, confidence: float) -> bool:
        """
        Send face detection alert.
        
        Args:
            image_path: Path to the processed image
            faces_detected: Number of faces detected
            confidence: Detection confidence
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"Face Detection Alert:\n"
            message += f"Image: {image_path}\n"
            message += f"Faces detected: {faces_detected}\n"
            message += f"Confidence: {confidence:.2f}"
            
            return self.send_notification(message, "info")
            
        except Exception as e:
            self.logger.error(f"Failed to send face detection alert: {e}")
            return False
    
    def send_face_recognition_alert(self, image_path: str, recognized_faces: list) -> bool:
        """
        Send face recognition alert.
        
        Args:
            image_path: Path to the processed image
            recognized_faces: List of recognized faces
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"Face Recognition Alert:\n"
            message += f"Image: {image_path}\n"
            message += f"Recognized faces: {len(recognized_faces)}\n"
            
            for face in recognized_faces:
                message += f"- {face.get('name', 'Unknown')} (confidence: {face.get('confidence', 0):.2f})\n"
            
            return self.send_notification(message, "info")
            
        except Exception as e:
            self.logger.error(f"Failed to send face recognition alert: {e}")
            return False
    
    def send_error_alert(self, error_message: str, error_details: str = None) -> bool:
        """
        Send error alert.
        
        Args:
            error_message: Error message
            error_details: Additional error details
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"Error Alert:\n"
            message += f"Error: {error_message}\n"
            
            if error_details:
                message += f"Details: {error_details}\n"
            
            return self.send_notification(message, "error")
            
        except Exception as e:
            self.logger.error(f"Failed to send error alert: {e}")
            return False
    
    def send_performance_alert(self, metric_name: str, value: float, threshold: float) -> bool:
        """
        Send performance alert.
        
        Args:
            metric_name: Name of the performance metric
            value: Current value
            threshold: Threshold value
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = f"Performance Alert:\n"
            message += f"Metric: {metric_name}\n"
            message += f"Current value: {value}\n"
            message += f"Threshold: {threshold}\n"
            message += f"Status: {'Above threshold' if value > threshold else 'Below threshold'}"
            
            return self.send_notification(message, "warning")
            
        except Exception as e:
            self.logger.error(f"Failed to send performance alert: {e}")
            return False
    
    def get_notification_history(self, limit: int = 100) -> list:
        """
        Get notification history.
        
        Args:
            limit: Maximum number of notifications to return
            
        Returns:
            List of recent notifications
        """
        try:
            # In a real implementation, this would query a database
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get notification history: {e}")
            return []
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information.
        
        Returns:
            Dictionary containing service information
        """
        return {
            'service_name': 'NotificationService',
            'version': '1.0.0',
            'email_configured': self.smtp_server is not None,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port,
            'from_email': self.from_email,
            'status': 'active'
        }
