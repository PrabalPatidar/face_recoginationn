"""
Alert management module for monitoring and notification system.

Provides alerting capabilities for system events, performance issues, and security incidents.
"""

import smtplib
import requests
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging


class AlertType(Enum):
    """Types of alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SYSTEM = "system"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Represents an alert."""
    id: str
    type: AlertType
    title: str
    message: str
    severity: AlertType
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AlertRule:
    """Represents an alert rule."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    alert_type: AlertType
    title: str
    message_template: str
    enabled: bool = True
    cooldown_minutes: int = 15
    last_triggered: Optional[datetime] = None


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[AlertRule] = []
        self.notification_channels: Dict[str, Callable] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        self.lock = threading.Lock()
        
        # Register default notification channels
        self._register_default_channels()
        
        # Register default alert rules
        self._register_default_rules()
    
    def _register_default_channels(self):
        """Register default notification channels."""
        self.register_channel("email", self._send_email_notification)
        self.register_channel("webhook", self._send_webhook_notification)
        self.register_channel("log", self._log_notification)
    
    def _register_default_rules(self):
        """Register default alert rules."""
        # High CPU usage
        self.add_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda metrics: metrics.get('cpu_percent', 0) > 90,
            alert_type=AlertType.CRITICAL,
            title="High CPU Usage",
            message_template="CPU usage is critically high: {cpu_percent}%"
        ))
        
        # High memory usage
        self.add_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda metrics: metrics.get('memory_percent', 0) > 90,
            alert_type=AlertType.CRITICAL,
            title="High Memory Usage",
            message_template="Memory usage is critically high: {memory_percent}%"
        ))
        
        # Low disk space
        self.add_rule(AlertRule(
            name="low_disk_space",
            condition=lambda metrics: metrics.get('disk_percent', 0) > 95,
            alert_type=AlertType.CRITICAL,
            title="Low Disk Space",
            message_template="Disk space is critically low: {disk_percent}% used"
        ))
        
        # High error rate
        self.add_rule(AlertRule(
            name="high_error_rate",
            condition=lambda metrics: metrics.get('error_rate', 0) > 10,
            alert_type=AlertType.ERROR,
            title="High Error Rate",
            message_template="Error rate is high: {error_rate}%"
        ))
        
        # Slow response time
        self.add_rule(AlertRule(
            name="slow_response_time",
            condition=lambda metrics: metrics.get('avg_response_time_ms', 0) > 5000,
            alert_type=AlertType.WARNING,
            title="Slow Response Time",
            message_template="Average response time is slow: {avg_response_time_ms}ms"
        ))
    
    def register_channel(self, name: str, notification_func: Callable):
        """Register a notification channel."""
        self.notification_channels[name] = notification_func
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [rule for rule in self.alert_rules if rule.name != rule_name]
    
    def create_alert(self, alert_type: AlertType, title: str, message: str, 
                    source: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{alert_type.value}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            title=title,
            message=message,
            severity=alert_type,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Maintain history size
            if len(self.alert_history) > self.max_history_size:
                self.alert_history.pop(0)
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                return True
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert."""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.type == alert_type]
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            return self.alert_history[-limit:] if limit > 0 else self.alert_history.copy()
    
    def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules."""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue
            
            # Check condition
            try:
                if rule.condition(metrics):
                    # Create alert
                    message = rule.message_template.format(**metrics)
                    alert = self.create_alert(
                        alert_type=rule.alert_type,
                        title=rule.title,
                        message=message,
                        source="metrics_check",
                        metadata={"rule_name": rule.name, "metrics": metrics}
                    )
                    
                    # Update last triggered time
                    rule.last_triggered = datetime.now()
                    
                    logging.info(f"Alert triggered: {rule.name} - {alert.title}")
            except Exception as e:
                logging.error(f"Error checking alert rule {rule.name}: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for channel_name, notification_func in self.notification_channels.items():
            try:
                notification_func(alert)
            except Exception as e:
                logging.error(f"Failed to send notification via {channel_name}: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        # This would be configured with actual SMTP settings
        # For now, just log the email content
        email_content = f"""
        Alert: {alert.title}
        Type: {alert.type.value}
        Severity: {alert.severity.value}
        Message: {alert.message}
        Source: {alert.source}
        Timestamp: {alert.timestamp}
        """
        logging.info(f"Email notification: {email_content}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        # This would be configured with actual webhook URL
        webhook_data = {
            "alert_id": alert.id,
            "type": alert.type.value,
            "title": alert.title,
            "message": alert.message,
            "severity": alert.severity.value,
            "source": alert.source,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata
        }
        logging.info(f"Webhook notification: {json.dumps(webhook_data)}")
    
    def _log_notification(self, alert: Alert):
        """Log notification."""
        logging.warning(f"ALERT: {alert.title} - {alert.message}")


class EmailNotifier:
    """Email notification service."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert_email(self, alert: Alert, recipients: List[str]):
        """Send alert email to recipients."""
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            =============
            Title: {alert.title}
            Type: {alert.type.value}
            Severity: {alert.severity.value}
            Message: {alert.message}
            Source: {alert.source}
            Timestamp: {alert.timestamp}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logging.info(f"Alert email sent to {recipients}")
        except Exception as e:
            logging.error(f"Failed to send alert email: {e}")


class WebhookNotifier:
    """Webhook notification service."""
    
    def __init__(self, webhook_url: str, timeout: int = 30):
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def send_alert_webhook(self, alert: Alert):
        """Send alert webhook."""
        try:
            payload = {
                "alert_id": alert.id,
                "type": alert.type.value,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            logging.info(f"Alert webhook sent successfully: {alert.id}")
        except Exception as e:
            logging.error(f"Failed to send alert webhook: {e}")


class SlackNotifier:
    """Slack notification service."""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send_alert_slack(self, alert: Alert):
        """Send alert to Slack."""
        try:
            # Color mapping for severity
            color_map = {
                AlertType.INFO: "good",
                AlertType.WARNING: "warning",
                AlertType.ERROR: "danger",
                AlertType.CRITICAL: "danger",
                AlertType.SECURITY: "danger",
                AlertType.PERFORMANCE: "warning",
                AlertType.SYSTEM: "warning"
            }
            
            payload = {
                "channel": self.channel,
                "attachments": [{
                    "color": color_map.get(alert.severity, "warning"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Type", "value": alert.type.value, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": "Face Scan Project",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            logging.info(f"Alert Slack notification sent: {alert.id}")
        except Exception as e:
            logging.error(f"Failed to send Slack notification: {e}")


# Global alert manager instance
alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager


def create_alert(alert_type: AlertType, title: str, message: str, 
                source: str, metadata: Dict[str, Any] = None) -> Alert:
    """Create an alert using the global alert manager."""
    return alert_manager.create_alert(alert_type, title, message, source, metadata)
