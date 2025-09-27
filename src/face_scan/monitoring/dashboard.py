"""
Monitoring dashboard module for real-time system monitoring.

Provides web-based dashboard for viewing metrics, alerts, and system health.
"""

import json
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from dataclasses import asdict
import logging


class MonitoringDashboard:
    """Web-based monitoring dashboard."""
    
    def __init__(self, metrics_collector, health_checker, alert_manager, 
                 host: str = '0.0.0.0', port: int = 8080):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.alert_manager = alert_manager
        self.host = host
        self.port = port
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'monitoring-dashboard-secret'
        
        # Register routes
        self._register_routes()
        
        # Dashboard data cache
        self.cache_lock = threading.Lock()
        self.cache_ttl = 30  # seconds
        self.cached_data = {}
        self.cache_timestamps = {}
    
    def _register_routes(self):
        """Register dashboard routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current metrics."""
            return jsonify(self._get_cached_data('metrics', self._fetch_metrics))
        
        @self.app.route('/api/health')
        def get_health():
            """Get health status."""
            return jsonify(self._get_cached_data('health', self._fetch_health))
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get active alerts."""
            return jsonify(self._get_cached_data('alerts', self._fetch_alerts))
        
        @self.app.route('/api/alerts/history')
        def get_alert_history():
            """Get alert history."""
            limit = request.args.get('limit', 100, type=int)
            return jsonify(self._get_alert_history(limit))
        
        @self.app.route('/api/metrics/history')
        def get_metrics_history():
            """Get metrics history."""
            metric_name = request.args.get('metric', '')
            duration_minutes = request.args.get('duration', 60, type=int)
            
            if not metric_name:
                return jsonify({'error': 'Metric name required'}), 400
            
            history = self.metrics_collector.get_metric_history(metric_name, duration_minutes)
            return jsonify([asdict(point) for point in history])
        
        @self.app.route('/api/health/history')
        def get_health_history():
            """Get health check history."""
            check_name = request.args.get('check', '')
            limit = request.args.get('limit', 50, type=int)
            
            history = self.health_checker.get_health_history(check_name, limit)
            return jsonify([asdict(check) for check in history])
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """Acknowledge an alert."""
            data = request.get_json()
            acknowledged_by = data.get('acknowledged_by', 'unknown')
            
            success = self.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
            if success:
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Alert not found'}), 404
        
        @self.app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """Resolve an alert."""
            success = self.alert_manager.resolve_alert(alert_id)
            if success:
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Alert not found'}), 404
        
        @self.app.route('/api/system/info')
        def get_system_info():
            """Get system information."""
            return jsonify(self._get_system_info())
    
    def _get_cached_data(self, key: str, fetch_func):
        """Get cached data or fetch new data if cache is expired."""
        with self.cache_lock:
            now = datetime.now()
            
            # Check if cache is valid
            if (key in self.cached_data and 
                key in self.cache_timestamps and
                (now - self.cache_timestamps[key]).seconds < self.cache_ttl):
                return self.cached_data[key]
            
            # Fetch new data
            data = fetch_func()
            self.cached_data[key] = data
            self.cache_timestamps[key] = now
            
            return data
    
    def _fetch_metrics(self):
        """Fetch current metrics."""
        return self.metrics_collector.get_all_metrics()
    
    def _fetch_health(self):
        """Fetch health status."""
        return self.health_checker.get_health_summary()
    
    def _fetch_alerts(self):
        """Fetch active alerts."""
        alerts = self.alert_manager.get_active_alerts()
        return [asdict(alert) for alert in alerts]
    
    def _get_alert_history(self, limit: int):
        """Get alert history."""
        history = self.alert_manager.get_alert_history(limit)
        return [asdict(alert) for alert in history]
    
    def _get_system_info(self):
        """Get system information."""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'uptime_seconds': self.metrics_collector.system_metrics.uptime_seconds
        }
    
    def run(self, debug: bool = False):
        """Run the dashboard server."""
        logging.info(f"Starting monitoring dashboard on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


class DashboardWidget:
    """Base class for dashboard widgets."""
    
    def __init__(self, title: str, widget_type: str):
        self.title = title
        self.widget_type = widget_type
        self.data = {}
        self.last_updated = None
    
    def update_data(self, data: Dict[str, Any]):
        """Update widget data."""
        self.data = data
        self.last_updated = datetime.now()
    
    def get_data(self) -> Dict[str, Any]:
        """Get widget data."""
        return {
            'title': self.title,
            'type': self.widget_type,
            'data': self.data,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class MetricsWidget(DashboardWidget):
    """Widget for displaying metrics."""
    
    def __init__(self, title: str, metric_name: str, chart_type: str = 'line'):
        super().__init__(title, 'metrics')
        self.metric_name = metric_name
        self.chart_type = chart_type
    
    def update_from_collector(self, metrics_collector):
        """Update widget data from metrics collector."""
        if self.chart_type == 'gauge':
            value = metrics_collector.get_gauge(self.metric_name)
            self.update_data({'value': value})
        elif self.chart_type == 'counter':
            value = metrics_collector.get_counter(self.metric_name)
            self.update_data({'value': value})
        elif self.chart_type == 'timer':
            stats = metrics_collector.get_timer_stats(self.metric_name)
            self.update_data(stats)


class HealthWidget(DashboardWidget):
    """Widget for displaying health status."""
    
    def __init__(self, title: str, check_name: str):
        super().__init__(title, 'health')
        self.check_name = check_name
    
    def update_from_checker(self, health_checker):
        """Update widget data from health checker."""
        result = health_checker.run_check(self.check_name)
        if result:
            self.update_data({
                'status': result.status.value,
                'message': result.message,
                'details': result.details
            })


class AlertsWidget(DashboardWidget):
    """Widget for displaying alerts."""
    
    def __init__(self, title: str, alert_type: str = None):
        super().__init__(title, 'alerts')
        self.alert_type = alert_type
    
    def update_from_manager(self, alert_manager):
        """Update widget data from alert manager."""
        if self.alert_type:
            alerts = alert_manager.get_alerts_by_type(self.alert_type)
        else:
            alerts = alert_manager.get_active_alerts()
        
        self.update_data({
            'alerts': [asdict(alert) for alert in alerts],
            'count': len(alerts)
        })


class DashboardBuilder:
    """Builder for creating custom dashboards."""
    
    def __init__(self):
        self.widgets: List[DashboardWidget] = []
        self.layout = []
    
    def add_metrics_widget(self, title: str, metric_name: str, 
                          chart_type: str = 'line', position: tuple = None):
        """Add a metrics widget."""
        widget = MetricsWidget(title, metric_name, chart_type)
        self.widgets.append(widget)
        
        if position:
            self.layout.append({
                'widget': len(self.widgets) - 1,
                'position': position,
                'size': (1, 1)
            })
        
        return self
    
    def add_health_widget(self, title: str, check_name: str, position: tuple = None):
        """Add a health widget."""
        widget = HealthWidget(title, check_name)
        self.widgets.append(widget)
        
        if position:
            self.layout.append({
                'widget': len(self.widgets) - 1,
                'position': position,
                'size': (1, 1)
            })
        
        return self
    
    def add_alerts_widget(self, title: str, alert_type: str = None, position: tuple = None):
        """Add an alerts widget."""
        widget = AlertsWidget(title, alert_type)
        self.widgets.append(widget)
        
        if position:
            self.layout.append({
                'widget': len(self.widgets) - 1,
                'position': position,
                'size': (1, 1)
            })
        
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the dashboard configuration."""
        return {
            'widgets': [widget.get_data() for widget in self.widgets],
            'layout': self.layout
        }
