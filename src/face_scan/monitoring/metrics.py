"""
Metrics collection and monitoring module.

Provides comprehensive metrics collection for performance monitoring and analytics.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import logging


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""
    request_count: int = 0
    request_duration_ms: float = 0.0
    face_detection_time_ms: float = 0.0
    face_recognition_time_ms: float = 0.0
    image_processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_bytes: int = 0
    process_count: int = 0
    load_average: float = 0.0
    uptime_seconds: float = 0.0


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.system_metrics = SystemMetrics()
        
        # Background monitoring
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self, interval: float = 30.0):
        """Start background monitoring of system metrics."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
    
    def _monitor_system_metrics(self, interval: float):
        """Background thread to collect system metrics."""
        while not self._stop_monitoring.wait(interval):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system metrics."""
        with self.lock:
            # CPU and Memory
            self.system_metrics.cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.system_metrics.memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.system_metrics.disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            self.system_metrics.network_io_bytes = network.bytes_sent + network.bytes_recv
            
            # Process count
            self.system_metrics.process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                self.system_metrics.load_average = load_avg[0]
            except AttributeError:
                self.system_metrics.load_average = 0.0
            
            # Uptime
            self.system_metrics.uptime_seconds = time.time() - self.start_time
            
            # Record metrics
            self.record_gauge('system.cpu_percent', self.system_metrics.cpu_percent)
            self.record_gauge('system.memory_percent', self.system_metrics.memory_percent)
            self.record_gauge('system.disk_usage_percent', self.system_metrics.disk_usage_percent)
            self.record_gauge('system.network_io_bytes', self.system_metrics.network_io_bytes)
            self.record_gauge('system.process_count', self.system_metrics.process_count)
            self.record_gauge('system.load_average', self.system_metrics.load_average)
            self.record_gauge('system.uptime_seconds', self.system_metrics.uptime_seconds)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            self._record_metric(name, self.counters[name], tags or {})
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        with self.lock:
            self.gauges[name] = value
            self._record_metric(name, value, tags or {})
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        with self.lock:
            self.timers[name].append(duration_ms)
            # Keep only recent measurements
            if len(self.timers[name]) > 100:
                self.timers[name] = self.timers[name][-100:]
            
            self._record_metric(name, duration_ms, tags or {})
    
    def _record_metric(self, name: str, value: float, tags: Dict[str, str]):
        """Record a metric point in history."""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags
        )
        self.metrics_history[name].append(metric_point)
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self.counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        values = self.timers.get(name, [])
        if not values:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'p95': 0, 'p99': 0}
        
        values.sort()
        count = len(values)
        min_val = values[0]
        max_val = values[-1]
        avg_val = sum(values) / count
        p95_idx = int(count * 0.95)
        p99_idx = int(count * 0.99)
        
        return {
            'count': count,
            'min': min_val,
            'max': max_val,
            'avg': avg_val,
            'p95': values[p95_idx] if p95_idx < count else max_val,
            'p99': values[p99_idx] if p99_idx < count else max_val
        }
    
    def get_metric_history(self, name: str, duration_minutes: int = 60) -> List[MetricPoint]:
        """Get metric history for a specific duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        history = self.metrics_history.get(name, deque())
        
        return [point for point in history if point.timestamp >= cutoff_time]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'timers': {name: self.get_timer_stats(name) for name in self.timers},
                'performance': {
                    'request_count': self.performance_metrics.request_count,
                    'request_duration_ms': self.performance_metrics.request_duration_ms,
                    'face_detection_time_ms': self.performance_metrics.face_detection_time_ms,
                    'face_recognition_time_ms': self.performance_metrics.face_recognition_time_ms,
                    'image_processing_time_ms': self.performance_metrics.image_processing_time_ms,
                    'memory_usage_mb': self.performance_metrics.memory_usage_mb,
                    'cpu_usage_percent': self.performance_metrics.cpu_usage_percent,
                    'error_count': self.performance_metrics.error_count,
                    'success_rate': self.performance_metrics.success_rate
                },
                'system': {
                    'cpu_percent': self.system_metrics.cpu_percent,
                    'memory_percent': self.system_metrics.memory_percent,
                    'disk_usage_percent': self.system_metrics.disk_usage_percent,
                    'network_io_bytes': self.system_metrics.network_io_bytes,
                    'process_count': self.system_metrics.process_count,
                    'load_average': self.system_metrics.load_average,
                    'uptime_seconds': self.system_metrics.uptime_seconds
                }
            }
    
    def record_request_metrics(self, duration_ms: float, success: bool = True):
        """Record request-related metrics."""
        with self.lock:
            self.performance_metrics.request_count += 1
            self.performance_metrics.request_duration_ms = duration_ms
            
            if not success:
                self.performance_metrics.error_count += 1
            
            # Calculate success rate
            total_requests = self.performance_metrics.request_count
            error_count = self.performance_metrics.error_count
            self.performance_metrics.success_rate = ((total_requests - error_count) / total_requests) * 100
            
            # Record metrics
            self.increment_counter('requests.total')
            self.record_timer('requests.duration_ms', duration_ms)
            
            if success:
                self.increment_counter('requests.success')
            else:
                self.increment_counter('requests.error')
            
            self.record_gauge('requests.success_rate', self.performance_metrics.success_rate)
    
    def record_face_detection_metrics(self, duration_ms: float, face_count: int):
        """Record face detection metrics."""
        with self.lock:
            self.performance_metrics.face_detection_time_ms = duration_ms
            
            self.record_timer('face_detection.duration_ms', duration_ms)
            self.increment_counter('face_detection.total')
            self.record_gauge('face_detection.face_count', face_count)
    
    def record_face_recognition_metrics(self, duration_ms: float, match_found: bool):
        """Record face recognition metrics."""
        with self.lock:
            self.performance_metrics.face_recognition_time_ms = duration_ms
            
            self.record_timer('face_recognition.duration_ms', duration_ms)
            self.increment_counter('face_recognition.total')
            
            if match_found:
                self.increment_counter('face_recognition.matches')
            else:
                self.increment_counter('face_recognition.no_matches')
    
    def record_image_processing_metrics(self, duration_ms: float, image_size_mb: float):
        """Record image processing metrics."""
        with self.lock:
            self.performance_metrics.image_processing_time_ms = duration_ms
            
            self.record_timer('image_processing.duration_ms', duration_ms)
            self.increment_counter('image_processing.total')
            self.record_gauge('image_processing.image_size_mb', image_size_mb)
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        metrics = self.get_all_metrics()
        
        if format == 'json':
            return json.dumps(metrics, indent=2, default=str)
        elif format == 'prometheus':
            return self._export_prometheus_format(metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Counters
        for name, value in metrics['counters'].items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Gauges
        for name, value in metrics['gauges'].items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Timers (as histograms)
        for name, stats in metrics['timers'].items():
            lines.append(f"# TYPE {name}_count counter")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"# TYPE {name}_sum counter")
            lines.append(f"{name}_sum {stats['avg'] * stats['count']}")
            lines.append(f"# TYPE {name}_avg gauge")
            lines.append(f"{name}_avg {stats['avg']}")
        
        return '\n'.join(lines)


class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str, tags: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.metrics_collector.record_timer(self.metric_name, duration_ms, self.tags)


def time_operation(metrics_collector: MetricsCollector, metric_name: str, tags: Dict[str, str] = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MetricsTimer(metrics_collector, metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator
