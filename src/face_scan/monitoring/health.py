"""
Health check module for monitoring system health and status.

Provides comprehensive health checking for all system components.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Represents a health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    duration_ms: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}


class HealthChecker:
    """Manages health checks for system components."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: List[HealthCheck] = []
        self.max_history_size = 100
        self.lock = threading.Lock()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("cpu_usage", self._check_cpu_usage)
        self.register_check("process_health", self._check_process_health)
    
    def register_check(self, name: str, check_function: Callable[[], HealthCheck]):
        """Register a health check function."""
        self.health_checks[name] = check_function
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
    
    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        try:
            start_time = time.time()
            result = self.health_checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            
            with self.lock:
                self.health_history.append(result)
                if len(self.health_history) > self.max_history_size:
                    self.health_history.pop(0)
            
            return result
        except Exception as e:
            error_result = HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )
            error_result.duration_ms = 0.0
            
            with self.lock:
                self.health_history.append(error_result)
            
            return error_result
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.health_checks:
            results[name] = self.run_check(name)
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        results = self.run_all_checks()
        overall_status = self.get_overall_health()
        
        summary = {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        for name, result in results.items():
            summary["checks"][name] = {
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "duration_ms": result.duration_ms
            }
        
        return summary
    
    def get_health_history(self, check_name: Optional[str] = None, limit: int = 50) -> List[HealthCheck]:
        """Get health check history."""
        with self.lock:
            history = self.health_history.copy()
        
        if check_name:
            history = [check for check in history if check.name == check_name]
        
        return history[-limit:] if limit > 0 else history
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = "System resources critically high"
            elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 85:
                status = HealthStatus.WARNING
                message = "System resources usage high"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            free_gb = disk.free / (1024**3)
            
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk space critically low: {usage_percent:.1f}% used"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = f"Disk space low: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal: {usage_percent:.1f}% used"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "free_gb": free_gb,
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}"
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            available_gb = memory.available / (1024**3)
            
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critically high: {usage_percent:.1f}%"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent:.1f}%"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "usage_percent": usage_percent,
                    "available_gb": available_gb,
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}"
            )
    
    def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            if cpu_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critically high: {cpu_percent:.1f}%"
            elif cpu_percent > 85:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_average": load_avg
                }
            )
        except Exception as e:
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU usage: {str(e)}"
            )
    
    def _check_process_health(self) -> HealthCheck:
        """Check process health."""
        try:
            current_process = psutil.Process()
            process_count = len(psutil.pids())
            
            # Check current process
            cpu_percent = current_process.cpu_percent()
            memory_info = current_process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            
            # Check if process is responsive
            try:
                current_process.status()
                process_status = "running"
            except psutil.NoSuchProcess:
                process_status = "not_found"
            except Exception:
                process_status = "unknown"
            
            if process_status != "running":
                status = HealthStatus.CRITICAL
                message = f"Process health critical: {process_status}"
            elif memory_mb > 1000:  # More than 1GB
                status = HealthStatus.WARNING
                message = f"Process memory usage high: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process health normal: {memory_mb:.1f}MB memory"
            
            return HealthCheck(
                name="process_health",
                status=status,
                message=message,
                details={
                    "process_status": process_status,
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent,
                    "total_processes": process_count,
                    "pid": current_process.pid
                }
            )
        except Exception as e:
            return HealthCheck(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check process health: {str(e)}"
            )


class DatabaseHealthCheck:
    """Health check for database connectivity."""
    
    def __init__(self, database_service):
        self.database_service = database_service
    
    def check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            # Test basic connectivity
            connection = self.database_service.get_connection()
            if not connection:
                return HealthCheck(
                    name="database_connectivity",
                    status=HealthStatus.CRITICAL,
                    message="Cannot establish database connection"
                )
            
            # Test query performance
            test_query = "SELECT 1"
            cursor = connection.cursor()
            cursor.execute(test_query)
            result = cursor.fetchone()
            cursor.close()
            
            query_time_ms = (time.time() - start_time) * 1000
            
            if query_time_ms > 5000:  # 5 seconds
                status = HealthStatus.CRITICAL
                message = f"Database query very slow: {query_time_ms:.1f}ms"
            elif query_time_ms > 1000:  # 1 second
                status = HealthStatus.WARNING
                message = f"Database query slow: {query_time_ms:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database connectivity normal: {query_time_ms:.1f}ms"
            
            return HealthCheck(
                name="database_connectivity",
                status=status,
                message=message,
                details={
                    "query_time_ms": query_time_ms,
                    "connection_active": True
                }
            )
        except Exception as e:
            return HealthCheck(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)}
            )


class ModelHealthCheck:
    """Health check for ML models."""
    
    def __init__(self, face_detector, face_recognizer):
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
    
    def check_model_health(self) -> HealthCheck:
        """Check ML model availability and performance."""
        try:
            start_time = time.time()
            
            # Test face detection model
            import numpy as np
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            try:
                faces = self.face_detector.detect_faces(test_image)
                detection_working = True
            except Exception as e:
                detection_working = False
                detection_error = str(e)
            
            # Test face recognition model
            try:
                if detection_working and len(faces) > 0:
                    # Create dummy encoding
                    dummy_encoding = np.random.rand(128)
                    self.face_recognizer.compare_faces([dummy_encoding], dummy_encoding)
                recognition_working = True
            except Exception as e:
                recognition_working = False
                recognition_error = str(e)
            
            test_time_ms = (time.time() - start_time) * 1000
            
            if not detection_working or not recognition_working:
                status = HealthStatus.CRITICAL
                message = "ML models not functioning properly"
                details = {
                    "detection_working": detection_working,
                    "recognition_working": recognition_working,
                    "test_time_ms": test_time_ms
                }
                if not detection_working:
                    details["detection_error"] = detection_error
                if not recognition_working:
                    details["recognition_error"] = recognition_error
            elif test_time_ms > 1000:
                status = HealthStatus.WARNING
                message = f"ML models responding slowly: {test_time_ms:.1f}ms"
                details = {
                    "detection_working": detection_working,
                    "recognition_working": recognition_working,
                    "test_time_ms": test_time_ms
                }
            else:
                status = HealthStatus.HEALTHY
                message = f"ML models functioning normally: {test_time_ms:.1f}ms"
                details = {
                    "detection_working": detection_working,
                    "recognition_working": recognition_working,
                    "test_time_ms": test_time_ms
                }
            
            return HealthCheck(
                name="model_health",
                status=status,
                message=message,
                details=details
            )
        except Exception as e:
            return HealthCheck(
                name="model_health",
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {str(e)}",
                details={"error": str(e)}
            )
