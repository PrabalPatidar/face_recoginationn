"""
Structured logging module for comprehensive application logging.

Provides structured logging with different levels, formatters, and handlers.
"""

import logging
import json
import sys
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum
import traceback


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """Structured logger with JSON formatting and multiple handlers."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.json_formatter = JSONFormatter()
        self.console_formatter = ConsoleFormatter()
        
        # Setup console handler
        self._setup_console_handler()
        
        # Setup file handler if specified
        if log_file:
            self._setup_file_handler(log_file)
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str):
        """Setup file handler with JSON formatting."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(file_handler)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method with structured data."""
        with self.lock:
            extra_data = {
                'timestamp': datetime.now().isoformat(),
                'logger': self.name,
                'level': level.value,
                'message': message,
                **kwargs
            }
            
            # Add thread information
            extra_data['thread_id'] = threading.get_ident()
            extra_data['thread_name'] = threading.current_thread().name
            
            # Log with appropriate level
            log_method = getattr(self.logger, level.value.lower())
            log_method(message, extra=extra_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exception'] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def log_request(self, method: str, path: str, status_code: int, duration_ms: float, **kwargs):
        """Log HTTP request."""
        self.info(
            f"{method} {path} - {status_code}",
            event_type="http_request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_face_detection(self, image_path: str, face_count: int, duration_ms: float, **kwargs):
        """Log face detection event."""
        self.info(
            f"Face detection completed: {face_count} faces found",
            event_type="face_detection",
            image_path=image_path,
            face_count=face_count,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_face_recognition(self, image_path: str, match_found: bool, confidence: float, duration_ms: float, **kwargs):
        """Log face recognition event."""
        self.info(
            f"Face recognition completed: match={match_found}, confidence={confidence:.3f}",
            event_type="face_recognition",
            image_path=image_path,
            match_found=match_found,
            confidence=confidence,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_security_event(self, event_type: str, user_id: Optional[str], ip_address: str, **kwargs):
        """Log security-related event."""
        self.warning(
            f"Security event: {event_type}",
            event_type="security",
            security_event=event_type,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs
        )
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """Log performance metric."""
        self.info(
            f"Performance metric: {metric_name}={value}{unit}",
            event_type="performance",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            **kwargs
        )
    
    def log_system_event(self, event_type: str, **kwargs):
        """Log system event."""
        self.info(
            f"System event: {event_type}",
            event_type="system",
            system_event=event_type,
            **kwargs
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            'timestamp': getattr(record, 'timestamp', datetime.now().isoformat()),
            'level': record.levelname,
            'logger': getattr(record, 'logger', record.name),
            'message': record.getMessage(),
            'thread_id': getattr(record, 'thread_id', threading.get_ident()),
            'thread_name': getattr(record, 'thread_name', threading.current_thread().name),
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record for console with colors."""
        # Get color for log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format message
        message = record.getMessage()
        
        # Add extra fields if present
        extra_fields = []
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                extra_fields.append(f"{key}={value}")
        
        # Build formatted message
        formatted = f"{color}[{timestamp}] {record.levelname:8} {record.name}: {message}{reset}"
        
        if extra_fields:
            formatted += f" | {' '.join(extra_fields)}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class LogManager:
    """Manages multiple loggers and log rotation."""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        self.loggers: Dict[str, StructuredLogger] = {}
        self.lock = threading.Lock()
    
    def get_logger(self, name: str, log_file: Optional[str] = None, level: LogLevel = LogLevel.INFO) -> StructuredLogger:
        """Get or create a logger."""
        with self.lock:
            if name not in self.loggers:
                if log_file is None:
                    log_file = str(self.log_directory / f"{name}.log")
                
                self.loggers[name] = StructuredLogger(name, log_file, level)
            
            return self.loggers[name]
    
    def get_application_logger(self) -> StructuredLogger:
        """Get the main application logger."""
        return self.get_logger("application", "logs/app.log")
    
    def get_security_logger(self) -> StructuredLogger:
        """Get the security logger."""
        return self.get_logger("security", "logs/security.log")
    
    def get_performance_logger(self) -> StructuredLogger:
        """Get the performance logger."""
        return self.get_logger("performance", "logs/performance.log")
    
    def get_error_logger(self) -> StructuredLogger:
        """Get the error logger."""
        return self.get_logger("error", "logs/error.log", LogLevel.ERROR)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_directory.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                    self.get_application_logger().info(f"Cleaned up old log file: {log_file}")
                except Exception as e:
                    self.get_application_logger().error(f"Failed to clean up log file {log_file}: {e}")


# Global log manager instance
log_manager = LogManager()


def get_logger(name: str) -> StructuredLogger:
    """Get a logger by name."""
    return log_manager.get_logger(name)


def get_application_logger() -> StructuredLogger:
    """Get the main application logger."""
    return log_manager.get_application_logger()


def get_security_logger() -> StructuredLogger:
    """Get the security logger."""
    return log_manager.get_security_logger()


def get_performance_logger() -> StructuredLogger:
    """Get the performance logger."""
    return log_manager.get_performance_logger()


def get_error_logger() -> StructuredLogger:
    """Get the error logger."""
    return log_manager.get_error_logger()
