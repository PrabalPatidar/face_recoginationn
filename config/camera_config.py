"""
Camera configuration settings.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Camera types
CAMERA_TYPES = {
    'webcam': 'webcam',
    'ip_camera': 'ip_camera',
    'usb_camera': 'usb_camera',
    'rtsp_camera': 'rtsp_camera'
}

# Default camera settings
DEFAULT_CAMERA_SETTINGS = {
    'index': int(os.getenv('DEFAULT_CAMERA_INDEX', '0')),
    'width': int(os.getenv('CAMERA_WIDTH', '640')),
    'height': int(os.getenv('CAMERA_HEIGHT', '480')),
    'fps': int(os.getenv('CAMERA_FPS', '30')),
    'format': 'MJPG',
    'buffer_size': 1,
    'auto_exposure': True,
    'auto_white_balance': True,
    'brightness': 50,
    'contrast': 50,
    'saturation': 50,
    'hue': 0
}

# Camera resolution presets
RESOLUTION_PRESETS = {
    'qvga': (320, 240),
    'vga': (640, 480),
    'svga': (800, 600),
    'xga': (1024, 768),
    'sxga': (1280, 1024),
    'uxga': (1600, 1200),
    'hd': (1280, 720),
    'fhd': (1920, 1080),
    '4k': (3840, 2160)
}

# Camera quality settings
QUALITY_SETTINGS = {
    'low': {
        'resolution': RESOLUTION_PRESETS['qvga'],
        'fps': 15,
        'compression': 80,
        'description': 'Low quality, fast processing'
    },
    'medium': {
        'resolution': RESOLUTION_PRESETS['vga'],
        'fps': 30,
        'compression': 90,
        'description': 'Balanced quality and performance'
    },
    'high': {
        'resolution': RESOLUTION_PRESETS['hd'],
        'fps': 30,
        'compression': 95,
        'description': 'High quality, slower processing'
    },
    'ultra': {
        'resolution': RESOLUTION_PRESETS['fhd'],
        'fps': 60,
        'compression': 98,
        'description': 'Ultra high quality, requires powerful hardware'
    }
}

# IP Camera settings
IP_CAMERA_SETTINGS = {
    'default_protocol': 'http',
    'default_port': 80,
    'timeout': 10,
    'retry_attempts': 3,
    'connection_pool_size': 5,
    'supported_protocols': ['http', 'https', 'rtsp', 'rtmp'],
    'authentication': {
        'enabled': False,
        'username': '',
        'password': '',
        'auth_type': 'basic'  # basic, digest, ntlm
    }
}

# RTSP Camera settings
RTSP_CAMERA_SETTINGS = {
    'default_port': 554,
    'timeout': 15,
    'buffer_size': 1024,
    'reconnect_interval': 5,
    'supported_codecs': ['h264', 'h265', 'mjpeg'],
    'stream_quality': 'medium'
}

# Camera detection settings
CAMERA_DETECTION = {
    'auto_detect': True,
    'scan_range': 10,  # Number of camera indices to scan
    'test_duration': 2,  # Seconds to test each camera
    'preferred_cameras': [0, 1, 2],  # Preferred camera indices
    'exclude_cameras': []  # Camera indices to exclude
}

# Camera calibration settings
CALIBRATION_SETTINGS = {
    'auto_calibrate': True,
    'calibration_interval': 3600,  # Seconds between calibrations
    'calibration_images': 20,  # Number of images for calibration
    'save_calibration': True,
    'calibration_file': BASE_DIR / 'data' / 'models' / 'camera_calibration.json'
}

# Camera streaming settings
STREAMING_SETTINGS = {
    'enabled': True,
    'port': 8080,
    'quality': 'medium',
    'max_clients': 10,
    'stream_format': 'mjpeg',
    'compression': 90,
    'frame_skip': 1,  # Process every nth frame
    'buffer_size': 5
}

# Camera recording settings
RECORDING_SETTINGS = {
    'enabled': False,
    'output_format': 'mp4',
    'codec': 'h264',
    'quality': 'medium',
    'max_duration': 3600,  # Maximum recording duration in seconds
    'output_directory': BASE_DIR / 'data' / 'raw' / 'videos',
    'auto_delete_old': True,
    'retention_days': 7
}

# Camera error handling
ERROR_HANDLING = {
    'max_errors': 5,
    'error_timeout': 30,  # Seconds to wait after error
    'auto_reconnect': True,
    'fallback_camera': 0,
    'error_logging': True,
    'error_notifications': False
}

# Camera performance monitoring
PERFORMANCE_MONITORING = {
    'enabled': True,
    'metrics_interval': 60,  # Seconds between metric collection
    'track_fps': True,
    'track_latency': True,
    'track_errors': True,
    'alert_thresholds': {
        'fps_drop': 0.5,  # Alert if FPS drops below 50% of target
        'latency_high': 100,  # Alert if latency exceeds 100ms
        'error_rate': 0.1  # Alert if error rate exceeds 10%
    }
}

# Camera security settings
SECURITY_SETTINGS = {
    'access_control': False,
    'allowed_ips': [],
    'blocked_ips': [],
    'rate_limiting': True,
    'max_requests_per_minute': 60,
    'ssl_enabled': False,
    'ssl_certificate': '',
    'ssl_key': ''
}

# Camera hardware settings
HARDWARE_SETTINGS = {
    'gpu_acceleration': False,
    'cuda_device': 0,
    'memory_limit': 1024,  # MB
    'cpu_cores': 0,  # 0 = use all available cores
    'threading': True,
    'async_processing': True
}

# Camera debugging settings
DEBUG_SETTINGS = {
    'enabled': False,
    'log_level': 'INFO',
    'save_frames': False,
    'frame_directory': BASE_DIR / 'logs' / 'debug_frames',
    'performance_profiling': False,
    'memory_profiling': False
}

# Camera configuration validation
CONFIG_VALIDATION = {
    'validate_on_startup': True,
    'strict_validation': False,
    'auto_fix_issues': True,
    'validation_timeout': 10
}

# Camera backup settings
BACKUP_SETTINGS = {
    'enabled': False,
    'backup_interval': 86400,  # 24 hours
    'backup_directory': BASE_DIR / 'data' / 'backup' / 'camera_config',
    'max_backups': 7,
    'compress_backups': True
}

# Camera update settings
UPDATE_SETTINGS = {
    'auto_update': False,
    'update_check_interval': 86400,  # 24 hours
    'update_channel': 'stable',  # stable, beta, alpha
    'backup_before_update': True
}

# Ensure camera directories exist
for directory in [
    RECORDING_SETTINGS['output_directory'],
    DEBUG_SETTINGS['frame_directory'],
    BACKUP_SETTINGS['backup_directory']
]:
    directory.mkdir(parents=True, exist_ok=True)
