"""
Validation utility functions.
"""

import re
import os
from typing import Any, List, Optional, Dict
from pathlib import Path
import logging


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if email is valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        True if phone number is valid, False otherwise
    """
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check if it's a valid length (7-15 digits)
    return 7 <= len(digits_only) <= 15


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None


def validate_file_path(file_path: str) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: File path to validate
        
    Returns:
        True if file path is valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.is_absolute() or not path.is_absolute()
    except Exception:
        return False


def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        True if file is a valid image, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    extension = Path(file_path).suffix.lower()
    
    if extension not in valid_extensions:
        return False
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB
        return False
    
    return True


def validate_video_file(file_path: str) -> bool:
    """
    Validate if file is a valid video.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        True if file is a valid video, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    extension = Path(file_path).suffix.lower()
    
    if extension not in valid_extensions:
        return False
    
    # Check file size (max 500MB)
    file_size = os.path.getsize(file_path)
    if file_size > 500 * 1024 * 1024:  # 500MB
        return False
    
    return True


def validate_face_detection_params(params: Dict[str, Any]) -> bool:
    """
    Validate face detection parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    required_params = ['method']
    valid_methods = ['haar_cascade', 'hog', 'cnn', 'mtcnn']
    
    # Check required parameters
    for param in required_params:
        if param not in params:
            return False
    
    # Validate method
    if params['method'] not in valid_methods:
        return False
    
    # Validate optional parameters
    if 'confidence_threshold' in params:
        threshold = params['confidence_threshold']
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            return False
    
    if 'min_face_size' in params:
        size = params['min_face_size']
        if not isinstance(size, int) or size <= 0:
            return False
    
    return True


def validate_face_recognition_params(params: Dict[str, Any]) -> bool:
    """
    Validate face recognition parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    # Validate tolerance
    if 'tolerance' in params:
        tolerance = params['tolerance']
        if not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
            return False
    
    # Validate confidence threshold
    if 'confidence_threshold' in params:
        threshold = params['confidence_threshold']
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            return False
    
    return True


def validate_camera_params(params: Dict[str, Any]) -> bool:
    """
    Validate camera parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        True if parameters are valid, False otherwise
    """
    # Validate camera index
    if 'camera_index' in params:
        index = params['camera_index']
        if not isinstance(index, int) or index < 0:
            return False
    
    # Validate resolution
    if 'width' in params:
        width = params['width']
        if not isinstance(width, int) or width <= 0:
            return False
    
    if 'height' in params:
        height = params['height']
        if not isinstance(height, int) or height <= 0:
            return False
    
    # Validate FPS
    if 'fps' in params:
        fps = params['fps']
        if not isinstance(fps, (int, float)) or fps <= 0:
            return False
    
    return True


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if API key is valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Check length (minimum 10 characters)
    if len(api_key) < 10:
        return False
    
    # Check if it contains only valid characters
    pattern = r'^[a-zA-Z0-9_-]+$'
    return re.match(pattern, api_key) is not None


def validate_username(username: str) -> bool:
    """
    Validate username format.
    
    Args:
        username: Username to validate
        
    Returns:
        True if username is valid, False otherwise
    """
    if not username or not isinstance(username, str):
        return False
    
    # Check length (3-20 characters)
    if not 3 <= len(username) <= 20:
        return False
    
    # Check if it contains only valid characters
    pattern = r'^[a-zA-Z0-9_-]+$'
    return re.match(pattern, username) is not None


def validate_password(password: str) -> bool:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        True if password is valid, False otherwise
    """
    if not password or not isinstance(password, str):
        return False
    
    # Check length (minimum 8 characters)
    if len(password) < 8:
        return False
    
    # Check if it contains at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False
    
    # Check if it contains at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False
    
    # Check if it contains at least one digit
    if not re.search(r'\d', password):
        return False
    
    # Check if it contains at least one special character
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    
    return True


def validate_json_data(data: Any) -> bool:
    """
    Validate JSON data structure.
    
    Args:
        data: Data to validate
        
    Returns:
        True if data is valid JSON, False otherwise
    """
    try:
        import json
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


def validate_config_file(config_path: str) -> bool:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if configuration file is valid, False otherwise
    """
    if not os.path.exists(config_path):
        return False
    
    try:
        # Check if it's a valid Python file
        if config_path.endswith('.py'):
            with open(config_path, 'r') as f:
                compile(f.read(), config_path, 'exec')
            return True
        
        # Check if it's a valid JSON file
        elif config_path.endswith('.json'):
            import json
            with open(config_path, 'r') as f:
                json.load(f)
            return True
        
        # Check if it's a valid YAML file
        elif config_path.endswith(('.yml', '.yaml')):
            import yaml
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error validating config file: {e}")
        return False


def validate_database_url(url: str) -> bool:
    """
    Validate database URL format.
    
    Args:
        url: Database URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check if it starts with a valid database protocol
    valid_protocols = ['sqlite://', 'postgresql://', 'mysql://', 'oracle://']
    return any(url.startswith(protocol) for protocol in valid_protocols)


def validate_port(port: int) -> bool:
    """
    Validate port number.
    
    Args:
        port: Port number to validate
        
    Returns:
        True if port is valid, False otherwise
    """
    return isinstance(port, int) and 1 <= port <= 65535


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address to validate
        
    Returns:
        True if IP address is valid, False otherwise
    """
    pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return re.match(pattern, ip) is not None


def validate_face_encoding(encoding: Any) -> bool:
    """
    Validate face encoding format.
    
    Args:
        encoding: Face encoding to validate
        
    Returns:
        True if encoding is valid, False otherwise
    """
    try:
        import numpy as np
        
        if not isinstance(encoding, np.ndarray):
            return False
        
        # Check if it's a 1D array
        if encoding.ndim != 1:
            return False
        
        # Check if it has the expected length (128 for face_recognition)
        if len(encoding) != 128:
            return False
        
        # Check if it contains valid float values
        if not np.isfinite(encoding).all():
            return False
        
        return True
        
    except Exception:
        return False
