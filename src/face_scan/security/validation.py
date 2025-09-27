"""
Input validation and sanitization module.

Provides security validation for user inputs, file uploads, and data processing.
"""

import re
import os
import magic
from typing import List, Optional, Union, Any
from pathlib import Path
import bleach
from PIL import Image
import cv2
import numpy as np


class SecurityValidator:
    """Validates inputs for security threats and malicious content."""
    
    def __init__(self):
        self.allowed_file_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.max_file_size = 16 * 1024 * 1024  # 16MB
        self.max_image_dimensions = (4096, 4096)
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'<iframe.*?>',
            r'<object.*?>',
            r'<embed.*?>',
            r'<link.*?>',
            r'<meta.*?>',
        ]
    
    def validate_filename(self, filename: str) -> bool:
        """Validate filename for security issues."""
        if not filename or len(filename) > 255:
            return False
        
        # Check for path traversal attempts
        if '..' in filename or '/' in filename or '\\' in filename:
            return False
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in filename for char in dangerous_chars):
            return False
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_file_extensions:
            return False
        
        return True
    
    def validate_file_size(self, file_size: int) -> bool:
        """Validate file size."""
        return 0 < file_size <= self.max_file_size
    
    def validate_file_type(self, file_path: str) -> bool:
        """Validate file type using magic number detection."""
        try:
            # Use python-magic to detect actual file type
            file_type = magic.from_file(file_path, mime=True)
            
            allowed_mime_types = {
                'image/jpeg',
                'image/png', 
                'image/bmp',
                'image/tiff',
                'image/webp'
            }
            
            return file_type in allowed_mime_types
        except Exception:
            return False
    
    def validate_image_content(self, image_path: str) -> bool:
        """Validate image content for malicious data."""
        try:
            # Try to open with PIL
            with Image.open(image_path) as img:
                # Check dimensions
                if img.size[0] > self.max_image_dimensions[0] or img.size[1] > self.max_image_dimensions[1]:
                    return False
                
                # Verify image can be processed
                img.verify()
            
            # Try to open with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # Check if image has reasonable dimensions
            height, width = img.shape[:2]
            if height == 0 or width == 0:
                return False
            
            return True
        except Exception:
            return False
    
    def validate_upload(self, file_path: str, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Comprehensive file upload validation.
        Returns (is_valid, error_message)
        """
        # Validate filename
        if not self.validate_filename(filename):
            return False, "Invalid filename"
        
        # Validate file size
        if not self.validate_file_size(file_size):
            return False, f"File size exceeds limit ({self.max_file_size} bytes)"
        
        # Validate file type
        if not self.validate_file_type(file_path):
            return False, "Invalid file type"
        
        # Validate image content
        if not self.validate_image_content(file_path):
            return False, "Invalid or corrupted image"
        
        return True, "Valid"
    
    def sanitize_string(self, text: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent XSS and injection attacks."""
        if not text:
            return ""
        
        # Limit length
        text = text[:max_length]
        
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Use bleach for HTML sanitization
        text = bleach.clean(text, tags=[], strip=True)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text.strip()
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url))
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(pattern, ip))


class InputSanitizer:
    """Sanitizes various types of input data."""
    
    def __init__(self):
        self.validator = SecurityValidator()
    
    def sanitize_user_input(self, data: Any) -> Any:
        """Sanitize user input based on type."""
        if isinstance(data, str):
            return self.validator.sanitize_string(data)
        elif isinstance(data, dict):
            return {key: self.sanitize_user_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_user_input(item) for item in data]
        else:
            return data
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Remove multiple consecutive underscores
        filename = re.sub(r'_+', '_', filename)
        
        # Remove leading/trailing underscores and dots
        filename = filename.strip('_.')
        
        # Ensure filename is not empty
        if not filename:
            filename = 'unnamed_file'
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    def sanitize_path(self, path: str) -> str:
        """Sanitize file path to prevent directory traversal."""
        # Normalize path
        path = os.path.normpath(path)
        
        # Remove any remaining dangerous patterns
        path = re.sub(r'\.\./', '', path)
        path = re.sub(r'\.\.\\', '', path)
        
        # Remove leading slashes/backslashes
        path = path.lstrip('/\\')
        
        return path
    
    def sanitize_sql_input(self, query: str) -> str:
        """Basic SQL injection prevention (use parameterized queries instead)."""
        # Remove common SQL injection patterns
        dangerous_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
            r'(\b(OR|AND)\s+\'\s*=\s*\')',
            r'(\b(OR|AND)\s+"\s*=\s*")',
            r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
            r'(\b(OR|AND)\s+\w+\s*=\s*\w+)',
            r'(\b(OR|AND)\s+\'\s*=\s*\')',
            r'(\b(OR|AND)\s+"\s*=\s*")',
        ]
        
        for pattern in dangerous_patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        return query.strip()


class DataValidator:
    """Validates data structures and formats."""
    
    @staticmethod
    def validate_face_encoding(encoding: np.ndarray) -> bool:
        """Validate face encoding array."""
        if not isinstance(encoding, np.ndarray):
            return False
        
        if encoding.shape != (128,):  # Standard face_recognition encoding size
            return False
        
        if not np.isfinite(encoding).all():
            return False
        
        return True
    
    @staticmethod
    def validate_coordinates(coords: List[float], expected_length: int = 4) -> bool:
        """Validate coordinate list (e.g., bounding box)."""
        if not isinstance(coords, list):
            return False
        
        if len(coords) != expected_length:
            return False
        
        if not all(isinstance(coord, (int, float)) for coord in coords):
            return False
        
        if not all(np.isfinite(coord) for coord in coords):
            return False
        
        return True
    
    @staticmethod
    def validate_confidence_score(score: float) -> bool:
        """Validate confidence score."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0 and np.isfinite(score)
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> bool:
        """Validate image array."""
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if image.size == 0:
            return False
        
        if not np.isfinite(image).all():
            return False
        
        return True
