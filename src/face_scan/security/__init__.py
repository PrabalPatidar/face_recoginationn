"""
Security module for Face Scan Project.

This module provides security-related functionality including:
- Authentication and authorization
- Data encryption and decryption
- Input validation and sanitization
- Security headers and CORS
- Rate limiting and throttling
"""

from .auth import AuthenticationManager, TokenManager
from .encryption import DataEncryption, FileEncryption
from .validation import SecurityValidator, InputSanitizer
from .rate_limiter import RateLimiter
from .permissions import PermissionManager

__all__ = [
    'AuthenticationManager',
    'TokenManager', 
    'DataEncryption',
    'FileEncryption',
    'SecurityValidator',
    'InputSanitizer',
    'RateLimiter',
    'PermissionManager'
]

__version__ = '1.0.0'
