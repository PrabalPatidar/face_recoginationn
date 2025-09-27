"""
Authentication middleware for API endpoints.
"""

from functools import wraps
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('Authorization')
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'MISSING_API_KEY',
                    'message': 'API key is required'
                }
            }), 401
        
        # Remove 'Bearer ' prefix if present
        if api_key.startswith('Bearer '):
            api_key = api_key[7:]
        
        # Validate API key (simplified - in production, use proper validation)
        if not validate_api_key(api_key):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_API_KEY',
                    'message': 'Invalid API key'
                }
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def validate_api_key(api_key):
    """
    Validate API key.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Simplified validation - in production, implement proper key validation
    # This could check against a database, environment variables, etc.
    
    if not api_key:
        return False
    
    # For demo purposes, accept any non-empty key
    # In production, implement proper key validation logic
    return len(api_key) > 0


def get_current_user():
    """
    Get current user from API key.
    
    Returns:
        User information or None
    """
    api_key = request.headers.get('Authorization')
    
    if api_key and api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    if validate_api_key(api_key):
        # In production, return actual user information
        return {
            'id': 'user_123',
            'name': 'API User',
            'api_key': api_key
        }
    
    return None
