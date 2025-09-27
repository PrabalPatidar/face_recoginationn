"""
Authentication and authorization module.

Provides JWT token management, user authentication, and role-based access control.
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import request, jsonify, current_app


class TokenManager:
    """Manages JWT token creation, validation, and refresh."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=7)
    
    def generate_access_token(self, user_id: str, roles: List[str] = None) -> str:
        """Generate a new access token."""
        payload = {
            'user_id': user_id,
            'roles': roles or [],
            'type': 'access',
            'exp': datetime.utcnow() + self.access_token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def generate_refresh_token(self, user_id: str) -> str:
        """Generate a new refresh token."""
        payload = {
            'user_id': user_id,
            'type': 'refresh',
            'exp': datetime.utcnow() + self.refresh_token_expiry,
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token."""
        payload = self.validate_token(refresh_token)
        if payload and payload.get('type') == 'refresh':
            return self.generate_access_token(payload['user_id'])
        return None


class AuthenticationManager:
    """Manages user authentication and session handling."""
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            salt, stored_hash = hashed_password.split(':')
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == stored_hash
        except ValueError:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials."""
        # This would typically query a database
        # For now, using a simple in-memory store
        users = {
            'admin': {
                'password_hash': self.hash_password('admin123'),
                'roles': ['admin', 'user'],
                'user_id': 'admin_001'
            },
            'user': {
                'password_hash': self.hash_password('user123'),
                'roles': ['user'],
                'user_id': 'user_001'
            }
        }
        
        if username in users:
            user_data = users[username]
            if self.verify_password(password, user_data['password_hash']):
                return {
                    'user_id': user_data['user_id'],
                    'username': username,
                    'roles': user_data['roles']
                }
        return None
    
    def login(self, username: str, password: str) -> Optional[Dict[str, str]]:
        """Perform user login and return tokens."""
        user = self.authenticate_user(username, password)
        if user:
            access_token = self.token_manager.generate_access_token(
                user['user_id'], user['roles']
            )
            refresh_token = self.token_manager.generate_refresh_token(user['user_id'])
            
            # Store session
            self.active_sessions[user['user_id']] = {
                'username': user['username'],
                'roles': user['roles'],
                'login_time': datetime.utcnow(),
                'last_activity': datetime.utcnow()
            }
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'user_id': user['user_id'],
                'roles': user['roles']
            }
        return None
    
    def logout(self, user_id: str) -> bool:
        """Logout user and invalidate session."""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            return True
        return False
    
    def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Get current user from token."""
        payload = self.token_manager.validate_token(token)
        if payload and payload.get('type') == 'access':
            user_id = payload.get('user_id')
            if user_id in self.active_sessions:
                return {
                    'user_id': user_id,
                    'roles': payload.get('roles', []),
                    'session': self.active_sessions[user_id]
                }
        return None


def require_auth(f):
    """Decorator to require authentication for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header'}), 401
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        auth_manager = current_app.config.get('auth_manager')
        if not auth_manager:
            return jsonify({'error': 'Authentication not configured'}), 500
        
        user = auth_manager.get_current_user(token)
        if not user:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(required_roles: List[str]):
    """Decorator to require specific roles for endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            user_roles = request.current_user.get('roles', [])
            if not any(role in user_roles for role in required_roles):
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_permission(permission: str):
    """Decorator to require specific permission for endpoints."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
            
            # This would typically check against a permission system
            # For now, using role-based permissions
            user_roles = request.current_user.get('roles', [])
            
            # Define permission mappings
            permission_roles = {
                'scan_faces': ['admin', 'user'],
                'manage_users': ['admin'],
                'view_analytics': ['admin'],
                'manage_models': ['admin']
            }
            
            allowed_roles = permission_roles.get(permission, [])
            if not any(role in user_roles for role in allowed_roles):
                return jsonify({'error': f'Permission {permission} required'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
