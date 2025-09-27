"""
Permission management module.

Provides role-based access control (RBAC) and permission management.
"""

from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass
from functools import wraps
from flask import request, jsonify, g


class Permission(Enum):
    """Available permissions in the system."""
    
    # Face scanning permissions
    SCAN_FACES = "scan_faces"
    VIEW_SCAN_RESULTS = "view_scan_results"
    EXPORT_SCAN_DATA = "export_scan_data"
    
    # User management permissions
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"
    CREATE_USERS = "create_users"
    DELETE_USERS = "delete_users"
    
    # Model management permissions
    MANAGE_MODELS = "manage_models"
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    DELETE_MODELS = "delete_models"
    
    # Analytics permissions
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_ANALYTICS = "export_analytics"
    MANAGE_DASHBOARDS = "manage_dashboards"
    
    # System permissions
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"
    MANAGE_CONFIG = "manage_config"
    BACKUP_DATA = "backup_data"
    
    # API permissions
    API_READ = "api_read"
    API_WRITE = "api_write"
    API_ADMIN = "api_admin"


class Role(Enum):
    """Available roles in the system."""
    
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    OPERATOR = "operator"
    USER = "user"
    GUEST = "guest"


@dataclass
class RolePermission:
    """Represents a role and its permissions."""
    role: Role
    permissions: Set[Permission]
    description: str = ""


class PermissionManager:
    """Manages roles, permissions, and access control."""
    
    def __init__(self):
        self.role_permissions: Dict[Role, Set[Permission]] = {}
        self.user_roles: Dict[str, Set[Role]] = {}
        self._initialize_default_permissions()
    
    def _initialize_default_permissions(self):
        """Initialize default role-permission mappings."""
        
        # Admin role - full access
        self.role_permissions[Role.ADMIN] = {
            Permission.SCAN_FACES,
            Permission.VIEW_SCAN_RESULTS,
            Permission.EXPORT_SCAN_DATA,
            Permission.MANAGE_USERS,
            Permission.VIEW_USERS,
            Permission.CREATE_USERS,
            Permission.DELETE_USERS,
            Permission.MANAGE_MODELS,
            Permission.TRAIN_MODELS,
            Permission.DEPLOY_MODELS,
            Permission.DELETE_MODELS,
            Permission.VIEW_ANALYTICS,
            Permission.EXPORT_ANALYTICS,
            Permission.MANAGE_DASHBOARDS,
            Permission.MANAGE_SYSTEM,
            Permission.VIEW_LOGS,
            Permission.MANAGE_CONFIG,
            Permission.BACKUP_DATA,
            Permission.API_READ,
            Permission.API_WRITE,
            Permission.API_ADMIN,
        }
        
        # Manager role - management access
        self.role_permissions[Role.MANAGER] = {
            Permission.SCAN_FACES,
            Permission.VIEW_SCAN_RESULTS,
            Permission.EXPORT_SCAN_DATA,
            Permission.VIEW_USERS,
            Permission.CREATE_USERS,
            Permission.VIEW_ANALYTICS,
            Permission.EXPORT_ANALYTICS,
            Permission.MANAGE_DASHBOARDS,
            Permission.VIEW_LOGS,
            Permission.API_READ,
            Permission.API_WRITE,
        }
        
        # Analyst role - analytics and reporting
        self.role_permissions[Role.ANALYST] = {
            Permission.VIEW_SCAN_RESULTS,
            Permission.EXPORT_SCAN_DATA,
            Permission.VIEW_ANALYTICS,
            Permission.EXPORT_ANALYTICS,
            Permission.MANAGE_DASHBOARDS,
            Permission.API_READ,
        }
        
        # Operator role - operational tasks
        self.role_permissions[Role.OPERATOR] = {
            Permission.SCAN_FACES,
            Permission.VIEW_SCAN_RESULTS,
            Permission.VIEW_ANALYTICS,
            Permission.API_READ,
        }
        
        # User role - basic access
        self.role_permissions[Role.USER] = {
            Permission.SCAN_FACES,
            Permission.VIEW_SCAN_RESULTS,
            Permission.API_READ,
        }
        
        # Guest role - limited access
        self.role_permissions[Role.GUEST] = {
            Permission.VIEW_SCAN_RESULTS,
        }
    
    def assign_role_to_user(self, user_id: str, role: Role):
        """Assign a role to a user."""
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        self.user_roles[user_id].add(role)
    
    def remove_role_from_user(self, user_id: str, role: Role):
        """Remove a role from a user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]
    
    def get_user_roles(self, user_id: str) -> Set[Role]:
        """Get all roles for a user."""
        return self.user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user based on their roles."""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role in user_roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        return permissions
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user_permissions = self.get_user_permissions(user_id)
        return permission in user_permissions
    
    def has_any_permission(self, user_id: str, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        user_permissions = self.get_user_permissions(user_id)
        return any(permission in user_permissions for permission in permissions)
    
    def has_all_permissions(self, user_id: str, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        user_permissions = self.get_user_permissions(user_id)
        return all(permission in user_permissions for permission in permissions)
    
    def add_permission_to_role(self, role: Role, permission: Permission):
        """Add a permission to a role."""
        if role not in self.role_permissions:
            self.role_permissions[role] = set()
        self.role_permissions[role].add(permission)
    
    def remove_permission_from_role(self, role: Role, permission: Permission):
        """Remove a permission from a role."""
        if role in self.role_permissions:
            self.role_permissions[role].discard(permission)
    
    def create_custom_role(self, role_name: str, permissions: Set[Permission]) -> Role:
        """Create a custom role with specified permissions."""
        # Create a new role enum value dynamically
        custom_role = Role(role_name)
        self.role_permissions[custom_role] = permissions
        return custom_role
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self.role_permissions.get(role, set())
    
    def list_all_roles(self) -> List[Role]:
        """List all available roles."""
        return list(self.role_permissions.keys())
    
    def list_all_permissions(self) -> List[Permission]:
        """List all available permissions."""
        return list(Permission)


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for an endpoint.
    
    Args:
        permission: Required permission
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get current user from request context
            current_user = getattr(g, 'current_user', None)
            if not current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            user_id = current_user.get('user_id')
            if not user_id:
                return jsonify({'error': 'User ID not found'}), 401
            
            # Get permission manager from app context
            permission_manager = getattr(g, 'permission_manager', None)
            if not permission_manager:
                return jsonify({'error': 'Permission system not configured'}), 500
            
            # Check permission
            if not permission_manager.has_permission(user_id, permission):
                return jsonify({
                    'error': f'Permission {permission.value} required',
                    'required_permission': permission.value
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_any_permission(permissions: List[Permission]):
    """
    Decorator to require any of the specified permissions.
    
    Args:
        permissions: List of permissions (user needs at least one)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            current_user = getattr(g, 'current_user', None)
            if not current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            user_id = current_user.get('user_id')
            if not user_id:
                return jsonify({'error': 'User ID not found'}), 401
            
            permission_manager = getattr(g, 'permission_manager', None)
            if not permission_manager:
                return jsonify({'error': 'Permission system not configured'}), 500
            
            if not permission_manager.has_any_permission(user_id, permissions):
                return jsonify({
                    'error': f'One of the following permissions required: {[p.value for p in permissions]}',
                    'required_permissions': [p.value for p in permissions]
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_all_permissions(permissions: List[Permission]):
    """
    Decorator to require all of the specified permissions.
    
    Args:
        permissions: List of permissions (user needs all)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            current_user = getattr(g, 'current_user', None)
            if not current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            user_id = current_user.get('user_id')
            if not user_id:
                return jsonify({'error': 'User ID not found'}), 401
            
            permission_manager = getattr(g, 'permission_manager', None)
            if not permission_manager:
                return jsonify({'error': 'Permission system not configured'}), 500
            
            if not permission_manager.has_all_permissions(user_id, permissions):
                return jsonify({
                    'error': f'All of the following permissions required: {[p.value for p in permissions]}',
                    'required_permissions': [p.value for p in permissions]
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def require_role(role: Role):
    """
    Decorator to require a specific role.
    
    Args:
        role: Required role
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            current_user = getattr(g, 'current_user', None)
            if not current_user:
                return jsonify({'error': 'Authentication required'}), 401
            
            user_id = current_user.get('user_id')
            if not user_id:
                return jsonify({'error': 'User ID not found'}), 401
            
            permission_manager = getattr(g, 'permission_manager', None)
            if not permission_manager:
                return jsonify({'error': 'Permission system not configured'}), 500
            
            user_roles = permission_manager.get_user_roles(user_id)
            if role not in user_roles:
                return jsonify({
                    'error': f'Role {role.value} required',
                    'required_role': role.value
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


class ResourcePermission:
    """Manages permissions for specific resources."""
    
    def __init__(self, resource_type: str, resource_id: str, owner_id: str):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.owner_id = owner_id
        self.permissions: Dict[str, Set[Permission]] = {}
    
    def grant_permission(self, user_id: str, permission: Permission):
        """Grant permission to a user for this resource."""
        if user_id not in self.permissions:
            self.permissions[user_id] = set()
        self.permissions[user_id].add(permission)
    
    def revoke_permission(self, user_id: str, permission: Permission):
        """Revoke permission from a user for this resource."""
        if user_id in self.permissions:
            self.permissions[user_id].discard(permission)
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission for this resource."""
        # Owner always has full access
        if user_id == self.owner_id:
            return True
        
        # Check specific permissions
        return permission in self.permissions.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user on this resource."""
        if user_id == self.owner_id:
            return set(Permission)  # Owner has all permissions
        
        return self.permissions.get(user_id, set())


class PermissionCache:
    """Caches permission checks for performance."""
    
    def __init__(self, cache_ttl: int = 300):  # 5 minutes
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamps: Dict[str, float] = {}
    
    def _get_cache_key(self, user_id: str, permission: Permission) -> str:
        """Generate cache key for permission check."""
        return f"{user_id}:{permission.value}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache_timestamps:
            return False
        
        import time
        return time.time() - self.cache_timestamps[cache_key] < self.cache_ttl
    
    def get_cached_permission(self, user_id: str, permission: Permission) -> Optional[bool]:
        """Get cached permission result."""
        cache_key = self._get_cache_key(user_id, permission)
        
        if self._is_cache_valid(cache_key):
            return self.cache.get(cache_key, {}).get('result')
        
        return None
    
    def cache_permission(self, user_id: str, permission: Permission, result: bool):
        """Cache permission result."""
        cache_key = self._get_cache_key(user_id, permission)
        
        self.cache[cache_key] = {'result': result}
        import time
        self.cache_timestamps[cache_key] = time.time()
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user."""
        keys_to_remove = [key for key in self.cache.keys() if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def clear_cache(self):
        """Clear all cached permissions."""
        self.cache.clear()
        self.cache_timestamps.clear()
