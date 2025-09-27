"""
Rate limiting and throttling module.

Provides rate limiting functionality to prevent abuse and ensure fair usage.
"""

import time
import threading
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from functools import wraps
from flask import request, jsonify, g
import redis
from datetime import datetime, timedelta


class RateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
    
    def _get_client_id(self) -> str:
        """Get client identifier (IP address or user ID)."""
        # Try to get user ID from request context
        if hasattr(g, 'user_id') and g.user_id:
            return f"user:{g.user_id}"
        
        # Fall back to IP address
        return f"ip:{request.remote_addr}"
    
    def _clean_old_requests(self, client_id: str):
        """Remove requests older than the window."""
        current_time = time.time()
        client_requests = self.requests[client_id]
        
        # Remove requests outside the window
        while client_requests and client_requests[0] <= current_time - self.window_seconds:
            client_requests.popleft()
    
    def is_allowed(self, client_id: Optional[str] = None) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed for client.
        
        Returns:
            (is_allowed, rate_limit_info)
        """
        if client_id is None:
            client_id = self._get_client_id()
        
        with self.lock:
            self._clean_old_requests(client_id)
            
            current_time = time.time()
            client_requests = self.requests[client_id]
            
            # Check if under limit
            if len(client_requests) < self.max_requests:
                client_requests.append(current_time)
                return True, {
                    'limit': self.max_requests,
                    'remaining': self.max_requests - len(client_requests),
                    'reset_time': int(current_time + self.window_seconds)
                }
            else:
                return False, {
                    'limit': self.max_requests,
                    'remaining': 0,
                    'reset_time': int(client_requests[0] + self.window_seconds)
                }
    
    def get_remaining_requests(self, client_id: Optional[str] = None) -> int:
        """Get number of remaining requests for client."""
        if client_id is None:
            client_id = self._get_client_id()
        
        with self.lock:
            self._clean_old_requests(client_id)
            return max(0, self.max_requests - len(self.requests[client_id]))


class RedisRateLimiter:
    """Redis-based rate limiter for distributed applications."""
    
    def __init__(self, redis_client: redis.Redis, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_client: Redis client instance
            max_requests: Maximum number of requests allowed in window
            window_seconds: Time window in seconds
        """
        self.redis = redis_client
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def _get_key(self, client_id: str) -> str:
        """Get Redis key for client."""
        return f"rate_limit:{client_id}"
    
    def is_allowed(self, client_id: str) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed using Redis.
        
        Returns:
            (is_allowed, rate_limit_info)
        """
        key = self._get_key(client_id)
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, self.window_seconds)
        
        results = pipe.execute()
        current_requests = results[1]
        
        if current_requests < self.max_requests:
            return True, {
                'limit': self.max_requests,
                'remaining': self.max_requests - current_requests - 1,
                'reset_time': current_time + self.window_seconds
            }
        else:
            # Remove the request we just added since it's not allowed
            self.redis.zrem(key, str(current_time))
            return False, {
                'limit': self.max_requests,
                'remaining': 0,
                'reset_time': current_time + self.window_seconds
            }


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on system load."""
    
    def __init__(self, base_max_requests: int = 100, window_seconds: int = 60):
        self.base_max_requests = base_max_requests
        self.window_seconds = window_seconds
        self.current_max_requests = base_max_requests
        self.system_load_history = deque(maxlen=10)
        self.rate_limiter = RateLimiter(base_max_requests, window_seconds)
    
    def update_system_load(self, load_percentage: float):
        """Update system load and adjust rate limits accordingly."""
        self.system_load_history.append(load_percentage)
        
        # Calculate average load
        avg_load = sum(self.system_load_history) / len(self.system_load_history)
        
        # Adjust rate limit based on load
        if avg_load > 80:
            # High load - reduce rate limit
            self.current_max_requests = max(10, int(self.base_max_requests * 0.5))
        elif avg_load > 60:
            # Medium load - slightly reduce rate limit
            self.current_max_requests = max(20, int(self.base_max_requests * 0.8))
        else:
            # Low load - use base rate limit
            self.current_max_requests = self.base_max_requests
        
        # Update the underlying rate limiter
        self.rate_limiter.max_requests = self.current_max_requests
    
    def is_allowed(self, client_id: Optional[str] = None) -> tuple[bool, Dict[str, int]]:
        """Check if request is allowed with adaptive limits."""
        return self.rate_limiter.is_allowed(client_id)


class TieredRateLimiter:
    """Rate limiter with different tiers for different user types."""
    
    def __init__(self):
        self.limiters = {
            'free': RateLimiter(max_requests=10, window_seconds=60),
            'premium': RateLimiter(max_requests=100, window_seconds=60),
            'enterprise': RateLimiter(max_requests=1000, window_seconds=60),
            'admin': RateLimiter(max_requests=10000, window_seconds=60)
        }
    
    def get_user_tier(self, user_id: Optional[str] = None) -> str:
        """Determine user tier based on user ID or request context."""
        if hasattr(g, 'user_roles') and g.user_roles:
            if 'admin' in g.user_roles:
                return 'admin'
            elif 'premium' in g.user_roles:
                return 'premium'
            elif 'enterprise' in g.user_roles:
                return 'enterprise'
        
        return 'free'
    
    def is_allowed(self, client_id: Optional[str] = None) -> tuple[bool, Dict[str, int]]:
        """Check if request is allowed based on user tier."""
        tier = self.get_user_tier()
        limiter = self.limiters[tier]
        return limiter.is_allowed(client_id)


def rate_limit(max_requests: int = 100, window_seconds: int = 60, 
               per: str = 'ip', message: str = 'Rate limit exceeded'):
    """
    Decorator for rate limiting Flask endpoints.
    
    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        per: Rate limit per 'ip' or 'user'
        message: Error message when limit exceeded
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get rate limiter from app context
            rate_limiter = getattr(g, 'rate_limiter', None)
            if not rate_limiter:
                # Create default rate limiter
                rate_limiter = RateLimiter(max_requests, window_seconds)
            
            # Determine client ID
            if per == 'user' and hasattr(g, 'user_id') and g.user_id:
                client_id = f"user:{g.user_id}"
            else:
                client_id = f"ip:{request.remote_addr}"
            
            # Check rate limit
            allowed, info = rate_limiter.is_allowed(client_id)
            
            if not allowed:
                response = jsonify({
                    'error': message,
                    'rate_limit': info
                })
                response.status_code = 429
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset_time'])
                return response
            
            # Add rate limit info to response headers
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(info['limit'])
                response.headers['X-RateLimit-Remaining'] = str(info['remaining'])
                response.headers['X-RateLimit-Reset'] = str(info['reset_time'])
            
            return response
        
        return decorated_function
    return decorator


def api_rate_limit(tier: str = 'free'):
    """
    Decorator for API rate limiting based on user tier.
    
    Args:
        tier: User tier ('free', 'premium', 'enterprise', 'admin')
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            tiered_limiter = getattr(g, 'tiered_limiter', None)
            if not tiered_limiter:
                tiered_limiter = TieredRateLimiter()
            
            allowed, info = tiered_limiter.is_allowed()
            
            if not allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded for your tier',
                    'tier': tier,
                    'rate_limit': info
                })
                response.status_code = 429
                return response
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


class RateLimitManager:
    """Manages multiple rate limiters for different endpoints."""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_limiter(self, name: str, max_requests: int, window_seconds: int):
        """Add a new rate limiter."""
        self.limiters[name] = RateLimiter(max_requests, window_seconds)
    
    def get_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        return self.limiters.get(name)
    
    def check_limit(self, name: str, client_id: Optional[str] = None) -> tuple[bool, Dict[str, int]]:
        """Check rate limit for a specific limiter."""
        limiter = self.get_limiter(name)
        if not limiter:
            return True, {'limit': 0, 'remaining': 0, 'reset_time': 0}
        
        return limiter.is_allowed(client_id)
