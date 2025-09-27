"""
CORS middleware for API endpoints.
"""

from flask_cors import CORS
from flask import Flask


def setup_cors(app: Flask):
    """
    Set up CORS for the Flask application.
    
    Args:
        app: Flask application instance
    """
    CORS(app, resources={
        r"/api/*": {
            "origins": ["*"],  # In production, specify allowed origins
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
