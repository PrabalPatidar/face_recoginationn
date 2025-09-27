"""
Face Scan Project - A comprehensive face detection and recognition system.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "A comprehensive face detection and recognition system"

# Import main components
from .app import create_app
from .main import main

__all__ = ['create_app', 'main', '__version__']
