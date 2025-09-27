"""
Application settings and configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Application settings
APP_NAME = "Face Scan Project"
APP_VERSION = "1.0.0"
DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database settings
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///face_scan.db')
DATABASE_HOST = os.getenv('DATABASE_HOST', 'localhost')
DATABASE_PORT = int(os.getenv('DATABASE_PORT', '5432'))
DATABASE_NAME = os.getenv('DATABASE_NAME', 'face_scan_db')
DATABASE_USER = os.getenv('DATABASE_USER', 'face_scan_user')
DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'password')

# Face recognition settings
FACE_DETECTION_MODEL = os.getenv('FACE_DETECTION_MODEL', 'hog')
FACE_RECOGNITION_MODEL = os.getenv('FACE_RECOGNITION_MODEL', 'face_recognition')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
FACE_ENCODING_TOLERANCE = float(os.getenv('FACE_ENCODING_TOLERANCE', '0.4'))

# Camera settings
DEFAULT_CAMERA_INDEX = int(os.getenv('DEFAULT_CAMERA_INDEX', '0'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))

# Storage settings
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', str(BASE_DIR / 'data' / 'raw' / 'images'))
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', str(BASE_DIR / 'data' / 'processed' / 'faces'))
MODEL_FOLDER = os.getenv('MODEL_FOLDER', str(BASE_DIR / 'data' / 'models'))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'app.log'))
ERROR_LOG_FILE = os.getenv('ERROR_LOG_FILE', str(BASE_DIR / 'logs' / 'error.log'))

# Performance settings
MAX_FACES_PER_IMAGE = int(os.getenv('MAX_FACES_PER_IMAGE', '10'))
IMAGE_QUALITY = int(os.getenv('IMAGE_QUALITY', '95'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))

# Security settings
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'jpg,jpeg,png,bmp,tiff').split(','))
MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB

# API settings
API_PREFIX = '/api/v1'
RATE_LIMIT_PER_MINUTE = 100
RATE_LIMIT_PER_HOUR = 1000

# Model paths
HAAR_CASCADE_PATH = BASE_DIR / 'data' / 'models' / 'face_detection' / 'haarcascade_frontalface_default.xml'
DLIB_SHAPE_PREDICTOR_PATH = BASE_DIR / 'data' / 'models' / 'pretrained' / 'shape_predictor_68_face_landmarks.dat'
FACENET_MODEL_PATH = BASE_DIR / 'data' / 'models' / 'pretrained' / '20180402-114759'

# Ensure directories exist
for directory in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Flask configuration
class Config:
    """Base configuration class."""
    SECRET_KEY = SECRET_KEY
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH
    UPLOAD_FOLDER = UPLOAD_FOLDER


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///face_scan_dev.db'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = DATABASE_URL


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
