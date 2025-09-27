#!/usr/bin/env python3
"""
Environment setup script for Face Scan Project.
This script sets up the initial environment, creates necessary directories,
and initializes the database.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_scan.database.connection import get_db_connection
from face_scan.database.models import create_tables


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/setup.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_directories():
    """Create necessary directories for the project."""
    logger = logging.getLogger(__name__)
    
    directories = [
        'data/raw/images',
        'data/raw/videos',
        'data/raw/datasets',
        'data/processed/faces',
        'data/processed/embeddings',
        'data/processed/annotations',
        'data/models/face_detection',
        'data/models/face_recognition',
        'data/models/pretrained',
        'data/samples/test_images',
        'data/samples/reference_faces',
        'logs',
        'static/css',
        'static/js',
        'static/images',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def setup_environment_file():
    """Set up environment file from template."""
    logger = logging.getLogger(__name__)
    
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        logger.info("Created .env file from template")
    elif env_file.exists():
        logger.info(".env file already exists")
    else:
        logger.warning("No env.example file found")


def setup_database():
    """Set up database tables and initial data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Create database connection
        conn = get_db_connection()
        
        # Create tables
        create_tables(conn)
        logger.info("Database tables created successfully")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


def download_sample_data():
    """Download sample data for testing."""
    logger = logging.getLogger(__name__)
    
    try:
        import requests
        from PIL import Image
        import numpy as np
        
        # Create sample test images
        sample_dir = Path('data/samples/test_images')
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample images
        for i in range(5):
            # Create a random image
            img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(sample_dir / f'sample_{i+1}.jpg')
        
        logger.info("Sample test images created")
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")


def verify_installation():
    """Verify that all required packages are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'opencv-python',
        'face-recognition',
        'flask',
        'numpy',
        'pillow',
        'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Set up Face Scan Project environment')
    parser.add_argument('--skip-db', action='store_true', help='Skip database setup')
    parser.add_argument('--skip-samples', action='store_true', help='Skip sample data creation')
    parser.add_argument('--verify-only', action='store_true', help='Only verify installation')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Face Scan Project environment setup")
    
    try:
        # Verify installation
        if not verify_installation():
            logger.error("Installation verification failed")
            sys.exit(1)
        
        if args.verify_only:
            logger.info("Installation verification completed successfully")
            return
        
        # Create directories
        logger.info("Creating directories...")
        create_directories()
        
        # Set up environment file
        logger.info("Setting up environment file...")
        setup_environment_file()
        
        # Set up database
        if not args.skip_db:
            logger.info("Setting up database...")
            setup_database()
        
        # Download sample data
        if not args.skip_samples:
            logger.info("Creating sample data...")
            download_sample_data()
        
        logger.info("Environment setup completed successfully!")
        logger.info("You can now run the application with: python src/face_scan/app.py")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
