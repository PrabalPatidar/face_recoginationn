#!/usr/bin/env python3
"""
Model download script for Face Scan Project.
This script downloads pre-trained models for face detection and recognition.
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from urllib.parse import urlparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/download_models.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def download_file(url, destination, chunk_size=8192):
    """Download a file from URL to destination."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Downloading {url} to {destination}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Downloaded {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract archive file."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz'] or '.tar.gz' in archive_path.name:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"Extracted to {extract_to}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def download_face_detection_models():
    """Download face detection models."""
    logger = logging.getLogger(__name__)
    
    models_dir = Path('data/models/face_detection')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenCV Haar Cascade models
    haar_models = {
        'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt2.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml',
        'haarcascade_profileface.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml'
    }
    
    for model_name, url in haar_models.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            if download_file(url, model_path):
                logger.info(f"Downloaded {model_name}")
            else:
                logger.error(f"Failed to download {model_name}")
        else:
            logger.info(f"{model_name} already exists")


def download_face_recognition_models():
    """Download face recognition models."""
    logger = logging.getLogger(__name__)
    
    models_dir = Path('data/models/face_recognition')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # MTCNN model
    mtcnn_url = "https://github.com/ipazc/mtcnn/releases/download/v1.0.0/mtcnn-1.0.0.tar.gz"
    mtcnn_archive = models_dir / "mtcnn-1.0.0.tar.gz"
    
    if not (models_dir / "mtcnn").exists():
        if download_file(mtcnn_url, mtcnn_archive):
            if extract_archive(mtcnn_archive, models_dir):
                mtcnn_archive.unlink()  # Remove archive after extraction
                logger.info("Downloaded MTCNN model")
            else:
                logger.error("Failed to extract MTCNN model")
        else:
            logger.error("Failed to download MTCNN model")
    else:
        logger.info("MTCNN model already exists")


def download_pretrained_models():
    """Download pre-trained models."""
    logger = logging.getLogger(__name__)
    
    models_dir = Path('data/models/pretrained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # FaceNet model
    facenet_url = "https://github.com/davidsandberg/facenet/releases/download/v1.0.0/20180402-114759.zip"
    facenet_archive = models_dir / "facenet-20180402-114759.zip"
    
    if not (models_dir / "20180402-114759").exists():
        if download_file(facenet_url, facenet_archive):
            if extract_archive(facenet_archive, models_dir):
                facenet_archive.unlink()  # Remove archive after extraction
                logger.info("Downloaded FaceNet model")
            else:
                logger.error("Failed to extract FaceNet model")
        else:
            logger.error("Failed to download FaceNet model")
    else:
        logger.info("FaceNet model already exists")


def download_dlib_models():
    """Download dlib models."""
    logger = logging.getLogger(__name__)
    
    models_dir = Path('data/models/pretrained')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # dlib shape predictor
    shape_predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    shape_predictor_archive = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    shape_predictor_file = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    if not shape_predictor_file.exists():
        if download_file(shape_predictor_url, shape_predictor_archive):
            # Extract bz2 file
            import bz2
            with bz2.BZ2File(shape_predictor_archive, 'rb') as source:
                with open(shape_predictor_file, 'wb') as target:
                    target.write(source.read())
            
            shape_predictor_archive.unlink()  # Remove archive
            logger.info("Downloaded dlib shape predictor")
        else:
            logger.error("Failed to download dlib shape predictor")
    else:
        logger.info("dlib shape predictor already exists")


def verify_models():
    """Verify that downloaded models are valid."""
    logger = logging.getLogger(__name__)
    
    models_to_check = [
        'data/models/face_detection/haarcascade_frontalface_default.xml',
        'data/models/face_recognition/mtcnn',
        'data/models/pretrained/20180402-114759',
        'data/models/pretrained/shape_predictor_68_face_landmarks.dat'
    ]
    
    all_valid = True
    
    for model_path in models_to_check:
        path = Path(model_path)
        if path.exists():
            logger.info(f"✓ {model_path} exists")
        else:
            logger.error(f"✗ {model_path} missing")
            all_valid = False
    
    return all_valid


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description='Download pre-trained models for Face Scan Project')
    parser.add_argument('--face-detection', action='store_true', help='Download face detection models')
    parser.add_argument('--face-recognition', action='store_true', help='Download face recognition models')
    parser.add_argument('--pretrained', action='store_true', help='Download pre-trained models')
    parser.add_argument('--dlib', action='store_true', help='Download dlib models')
    parser.add_argument('--all', action='store_true', help='Download all models')
    parser.add_argument('--verify', action='store_true', help='Verify downloaded models')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting model download process")
    
    try:
        # Create models directory
        Path('data/models').mkdir(parents=True, exist_ok=True)
        
        if args.verify:
            if verify_models():
                logger.info("All models verified successfully")
            else:
                logger.error("Some models are missing or invalid")
                sys.exit(1)
            return
        
        if args.all or not any([args.face_detection, args.face_recognition, args.pretrained, args.dlib]):
            # Download all models by default
            logger.info("Downloading all models...")
            download_face_detection_models()
            download_face_recognition_models()
            download_pretrained_models()
            download_dlib_models()
        else:
            if args.face_detection:
                download_face_detection_models()
            if args.face_recognition:
                download_face_recognition_models()
            if args.pretrained:
                download_pretrained_models()
            if args.dlib:
                download_dlib_models()
        
        # Verify downloaded models
        if verify_models():
            logger.info("All models downloaded and verified successfully!")
        else:
            logger.error("Some models failed to download or verify")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
