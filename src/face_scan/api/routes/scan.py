"""
Face scanning endpoints.
"""

import time
import logging
from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from face_scan.core.face_detector import FaceDetector
from face_scan.core.face_recognizer import FaceRecognizer
from face_scan.core.face_encoder import FaceEncoder
from face_scan.core.image_processor import ImageProcessor

scan_bp = Blueprint('scan', __name__)
logger = logging.getLogger(__name__)

# Initialize components
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
face_encoder = FaceEncoder()
image_processor = ImageProcessor()


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_uploaded_image(file):
    """Process uploaded image file."""
    try:
        # Read image from file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise


@scan_bp.route('/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image."""
    try:
        start_time = time.time()
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FILE',
                    'message': 'No image file provided'
                }
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FILE',
                    'message': 'No file selected'
                }
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_FILE',
                    'message': 'Invalid file type'
                }
            }), 400
        
        # Process image
        image = process_uploaded_image(file)
        
        # Detect faces
        face_locations = face_detector.detect_faces(image)
        
        # Format response
        faces = []
        for i, (x, y, w, h) in enumerate(face_locations):
            faces.append({
                'id': i + 1,
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'confidence': 1.0  # Placeholder - would need actual confidence from detector
            })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'faces': faces,
            'processing_time': round(processing_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': 'Error processing image',
                'details': str(e)
            }
        }), 500


@scan_bp.route('/recognize', methods=['POST'])
def recognize_faces():
    """Recognize faces in uploaded image."""
    try:
        start_time = time.time()
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FILE',
                    'message': 'No image file provided'
                }
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FILE',
                    'message': 'No file selected'
                }
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_FILE',
                    'message': 'Invalid file type'
                }
            }), 400
        
        # Process image
        image = process_uploaded_image(file)
        
        # Detect faces
        face_locations = face_detector.detect_faces(image)
        
        if not face_locations:
            return jsonify({
                'success': True,
                'faces_recognized': 0,
                'faces': [],
                'processing_time': round(time.time() - start_time, 3)
            })
        
        # Get face encodings
        face_encodings = face_encoder.encode_faces(image, face_locations)
        
        # Recognize faces
        recognition_results = face_recognizer.recognize_faces(face_encodings)
        
        # Format response
        faces = []
        for i, ((x, y, w, h), (name, confidence)) in enumerate(zip(face_locations, recognition_results)):
            faces.append({
                'id': i + 1,
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                },
                'name': name,
                'confidence': round(confidence, 3),
                'encoding_id': f"enc_{i+1}" if name != "Unknown" else None
            })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'faces_recognized': len(faces),
            'faces': faces,
            'processing_time': round(processing_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': 'Error processing image',
                'details': str(e)
            }
        }), 500
