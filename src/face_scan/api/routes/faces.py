"""
Face management endpoints.
"""

import logging
from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from face_scan.core.face_recognizer import FaceRecognizer
from face_scan.core.face_encoder import FaceEncoder
from face_scan.core.face_detector import FaceDetector

faces_bp = Blueprint('faces', __name__)
logger = logging.getLogger(__name__)

# Initialize components
face_recognizer = FaceRecognizer()
face_encoder = FaceEncoder()
face_detector = FaceDetector()


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


@faces_bp.route('/', methods=['GET'])
def get_faces():
    """Get all known faces."""
    try:
        known_faces = face_recognizer.get_known_faces()
        
        faces = []
        for i, (name, encoding) in enumerate(known_faces.items()):
            faces.append({
                'id': i + 1,
                'name': name,
                'encoding_id': f"enc_{i+1}",
                'created_at': "2023-12-01T10:00:00Z",  # Placeholder
                'updated_at': "2023-12-01T10:00:00Z"  # Placeholder
            })
        
        return jsonify({
            'success': True,
            'faces': faces,
            'total_count': len(faces)
        })
        
    except Exception as e:
        logger.error(f"Error getting faces: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'DATABASE_ERROR',
                'message': 'Error retrieving faces',
                'details': str(e)
            }
        }), 500


@faces_bp.route('/', methods=['POST'])
def add_face():
    """Add a new face to the recognition database."""
    try:
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
        
        # Get name from form data
        name = request.form.get('name')
        if not name:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'MISSING_NAME',
                    'message': 'Name is required'
                }
            }), 400
        
        # Process image
        image = process_uploaded_image(file)
        
        # Detect faces
        face_locations = face_detector.detect_faces(image)
        
        if not face_locations:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NO_FACE_DETECTED',
                    'message': 'No face detected in the image'
                }
            }), 400
        
        # Get face encoding from first detected face
        face_encoding = face_encoder.encode_face(image, face_locations[0])
        
        if face_encoding is None:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'ENCODING_FAILED',
                    'message': 'Failed to encode face'
                }
            }), 500
        
        # Add to recognizer
        face_recognizer.add_known_face(face_encoding, name)
        
        return jsonify({
            'success': True,
            'face': {
                'id': face_recognizer.get_known_faces_count(),
                'name': name,
                'encoding_id': f"enc_{face_recognizer.get_known_faces_count()}",
                'created_at': "2023-12-01T10:00:00Z"  # Placeholder
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding face: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': 'Error processing face',
                'details': str(e)
            }
        }), 500


@faces_bp.route('/<int:face_id>', methods=['GET'])
def get_face(face_id):
    """Get a specific face by ID."""
    try:
        known_faces = face_recognizer.get_known_faces()
        face_names = list(known_faces.keys())
        
        if face_id < 1 or face_id > len(face_names):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': 'Face not found'
                }
            }), 404
        
        name = face_names[face_id - 1]
        
        return jsonify({
            'success': True,
            'face': {
                'id': face_id,
                'name': name,
                'encoding_id': f"enc_{face_id}",
                'created_at': "2023-12-01T10:00:00Z",  # Placeholder
                'updated_at': "2023-12-01T10:00:00Z"  # Placeholder
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting face: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'DATABASE_ERROR',
                'message': 'Error retrieving face',
                'details': str(e)
            }
        }), 500


@faces_bp.route('/<int:face_id>', methods=['PUT'])
def update_face(face_id):
    """Update a face's information."""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'INVALID_REQUEST',
                    'message': 'Name is required'
                }
            }), 400
        
        known_faces = face_recognizer.get_known_faces()
        face_names = list(known_faces.keys())
        
        if face_id < 1 or face_id > len(face_names):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': 'Face not found'
                }
            }), 404
        
        old_name = face_names[face_id - 1]
        new_name = data['name']
        
        # Remove old face and add with new name
        face_recognizer.remove_known_face(old_name)
        face_recognizer.add_known_face(known_faces[old_name], new_name)
        
        return jsonify({
            'success': True,
            'face': {
                'id': face_id,
                'name': new_name,
                'encoding_id': f"enc_{face_id}",
                'created_at': "2023-12-01T10:00:00Z",  # Placeholder
                'updated_at': "2023-12-01T10:00:00Z"  # Placeholder
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating face: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': 'Error updating face',
                'details': str(e)
            }
        }), 500


@faces_bp.route('/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete a face from the recognition database."""
    try:
        known_faces = face_recognizer.get_known_faces()
        face_names = list(known_faces.keys())
        
        if face_id < 1 or face_id > len(face_names):
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': 'Face not found'
                }
            }), 404
        
        name = face_names[face_id - 1]
        face_recognizer.remove_known_face(name)
        
        return jsonify({
            'success': True,
            'message': 'Face deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'PROCESSING_ERROR',
                'message': 'Error deleting face',
                'details': str(e)
            }
        }), 500
