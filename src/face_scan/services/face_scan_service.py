"""
Face scanning service for processing images.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from face_scan.core.face_detector import FaceDetector
from face_scan.core.face_recognizer import FaceRecognizer
from face_scan.core.face_encoder import FaceEncoder
from face_scan.core.image_processor import ImageProcessor


class FaceScanService:
    """Service for face scanning operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.face_encoder = FaceEncoder()
        self.image_processor = ImageProcessor()
    
    def detect_faces(self, image: np.ndarray, method: str = 'haar_cascade') -> Dict[str, Any]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image
            method: Detection method to use
            
        Returns:
            Dictionary containing detection results
        """
        try:
            start_time = time.time()
            
            # Initialize detector with specified method
            detector = FaceDetector(method)
            
            # Detect faces
            face_locations = detector.detect_faces(image)
            
            # Format results
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
            
            return {
                'success': True,
                'faces_detected': len(faces),
                'faces': faces,
                'processing_time': processing_time,
                'method': method
            }
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'faces': [],
                'processing_time': 0
            }
    
    def recognize_faces(self, image: np.ndarray, method: str = 'haar_cascade') -> Dict[str, Any]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image
            method: Detection method to use
            
        Returns:
            Dictionary containing recognition results
        """
        try:
            start_time = time.time()
            
            # Initialize detector with specified method
            detector = FaceDetector(method)
            
            # Detect faces
            face_locations = detector.detect_faces(image)
            
            if not face_locations:
                return {
                    'success': True,
                    'faces_recognized': 0,
                    'faces': [],
                    'processing_time': time.time() - start_time
                }
            
            # Get face encodings
            face_encodings = self.face_encoder.encode_faces(image, face_locations)
            
            # Recognize faces
            recognition_results = self.face_recognizer.recognize_faces(face_encodings)
            
            # Format results
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
                    'confidence': confidence,
                    'encoding_id': f"enc_{i+1}" if name != "Unknown" else None
                })
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'faces_recognized': len(faces),
                'faces': faces,
                'processing_time': processing_time,
                'method': method
            }
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_recognized': 0,
                'faces': [],
                'processing_time': 0
            }
    
    def add_known_face(self, image: np.ndarray, name: str, method: str = 'haar_cascade') -> Dict[str, Any]:
        """
        Add a known face to the recognition database.
        
        Args:
            image: Input image containing the face
            name: Name associated with the face
            method: Detection method to use
            
        Returns:
            Dictionary containing the result
        """
        try:
            # Initialize detector with specified method
            detector = FaceDetector(method)
            
            # Detect faces
            face_locations = detector.detect_faces(image)
            
            if not face_locations:
                return {
                    'success': False,
                    'error': 'No face detected in the image'
                }
            
            # Get face encoding from first detected face
            face_encoding = self.face_encoder.encode_face(image, face_locations[0])
            
            if face_encoding is None:
                return {
                    'success': False,
                    'error': 'Failed to encode face'
                }
            
            # Add to recognizer
            self.face_recognizer.add_known_face(face_encoding, name)
            
            return {
                'success': True,
                'message': f'Face added successfully for {name}',
                'face_count': self.face_recognizer.get_known_faces_count()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add known face: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_known_faces(self) -> List[Dict[str, Any]]:
        """
        Get all known faces.
        
        Returns:
            List of known faces
        """
        try:
            known_faces = self.face_recognizer.get_known_faces()
            
            faces = []
            for i, (name, encoding) in enumerate(known_faces.items()):
                faces.append({
                    'id': i + 1,
                    'name': name,
                    'encoding_id': f"enc_{i+1}",
                    'created_at': "2023-12-01T10:00:00Z",  # Placeholder
                    'updated_at': "2023-12-01T10:00:00Z"  # Placeholder
                })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Failed to get known faces: {e}")
            return []
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face by name.
        
        Args:
            name: Name of the face to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            return self.face_recognizer.remove_known_face(name)
        except Exception as e:
            self.logger.error(f"Failed to remove known face: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the service.
        
        Returns:
            Dictionary containing service information
        """
        return {
            'service_name': 'FaceScanService',
            'version': '1.0.0',
            'known_faces_count': self.face_recognizer.get_known_faces_count(),
            'supported_methods': ['haar_cascade', 'hog', 'cnn', 'mtcnn'],
            'status': 'active'
        }
