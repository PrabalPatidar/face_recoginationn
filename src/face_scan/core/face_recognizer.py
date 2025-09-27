"""
Face recognition module using face_recognition library and custom models.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
import face_recognition
from pathlib import Path

# Add config to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import FACE_ENCODING_TOLERANCE, CONFIDENCE_THRESHOLD


class FaceRecognizer:
    """Face recognition using multiple methods."""
    
    def __init__(self, tolerance: float = None):
        """
        Initialize face recognizer.
        
        Args:
            tolerance: Face encoding tolerance (lower = more strict)
        """
        self.tolerance = tolerance or FACE_ENCODING_TOLERANCE
        self.logger = logging.getLogger(__name__)
        self.known_encodings = []
        self.known_names = []
    
    def add_known_face(self, encoding: np.ndarray, name: str):
        """
        Add a known face encoding.
        
        Args:
            encoding: Face encoding
            name: Name associated with the face
        """
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self.logger.info(f"Added known face: {name}")
    
    def recognize_face(self, face_encoding: np.ndarray, confidence_threshold: float = None) -> Tuple[str, float]:
        """
        Recognize a face from its encoding.
        
        Args:
            face_encoding: Face encoding to recognize
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (name, confidence)
        """
        if not self.known_encodings:
            return "Unknown", 0.0
        
        threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        
        try:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding, 
                tolerance=self.tolerance
            )
            
            # Get face distances
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index] and face_distances[best_match_index] < threshold:
                confidence = 1.0 - face_distances[best_match_index]
                return self.known_names[best_match_index], confidence
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return "Unknown", 0.0
    
    def recognize_faces(self, face_encodings: List[np.ndarray], confidence_threshold: float = None) -> List[Tuple[str, float]]:
        """
        Recognize multiple faces.
        
        Args:
            face_encodings: List of face encodings
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of (name, confidence) tuples
        """
        results = []
        for encoding in face_encodings:
            name, confidence = self.recognize_face(encoding, confidence_threshold)
            results.append((name, confidence))
        
        return results
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> bool:
        """
        Compare two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            True if faces match, False otherwise
        """
        try:
            return face_recognition.compare_faces([encoding1], encoding2, tolerance=self.tolerance)[0]
        except Exception as e:
            self.logger.error(f"Face comparison failed: {e}")
            return False
    
    def calculate_face_distance(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate distance between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Face distance (0.0 = identical, 1.0 = completely different)
        """
        try:
            return face_recognition.face_distance([encoding1], encoding2)[0]
        except Exception as e:
            self.logger.error(f"Face distance calculation failed: {e}")
            return 1.0
    
    def get_known_faces_count(self) -> int:
        """Get the number of known faces."""
        return len(self.known_encodings)
    
    def clear_known_faces(self):
        """Clear all known faces."""
        self.known_encodings.clear()
        self.known_names.clear()
        self.logger.info("Cleared all known faces")
    
    def remove_known_face(self, name: str) -> bool:
        """
        Remove a known face by name.
        
        Args:
            name: Name of the face to remove
            
        Returns:
            True if face was removed, False if not found
        """
        try:
            index = self.known_names.index(name)
            del self.known_encodings[index]
            del self.known_names[index]
            self.logger.info(f"Removed known face: {name}")
            return True
        except ValueError:
            self.logger.warning(f"Face not found: {name}")
            return False
    
    def get_known_faces(self) -> Dict[str, np.ndarray]:
        """
        Get all known faces.
        
        Returns:
            Dictionary mapping names to encodings
        """
        return dict(zip(self.known_names, self.known_encodings))
    
    def save_known_faces(self, filepath: str):
        """
        Save known faces to file.
        
        Args:
            filepath: Path to save the faces
        """
        try:
            import pickle
            
            data = {
                'encodings': self.known_encodings,
                'names': self.known_names,
                'tolerance': self.tolerance
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Saved known faces to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save known faces: {e}")
            raise
    
    def load_known_faces(self, filepath: str):
        """
        Load known faces from file.
        
        Args:
            filepath: Path to load the faces from
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.known_encodings = data['encodings']
            self.known_names = data['names']
            self.tolerance = data.get('tolerance', self.tolerance)
            
            self.logger.info(f"Loaded {len(self.known_encodings)} known faces from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {e}")
            raise
