"""
Face recognition model implementation.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
import face_recognition

from .base_model import BaseModel


class FaceRecognitionModel(BaseModel):
    """Face recognition model using face_recognition library."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face recognition model.
        
        Args:
            model_path: Path to the model file (not used for face_recognition library)
        """
        super().__init__(model_path)
        self.logger = logging.getLogger(__name__)
        self.known_encodings = []
        self.known_names = []
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the face recognition model.
        
        Args:
            model_path: Path to the model file (not used for face_recognition library)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # The face_recognition library doesn't require explicit model loading
            # The model is loaded automatically when the library is imported
            self.model_path = model_path
            self.is_loaded = True
            self.logger.info("Face recognition model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading face recognition model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Generate face encodings for the input image.
        
        Args:
            input_data: Input image as numpy array
            
        Returns:
            List of face encodings
        """
        if not self.validate_input(input_data):
            return []
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Generate face encodings
            encodings = face_recognition.face_encodings(input_data)
            return encodings
            
        except Exception as e:
            self.logger.error(f"Error during face encoding: {e}")
            return []
    
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
    
    def recognize_face(self, face_encoding: np.ndarray, tolerance: float = 0.4) -> Tuple[str, float]:
        """
        Recognize a face from its encoding.
        
        Args:
            face_encoding: Face encoding to recognize
            tolerance: Face encoding tolerance
            
        Returns:
            Tuple of (name, confidence)
        """
        if not self.known_encodings:
            return "Unknown", 0.0
        
        try:
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, 
                face_encoding, 
                tolerance=tolerance
            )
            
            # Get face distances
            face_distances = face_recognition.face_distance(
                self.known_encodings, 
                face_encoding
            )
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                confidence = 1.0 - face_distances[best_match_index]
                return self.known_names[best_match_index], confidence
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            self.logger.error(f"Error during face recognition: {e}")
            return "Unknown", 0.0
    
    def get_known_faces_count(self) -> int:
        """Get the number of known faces."""
        return len(self.known_encodings)
    
    def clear_known_faces(self):
        """Clear all known faces."""
        self.known_encodings.clear()
        self.known_names.clear()
        self.logger.info("Cleared all known faces")
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'model_type': 'face_recognition',
            'algorithm': 'face_recognition_library',
            'input_format': 'rgb_image',
            'output_format': 'face_encodings',
            'encoding_size': 128,
            'known_faces_count': len(self.known_encodings)
        })
        return info
