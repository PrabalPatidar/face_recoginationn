"""
Face encoding module for generating face embeddings.
"""

import numpy as np
import logging
from typing import List, Optional
import face_recognition
from pathlib import Path


class FaceEncoder:
    """Face encoding using face_recognition library."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def encode_face(self, image: np.ndarray, face_location: tuple = None) -> Optional[np.ndarray]:
        """
        Encode a face from an image.
        
        Args:
            image: Input image
            face_location: Optional face location (top, right, bottom, left)
            
        Returns:
            Face encoding or None if no face found
        """
        try:
            if face_location is None:
                # Detect face first
                face_locations = face_recognition.face_locations(image)
                if not face_locations:
                    return None
                face_location = face_locations[0]
            
            # Get face encoding
            encodings = face_recognition.face_encodings(image, [face_location])
            
            if encodings:
                return encodings[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Face encoding failed: {e}")
            return None
    
    def encode_faces(self, image: np.ndarray, face_locations: List[tuple] = None) -> List[np.ndarray]:
        """
        Encode multiple faces from an image.
        
        Args:
            image: Input image
            face_locations: Optional list of face locations
            
        Returns:
            List of face encodings
        """
        try:
            if face_locations is None:
                # Detect faces first
                face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return []
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image, face_locations)
            
            return encodings
            
        except Exception as e:
            self.logger.error(f"Face encoding failed: {e}")
            return []
    
    def get_encoding_size(self) -> int:
        """Get the size of face encodings."""
        return 128  # face_recognition library uses 128-dimensional encodings
