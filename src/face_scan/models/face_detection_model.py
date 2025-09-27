"""
Face detection model implementation.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from .base_model import BaseModel


class FaceDetectionModel(BaseModel):
    """Face detection model using OpenCV Haar Cascades."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face detection model.
        
        Args:
            model_path: Path to the Haar cascade model file
        """
        super().__init__(model_path)
        self.logger = logging.getLogger(__name__)
        self.cascade_classifier = None
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the Haar cascade model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            self.cascade_classifier = cv2.CascadeClassifier(model_path)
            
            if self.cascade_classifier.empty():
                self.logger.error("Failed to load Haar cascade classifier")
                return False
            
            self.model_path = model_path
            self.is_loaded = True
            self.logger.info(f"Face detection model loaded from: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading face detection model: {e}")
            return False
    
    def predict(self, input_data: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image.
        
        Args:
            input_data: Input image as numpy array
            
        Returns:
            List of face bounding boxes as (x, y, width, height)
        """
        if not self.validate_input(input_data):
            return []
        
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert to grayscale if needed
            if len(input_data.shape) == 3:
                gray = cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = input_data
            
            # Detect faces
            faces = self.cascade_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of tuples
            face_locations = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            
            return face_locations
            
        except Exception as e:
            self.logger.error(f"Error during face detection: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = super().get_model_info()
        info.update({
            'model_type': 'face_detection',
            'algorithm': 'haar_cascade',
            'input_format': 'grayscale_image',
            'output_format': 'bounding_boxes',
            'cascade_loaded': self.cascade_classifier is not None and not self.cascade_classifier.empty()
        })
        return info
