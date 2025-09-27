"""
Face detection module using OpenCV and dlib.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

# Add config to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import HAAR_CASCADE_PATH
from config.model_config import FACE_DETECTION_MODELS


class FaceDetector:
    """Face detection using multiple methods."""
    
    def __init__(self, method='haar_cascade'):
        """
        Initialize face detector.
        
        Args:
            method: Detection method ('haar_cascade', 'hog', 'cnn', 'mtcnn')
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the face detector based on method."""
        try:
            if self.method == 'haar_cascade':
                self._init_haar_cascade()
            elif self.method == 'hog':
                self._init_hog()
            elif self.method == 'cnn':
                self._init_cnn()
            elif self.method == 'mtcnn':
                self._init_mtcnn()
            else:
                raise ValueError(f"Unsupported detection method: {self.method}")
                
            self.logger.info(f"Face detector initialized with method: {self.method}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {e}")
            raise
    
    def _init_haar_cascade(self):
        """Initialize Haar Cascade detector."""
        cascade_path = HAAR_CASCADE_PATH
        if not cascade_path.exists():
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
        
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
    
    def _init_hog(self):
        """Initialize HOG detector."""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
        except ImportError:
            raise ImportError("dlib is required for HOG face detection")
    
    def _init_cnn(self):
        """Initialize CNN detector."""
        try:
            import dlib
            # Load CNN face detection model
            model_path = Path(__file__).parent.parent.parent.parent / "data" / "models" / "face_detection" / "mmod_human_face_detector.dat"
            if not model_path.exists():
                raise FileNotFoundError(f"CNN model not found: {model_path}")
            
            self.detector = dlib.cnn_face_detection_model_v1(str(model_path))
        except ImportError:
            raise ImportError("dlib is required for CNN face detection")
    
    def _init_mtcnn(self):
        """Initialize MTCNN detector."""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
        except ImportError:
            raise ImportError("mtcnn is required for MTCNN face detection")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes as (x, y, width, height)
        """
        if image is None:
            raise ValueError("Input image is None")
        
        try:
            if self.method == 'haar_cascade':
                return self._detect_haar_cascade(image)
            elif self.method == 'hog':
                return self._detect_hog(image)
            elif self.method == 'cnn':
                return self._detect_cnn(image)
            elif self.method == 'mtcnn':
                return self._detect_mtcnn(image)
            else:
                raise ValueError(f"Unsupported detection method: {self.method}")
                
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_haar_cascade(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def _detect_hog(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using HOG."""
        import dlib
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector(rgb_image)
        
        # Convert to (x, y, width, height) format
        faces = []
        for detection in detections:
            x = detection.left()
            y = detection.top()
            w = detection.width()
            h = detection.height()
            faces.append((x, y, w, h))
        
        return faces
    
    def _detect_cnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using CNN."""
        import dlib
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector(rgb_image)
        
        # Convert to (x, y, width, height) format
        faces = []
        for detection in detections:
            rect = detection.rect
            x = rect.left()
            y = rect.top()
            w = rect.width()
            h = rect.height()
            faces.append((x, y, w, h))
        
        return faces
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        # Convert to (x, y, width, height) format
        faces = []
        for detection in detections:
            x, y, w, h = detection['box']
            faces.append((x, y, w, h))
        
        return faces
    
    def get_face_encodings(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
        """
        Get face encodings for detected faces.
        
        Args:
            image: Input image
            face_locations: Optional face locations (if None, will detect faces)
            
        Returns:
            List of face encodings
        """
        try:
            import face_recognition
            
            if face_locations is None:
                face_locations = self.detect_faces(image)
            
            if not face_locations:
                return []
            
            # Convert face locations to face_recognition format
            face_locations_fr = []
            for x, y, w, h in face_locations:
                # face_recognition uses (top, right, bottom, left) format
                face_locations_fr.append((y, x + w, y + h, x))
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image, face_locations_fr)
            
            return encodings
            
        except ImportError:
            self.logger.error("face_recognition library not available")
            return []
        except Exception as e:
            self.logger.error(f"Failed to get face encodings: {e}")
            return []
    
    def draw_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                   color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            face_locations: List of face bounding boxes
            color: BGR color for bounding boxes
            thickness: Thickness of bounding box lines
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for x, y, w, h in face_locations:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)
        
        return result_image
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        model_info = FACE_DETECTION_MODELS.get(self.method, {})
        return {
            'method': self.method,
            'name': model_info.get('name', self.method),
            'type': model_info.get('type', 'unknown'),
            'description': model_info.get('description', ''),
            'performance': model_info.get('performance', {})
        }
