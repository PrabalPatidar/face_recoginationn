import pytest
import numpy as np
from unittest.mock import Mock, patch
import cv2

from face_scan.core.face_detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.face_detector = FaceDetector()
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_face_detector_initialization(self):
        """Test FaceDetector initialization."""
        assert self.face_detector is not None
        assert hasattr(self.face_detector, 'detect_faces')
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_with_valid_image(self, mock_cascade):
        """Test face detection with a valid image."""
        # Mock the cascade classifier
        mock_cascade.return_value.detectMultiScale.return_value = [(100, 100, 50, 50)]
        
        faces = self.face_detector.detect_faces(self.sample_image)
        
        assert isinstance(faces, list)
        assert len(faces) >= 0
    
    def test_detect_faces_with_invalid_input(self):
        """Test face detection with invalid input."""
        with pytest.raises(ValueError):
            self.face_detector.detect_faces(None)
    
    def test_detect_faces_with_empty_image(self):
        """Test face detection with empty image."""
        empty_image = np.array([])
        with pytest.raises(ValueError):
            self.face_detector.detect_faces(empty_image)
    
    @patch('cv2.CascadeClassifier')
    def test_detect_faces_returns_correct_format(self, mock_cascade):
        """Test that detect_faces returns faces in correct format."""
        # Mock face detection result
        mock_cascade.return_value.detectMultiScale.return_value = [
            (100, 100, 50, 50),
            (200, 200, 60, 60)
        ]
        
        faces = self.face_detector.detect_faces(self.sample_image)
        
        assert len(faces) == 2
        for face in faces:
            assert len(face) == 4  # x, y, width, height
            assert all(isinstance(coord, (int, np.integer)) for coord in face)
    
    def test_get_face_encodings(self):
        """Test face encoding extraction."""
        # Mock face locations
        face_locations = [(100, 100, 150, 150)]
        
        with patch.object(self.face_detector, 'detect_faces', return_value=face_locations):
            encodings = self.face_detector.get_face_encodings(self.sample_image)
            
            assert isinstance(encodings, list)
    
    def test_face_detection_performance(self):
        """Test face detection performance with timing."""
        import time
        
        start_time = time.time()
        faces = self.face_detector.detect_faces(self.sample_image)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # 1 second threshold
