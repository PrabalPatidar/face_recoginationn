import pytest
import numpy as np
from unittest.mock import Mock, patch

from face_scan.core.face_recognizer import FaceRecognizer


class TestFaceRecognizer:
    """Test cases for FaceRecognizer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.face_recognizer = FaceRecognizer()
        self.sample_encoding = np.random.rand(128)  # Typical face encoding size
        self.known_encodings = [
            np.random.rand(128),
            np.random.rand(128),
            np.random.rand(128)
        ]
        self.known_names = ["John", "Jane", "Bob"]
    
    def test_face_recognizer_initialization(self):
        """Test FaceRecognizer initialization."""
        assert self.face_recognizer is not None
        assert hasattr(self.face_recognizer, 'recognize_face')
    
    def test_recognize_face_with_known_face(self):
        """Test face recognition with a known face."""
        # Mock face recognition to return a match
        with patch('face_recognition.compare_faces', return_value=[True, False, False]):
            with patch('face_recognition.face_distance', return_value=[0.3, 0.8, 0.9]):
                result = self.face_recognizer.recognize_face(
                    self.sample_encoding, 
                    self.known_encodings, 
                    self.known_names
                )
                
                assert result is not None
                assert result in self.known_names
    
    def test_recognize_face_with_unknown_face(self):
        """Test face recognition with an unknown face."""
        # Mock face recognition to return no matches
        with patch('face_recognition.compare_faces', return_value=[False, False, False]):
            result = self.face_recognizer.recognize_face(
                self.sample_encoding, 
                self.known_encodings, 
                self.known_names
            )
            
            assert result == "Unknown"
    
    def test_recognize_face_with_invalid_encoding(self):
        """Test face recognition with invalid encoding."""
        with pytest.raises(ValueError):
            self.face_recognizer.recognize_face(
                None, 
                self.known_encodings, 
                self.known_names
            )
    
    def test_recognize_face_with_empty_known_encodings(self):
        """Test face recognition with empty known encodings."""
        result = self.face_recognizer.recognize_face(
            self.sample_encoding, 
            [], 
            []
        )
        
        assert result == "Unknown"
    
    def test_recognize_face_with_mismatched_arrays(self):
        """Test face recognition with mismatched encodings and names."""
        with pytest.raises(ValueError):
            self.face_recognizer.recognize_face(
                self.sample_encoding, 
                self.known_encodings, 
                ["John"]  # Mismatched length
            )
    
    def test_compare_faces(self):
        """Test face comparison functionality."""
        encoding1 = np.random.rand(128)
        encoding2 = np.random.rand(128)
        
        with patch('face_recognition.compare_faces', return_value=[True]):
            result = self.face_recognizer.compare_faces(encoding1, encoding2)
            assert result is True
    
    def test_face_distance_calculation(self):
        """Test face distance calculation."""
        encoding1 = np.random.rand(128)
        encoding2 = np.random.rand(128)
        
        with patch('face_recognition.face_distance', return_value=[0.4]):
            distance = self.face_recognizer.calculate_face_distance(encoding1, encoding2)
            assert isinstance(distance, float)
            assert 0 <= distance <= 1
    
    def test_batch_face_recognition(self):
        """Test batch face recognition."""
        encodings = [np.random.rand(128) for _ in range(3)]
        
        with patch.object(self.face_recognizer, 'recognize_face', return_value="John"):
            results = self.face_recognizer.batch_recognize_faces(
                encodings, 
                self.known_encodings, 
                self.known_names
            )
            
            assert len(results) == 3
            assert all(result == "John" for result in results)
    
    def test_confidence_threshold(self):
        """Test recognition with confidence threshold."""
        # Test with low confidence (should return Unknown)
        with patch('face_recognition.compare_faces', return_value=[True]):
            with patch('face_recognition.face_distance', return_value=[0.8]):  # High distance = low confidence
                result = self.face_recognizer.recognize_face(
                    self.sample_encoding, 
                    self.known_encodings, 
                    self.known_names,
                    confidence_threshold=0.5
                )
                
                assert result == "Unknown"
