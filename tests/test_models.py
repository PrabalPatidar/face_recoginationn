import pytest
import numpy as np
from unittest.mock import Mock, patch

from face_scan.models.face_detection_model import FaceDetectionModel
from face_scan.models.face_recognition_model import FaceRecognitionModel
from face_scan.models.base_model import BaseModel


class TestBaseModel:
    """Test cases for BaseModel class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.base_model = BaseModel()
    
    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        assert self.base_model is not None
        assert hasattr(self.base_model, 'load_model')
        assert hasattr(self.base_model, 'predict')
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.base_model.load_model("test_path")
        
        with pytest.raises(NotImplementedError):
            self.base_model.predict(np.array([]))


class TestFaceDetectionModel:
    """Test cases for FaceDetectionModel class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.face_detection_model = FaceDetectionModel()
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_face_detection_model_initialization(self):
        """Test FaceDetectionModel initialization."""
        assert self.face_detection_model is not None
        assert hasattr(self.face_detection_model, 'load_model')
        assert hasattr(self.face_detection_model, 'predict')
    
    @patch('cv2.CascadeClassifier')
    def test_load_model(self, mock_cascade):
        """Test model loading."""
        mock_cascade.return_value = Mock()
        
        result = self.face_detection_model.load_model("test_model_path")
        
        assert result is True
        mock_cascade.assert_called_once()
    
    @patch('cv2.CascadeClassifier')
    def test_predict_with_valid_image(self, mock_cascade):
        """Test prediction with valid image."""
        # Mock the cascade classifier
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = [(100, 100, 50, 50)]
        mock_cascade.return_value = mock_classifier
        
        self.face_detection_model.load_model("test_path")
        faces = self.face_detection_model.predict(self.sample_image)
        
        assert isinstance(faces, list)
        assert len(faces) >= 0
    
    def test_predict_with_invalid_input(self):
        """Test prediction with invalid input."""
        with pytest.raises(ValueError):
            self.face_detection_model.predict(None)
    
    def test_predict_without_loaded_model(self):
        """Test prediction without loaded model."""
        with pytest.raises(RuntimeError):
            self.face_detection_model.predict(self.sample_image)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.face_detection_model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'version' in info


class TestFaceRecognitionModel:
    """Test cases for FaceRecognitionModel class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.face_recognition_model = FaceRecognitionModel()
        self.sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.sample_encoding = np.random.rand(128)
    
    def test_face_recognition_model_initialization(self):
        """Test FaceRecognitionModel initialization."""
        assert self.face_recognition_model is not None
        assert hasattr(self.face_recognition_model, 'load_model')
        assert hasattr(self.face_recognition_model, 'predict')
    
    @patch('face_recognition.face_encodings')
    def test_load_model(self, mock_face_encodings):
        """Test model loading."""
        result = self.face_recognition_model.load_model("test_model_path")
        
        assert result is True
    
    @patch('face_recognition.face_encodings')
    def test_predict_with_valid_image(self, mock_face_encodings):
        """Test prediction with valid image."""
        mock_face_encodings.return_value = [self.sample_encoding]
        
        self.face_recognition_model.load_model("test_path")
        encodings = self.face_recognition_model.predict(self.sample_image)
        
        assert isinstance(encodings, list)
        assert len(encodings) >= 0
    
    def test_predict_with_invalid_input(self):
        """Test prediction with invalid input."""
        with pytest.raises(ValueError):
            self.face_recognition_model.predict(None)
    
    def test_predict_without_loaded_model(self):
        """Test prediction without loaded model."""
        with pytest.raises(RuntimeError):
            self.face_recognition_model.predict(self.sample_image)
    
    def test_encode_face(self):
        """Test face encoding functionality."""
        with patch('face_recognition.face_encodings', return_value=[self.sample_encoding]):
            encoding = self.face_recognition_model.encode_face(self.sample_image)
            
            assert isinstance(encoding, np.ndarray)
            assert len(encoding) == 128
    
    def test_compare_encodings(self):
        """Test encoding comparison."""
        encoding1 = np.random.rand(128)
        encoding2 = np.random.rand(128)
        
        with patch('face_recognition.compare_faces', return_value=[True]):
            result = self.face_recognition_model.compare_encodings(encoding1, encoding2)
            assert result is True
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.face_recognition_model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'version' in info
        assert 'encoding_size' in info
