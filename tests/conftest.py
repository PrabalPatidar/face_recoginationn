import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_encoding():
    """Create a sample face encoding for testing."""
    return np.random.rand(128)


@pytest.fixture
def sample_face_locations():
    """Create sample face locations for testing."""
    return [(100, 100, 50, 50), (200, 200, 60, 60)]


@pytest.fixture
def sample_known_encodings():
    """Create sample known face encodings for testing."""
    return [
        np.random.rand(128),
        np.random.rand(128),
        np.random.rand(128)
    ]


@pytest.fixture
def sample_known_names():
    """Create sample known names for testing."""
    return ["John", "Jane", "Bob"]


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_cv2_cascade():
    """Mock OpenCV cascade classifier."""
    with patch('cv2.CascadeClassifier') as mock_cascade:
        mock_classifier = Mock()
        mock_classifier.detectMultiScale.return_value = [(100, 100, 50, 50)]
        mock_cascade.return_value = mock_classifier
        yield mock_cascade


@pytest.fixture
def mock_face_recognition():
    """Mock face_recognition library functions."""
    with patch('face_recognition.face_encodings') as mock_encodings, \
         patch('face_recognition.compare_faces') as mock_compare, \
         patch('face_recognition.face_distance') as mock_distance:
        
        mock_encodings.return_value = [np.random.rand(128)]
        mock_compare.return_value = [True]
        mock_distance.return_value = [0.3]
        
        yield {
            'face_encodings': mock_encodings,
            'compare_faces': mock_compare,
            'face_distance': mock_distance
        }


@pytest.fixture
def mock_database():
    """Mock database connection and operations."""
    with patch('face_scan.database.connection.get_db_connection') as mock_conn:
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn.return_value = mock_connection
        yield mock_connection


@pytest.fixture
def mock_flask_app():
    """Mock Flask application for testing."""
    with patch('face_scan.app.create_app') as mock_app:
        app = Mock()
        app.test_client.return_value = Mock()
        mock_app.return_value = app
        yield app


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'DATABASE_URL': 'sqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key',
        'FACE_DETECTION_MODEL': 'hog',
        'FACE_RECOGNITION_MODEL': 'face_recognition',
        'CONFIDENCE_THRESHOLD': 0.6,
        'FACE_ENCODING_TOLERANCE': 0.4,
        'CAMERA_WIDTH': 640,
        'CAMERA_HEIGHT': 480,
        'CAMERA_FPS': 30,
        'MAX_FACES_PER_IMAGE': 10,
        'IMAGE_QUALITY': 95,
        'BATCH_SIZE': 32
    }


@pytest.fixture
def mock_camera():
    """Mock camera interface for testing."""
    with patch('face_scan.camera.camera_manager.CameraManager') as mock_camera_manager:
        mock_camera = Mock()
        mock_camera.capture_frame.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_camera.is_available.return_value = True
        mock_camera_manager.return_value = mock_camera
        yield mock_camera


@pytest.fixture
def mock_storage_service():
    """Mock storage service for testing."""
    with patch('face_scan.services.storage_service.StorageService') as mock_storage:
        mock_service = Mock()
        mock_service.save_image.return_value = "test_image_path.jpg"
        mock_service.load_image.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_service.delete_image.return_value = True
        mock_storage.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_notification_service():
    """Mock notification service for testing."""
    with patch('face_scan.services.notification_service.NotificationService') as mock_notification:
        mock_service = Mock()
        mock_service.send_notification.return_value = True
        mock_service.send_email.return_value = True
        mock_notification.return_value = mock_service
        yield mock_service


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to tests that might take longer
        if "performance" in item.name or "batch" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to tests that test multiple components
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to individual component tests
        if "test_" in item.name and "integration" not in item.name:
            item.add_marker(pytest.mark.unit)
