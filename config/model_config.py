"""
Model configuration settings.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model types
MODEL_TYPES = {
    'face_detection': 'face_detection',
    'face_recognition': 'face_recognition',
    'face_encoding': 'face_encoding'
}

# Face detection models
FACE_DETECTION_MODELS = {
    'haar_cascade': {
        'name': 'Haar Cascade',
        'path': BASE_DIR / 'data' / 'models' / 'face_detection' / 'haarcascade_frontalface_default.xml',
        'type': 'opencv',
        'description': 'OpenCV Haar Cascade classifier for face detection'
    },
    'hog': {
        'name': 'HOG (Histogram of Oriented Gradients)',
        'type': 'dlib',
        'description': 'Dlib HOG-based face detector'
    },
    'cnn': {
        'name': 'CNN (Convolutional Neural Network)',
        'type': 'dlib',
        'description': 'Dlib CNN-based face detector (more accurate but slower)'
    },
    'mtcnn': {
        'name': 'MTCNN (Multi-task CNN)',
        'path': BASE_DIR / 'data' / 'models' / 'face_detection' / 'mtcnn',
        'type': 'tensorflow',
        'description': 'Multi-task CNN for face detection and landmark detection'
    }
}

# Face recognition models
FACE_RECOGNITION_MODELS = {
    'face_recognition': {
        'name': 'Face Recognition Library',
        'type': 'dlib',
        'description': 'Dlib-based face recognition using 128-dimensional encodings'
    },
    'facenet': {
        'name': 'FaceNet',
        'path': BASE_DIR / 'data' / 'models' / 'face_recognition' / 'facenet',
        'type': 'tensorflow',
        'description': 'Google FaceNet model for face recognition'
    },
    'arcface': {
        'name': 'ArcFace',
        'path': BASE_DIR / 'data' / 'models' / 'face_recognition' / 'arcface',
        'type': 'tensorflow',
        'description': 'ArcFace model for face recognition'
    },
    'vgg_face': {
        'name': 'VGG Face',
        'path': BASE_DIR / 'data' / 'models' / 'face_recognition' / 'vgg_face',
        'type': 'tensorflow',
        'description': 'VGG Face model for face recognition'
    }
}

# Model performance settings
MODEL_PERFORMANCE = {
    'haar_cascade': {
        'speed': 'fast',
        'accuracy': 'medium',
        'memory_usage': 'low',
        'gpu_required': False
    },
    'hog': {
        'speed': 'fast',
        'accuracy': 'good',
        'memory_usage': 'low',
        'gpu_required': False
    },
    'cnn': {
        'speed': 'slow',
        'accuracy': 'excellent',
        'memory_usage': 'high',
        'gpu_required': True
    },
    'mtcnn': {
        'speed': 'medium',
        'accuracy': 'excellent',
        'memory_usage': 'medium',
        'gpu_required': True
    },
    'face_recognition': {
        'speed': 'fast',
        'accuracy': 'good',
        'memory_usage': 'low',
        'gpu_required': False
    },
    'facenet': {
        'speed': 'medium',
        'accuracy': 'excellent',
        'memory_usage': 'medium',
        'gpu_required': True
    }
}

# Model configuration for training
TRAINING_CONFIG = {
    'face_detection': {
        'input_size': (224, 224, 3),
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'loss_function': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'data_augmentation': {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': [0.8, 1.2]
        }
    },
    'face_recognition': {
        'input_size': (224, 224, 3),
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0001,
        'optimizer': 'adam',
        'loss_function': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy'],
        'embedding_size': 128,
        'data_augmentation': {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip': True,
            'brightness_range': [0.9, 1.1],
            'zoom_range': 0.1
        }
    }
}

# Model evaluation metrics
EVALUATION_METRICS = {
    'face_detection': ['accuracy', 'precision', 'recall', 'f1_score', 'mAP'],
    'face_recognition': ['accuracy', 'precision', 'recall', 'f1_score', 'top_k_accuracy']
}

# Model optimization settings
OPTIMIZATION_SETTINGS = {
    'quantization': {
        'enabled': True,
        'type': 'int8',
        'target_accuracy_loss': 0.01
    },
    'pruning': {
        'enabled': False,
        'sparsity': 0.3,
        'target_accuracy_loss': 0.02
    },
    'distillation': {
        'enabled': False,
        'teacher_model': 'facenet',
        'student_model': 'mobile_net',
        'temperature': 3.0
    }
}

# Model deployment settings
DEPLOYMENT_SETTINGS = {
    'supported_formats': ['h5', 'pb', 'tflite', 'onnx', 'trt'],
    'default_format': 'h5',
    'optimization_level': 'balanced',  # 'speed', 'balanced', 'accuracy'
    'batch_processing': True,
    'async_processing': True
}

# Model versioning
MODEL_VERSIONING = {
    'version_format': 'major.minor.patch',
    'current_version': '1.0.0',
    'backup_versions': 5,
    'auto_update': False
}

# Model monitoring
MODEL_MONITORING = {
    'performance_tracking': True,
    'accuracy_monitoring': True,
    'drift_detection': True,
    'alert_thresholds': {
        'accuracy_drop': 0.05,
        'latency_increase': 0.2,
        'error_rate': 0.01
    }
}

# Model paths configuration
MODEL_PATHS = {
    'base_path': BASE_DIR / 'data' / 'models',
    'face_detection': BASE_DIR / 'data' / 'models' / 'face_detection',
    'face_recognition': BASE_DIR / 'data' / 'models' / 'face_recognition',
    'pretrained': BASE_DIR / 'data' / 'models' / 'pretrained',
    'custom': BASE_DIR / 'data' / 'models' / 'custom',
    'backup': BASE_DIR / 'data' / 'models' / 'backup'
}

# Ensure model directories exist
for path in MODEL_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Model loading settings
MODEL_LOADING = {
    'lazy_loading': True,
    'cache_models': True,
    'max_cache_size': 5,
    'preload_models': ['haar_cascade', 'face_recognition']
}

# Model inference settings
INFERENCE_SETTINGS = {
    'batch_size': 1,
    'max_batch_size': 32,
    'timeout': 30,
    'retry_attempts': 3,
    'fallback_model': 'haar_cascade'
}

# Model validation settings
VALIDATION_SETTINGS = {
    'test_data_ratio': 0.2,
    'validation_data_ratio': 0.2,
    'cross_validation_folds': 5,
    'minimum_samples_per_class': 10,
    'maximum_samples_per_class': 1000
}
