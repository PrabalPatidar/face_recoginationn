#!/usr/bin/env python3
"""
Model training script for Face Scan Project.
This script trains custom face detection and recognition models.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please install: pip install tensorflow scikit-learn matplotlib seaborn")
    sys.exit(1)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train_model.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class ModelTrainer:
    """Base class for model training."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.history = None
        
    def load_data(self, data_dir):
        """Load and preprocess training data."""
        raise NotImplementedError
        
    def build_model(self):
        """Build the model architecture."""
        raise NotImplementedError
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model."""
        raise NotImplementedError
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        raise NotImplementedError
        
    def save_model(self, model_path):
        """Save the trained model."""
        raise NotImplementedError


class FaceDetectionTrainer(ModelTrainer):
    """Trainer for face detection models."""
    
    def __init__(self, config):
        super().__init__(config)
        
    def load_data(self, data_dir):
        """Load face detection data."""
        self.logger.info(f"Loading data from {data_dir}")
        
        # Load positive samples (faces)
        face_dir = Path(data_dir) / "faces"
        non_face_dir = Path(data_dir) / "non_faces"
        
        X, y = [], []
        
        # Load face images
        for img_path in face_dir.glob("*.jpg"):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            X.append(img_array)
            y.append(1)  # Face label
        
        # Load non-face images
        for img_path in non_face_dir.glob("*.jpg"):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            X.append(img_array)
            y.append(0)  # Non-face label
        
        X = np.array(X) / 255.0  # Normalize
        y = np.array(y)
        
        self.logger.info(f"Loaded {len(X)} samples: {np.sum(y)} faces, {len(y) - np.sum(y)} non-faces")
        return X, y
        
    def build_model(self):
        """Build face detection model."""
        self.logger.info("Building face detection model")
        
        # Use MobileNetV2 as base
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the face detection model."""
        self.logger.info("Training face detection model")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2]
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                'data/models/face_detection/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=self.config.get('batch_size', 32)),
            steps_per_epoch=len(X_train) // self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 50),
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
        
    def evaluate(self, X_test, y_test):
        """Evaluate the face detection model."""
        self.logger.info("Evaluating face detection model")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        # Metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_test, y_pred_binary, target_names=['Non-Face', 'Face'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Face', 'Face'],
                   yticklabels=['Non-Face', 'Face'])
        plt.title('Face Detection Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/face_detection_confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }


class FaceRecognitionTrainer(ModelTrainer):
    """Trainer for face recognition models."""
    
    def __init__(self, config):
        super().__init__(config)
        
    def load_data(self, data_dir):
        """Load face recognition data."""
        self.logger.info(f"Loading data from {data_dir}")
        
        X, y, class_names = [], [], []
        
        # Load images from person directories
        data_path = Path(data_dir)
        for person_dir in data_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                class_names.append(person_name)
                
                for img_path in person_dir.glob("*.jpg"):
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    X.append(img_array)
                    y.append(len(class_names) - 1)  # Class index
        
        X = np.array(X) / 255.0  # Normalize
        y = np.array(y)
        
        self.logger.info(f"Loaded {len(X)} samples from {len(class_names)} classes")
        self.logger.info(f"Classes: {class_names}")
        
        return X, y, class_names
        
    def build_model(self):
        """Build face recognition model."""
        self.logger.info("Building face recognition model")
        
        # Use ResNet50 as base
        base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tune some layers
        base_model.trainable = True
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.config.get('num_classes', 10), activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.0001)),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the face recognition model."""
        self.logger.info("Training face recognition model")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.9, 1.1],
            zoom_range=0.1
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                'data/models/face_recognition/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=self.config.get('batch_size', 32)),
            steps_per_epoch=len(X_train) // self.config.get('batch_size', 32),
            epochs=self.config.get('epochs', 100),
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
        
    def evaluate(self, X_test, y_test):
        """Evaluate the face recognition model."""
        self.logger.info("Evaluating face recognition model")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_test, y_pred_classes)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Face Recognition Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('results/face_recognition_confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }


def plot_training_history(history, model_type):
    """Plot training history."""
    logger = logging.getLogger(__name__)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_type} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_type} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{model_type.lower().replace(" ", "_")}_training_history.png')
    plt.close()
    
    logger.info(f"Training history plot saved")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train face detection and recognition models')
    parser.add_argument('--model-type', choices=['detection', 'recognition'], required=True,
                       help='Type of model to train')
    parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set split ratio')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation set split ratio')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting model training")
    
    try:
        # Create results directory
        Path('results').mkdir(exist_ok=True)
        
        # Load configuration
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_split': args.test_split,
            'val_split': args.val_split
        }
        
        if args.config:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        
        # Initialize trainer
        if args.model_type == 'detection':
            trainer = FaceDetectionTrainer(config)
        else:
            trainer = FaceRecognitionTrainer(config)
        
        # Load data
        if args.model_type == 'detection':
            X, y = trainer.load_data(args.data_dir)
            class_names = None
        else:
            X, y, class_names = trainer.load_data(args.data_dir)
            config['num_classes'] = len(class_names)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config['test_split'], random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=config['val_split'], random_state=42, stratify=y_temp
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Build model
        model = trainer.build_model()
        model.summary()
        
        # Train model
        history = trainer.train(X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_training_history(history, args.model_type.title())
        
        # Evaluate model
        results = trainer.evaluate(X_test, y_test)
        
        # Save results
        results_file = f'results/{args.model_type}_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
