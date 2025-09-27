#!/usr/bin/env python3
"""
Data preprocessing script for Face Scan Project.
This script processes raw images and prepares them for training.
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import cv2
    from PIL import Image, ImageEnhance, ImageFilter
    import face_recognition
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("Please install: pip install opencv-python pillow face-recognition scikit-learn")
    sys.exit(1)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/data_preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class ImageProcessor:
    """Image processing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def resize_image(self, image, target_size=(224, 224)):
        """Resize image to target size."""
        if isinstance(image, np.ndarray):
            return cv2.resize(image, target_size)
        else:
            return image.resize(target_size, Image.Resampling.LANCZOS)
    
    def normalize_image(self, image):
        """Normalize image pixel values to [0, 1]."""
        if isinstance(image, np.ndarray):
            return image.astype(np.float32) / 255.0
        else:
            return np.array(image) / 255.0
    
    def enhance_image(self, image, brightness=1.0, contrast=1.0, sharpness=1.0):
        """Enhance image quality."""
        if isinstance(image, np.ndarray):
            # Convert to PIL for enhancement
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        return image
    
    def apply_gaussian_blur(self, image, radius=1.0):
        """Apply Gaussian blur to image."""
        if isinstance(image, np.ndarray):
            return cv2.GaussianBlur(image, (0, 0), radius)
        else:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))


class FaceExtractor:
    """Face extraction utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_faces(self, image):
        """Detect faces in image using face_recognition library."""
        try:
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = np.array(image)
            
            face_locations = face_recognition.face_locations(rgb_image)
            return face_locations
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def extract_face(self, image, face_location, padding=0.2):
        """Extract face from image with padding."""
        try:
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size
                image = np.array(image)
            
            top, right, bottom, left = face_location
            
            # Add padding
            padding_h = int((bottom - top) * padding)
            padding_w = int((right - left) * padding)
            
            top = max(0, top - padding_h)
            left = max(0, left - padding_w)
            bottom = min(height, bottom + padding_h)
            right = min(width, right + padding_w)
            
            # Extract face
            face = image[top:bottom, left:right]
            
            return face, (top, right, bottom, left)
        except Exception as e:
            self.logger.error(f"Face extraction failed: {e}")
            return None, None
    
    def extract_all_faces(self, image, min_face_size=50):
        """Extect all faces from image."""
        face_locations = self.detect_faces(image)
        faces = []
        
        for face_location in face_locations:
            face, _ = self.extract_face(image, face_location)
            if face is not None:
                # Check minimum face size
                if face.shape[0] >= min_face_size and face.shape[1] >= min_face_size:
                    faces.append(face)
        
        return faces


class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.image_processor = ImageProcessor()
        self.face_extractor = FaceExtractor()
        
    def process_single_image(self, image_path, output_dir, person_name=None):
        """Process a single image."""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                return False
            
            # Extract faces
            faces = self.face_extractor.extract_all_faces(image)
            
            if not faces:
                self.logger.warning(f"No faces found in: {image_path}")
                return False
            
            # Process each face
            for i, face in enumerate(faces):
                # Resize face
                face_resized = self.image_processor.resize_image(face, self.config.get('target_size', (224, 224)))
                
                # Normalize
                face_normalized = self.image_processor.normalize_image(face_resized)
                
                # Convert back to uint8 for saving
                face_uint8 = (face_normalized * 255).astype(np.uint8)
                
                # Save face
                if person_name:
                    face_filename = f"{person_name}_{image_path.stem}_face_{i}.jpg"
                else:
                    face_filename = f"{image_path.stem}_face_{i}.jpg"
                
                face_path = output_dir / face_filename
                cv2.imwrite(str(face_path), face_uint8)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir, person_name=None):
        """Process all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        self.logger.info(f"Processing {len(image_files)} images from {input_dir}")
        
        processed_count = 0
        for image_file in image_files:
            if self.process_single_image(image_file, output_path, person_name):
                processed_count += 1
        
        self.logger.info(f"Successfully processed {processed_count}/{len(image_files)} images")
        return processed_count
    
    def create_face_dataset(self, input_dir, output_dir):
        """Create face dataset from person directories."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        faces_dir = output_path / "faces"
        non_faces_dir = output_path / "non_faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        non_faces_dir.mkdir(parents=True, exist_ok=True)
        
        total_faces = 0
        total_non_faces = 0
        
        # Process each person directory
        for person_dir in input_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                self.logger.info(f"Processing person: {person_name}")
                
                # Process person's images
                faces_count = self.process_directory(person_dir, faces_dir, person_name)
                total_faces += faces_count
        
        # Create non-face samples (background images)
        self.logger.info("Creating non-face samples...")
        # This would typically involve extracting background regions
        # For now, we'll create some synthetic non-face samples
        
        self.logger.info(f"Dataset creation completed:")
        self.logger.info(f"  Faces: {total_faces}")
        self.logger.info(f"  Non-faces: {total_non_faces}")
        
        return total_faces, total_non_faces
    
    def split_dataset(self, data_dir, output_dir, test_ratio=0.2, val_ratio=0.2):
        """Split dataset into train/validation/test sets."""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Get all person directories
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for person_dir in person_dirs:
            person_name = person_dir.name
            
            # Get all images for this person
            image_files = list(person_dir.glob("*.jpg"))
            
            if len(image_files) < 3:
                self.logger.warning(f"Not enough images for {person_name}, skipping")
                continue
            
            # Split images
            train_files, temp_files = train_test_split(
                image_files, test_size=(test_ratio + val_ratio), random_state=42
            )
            val_files, test_files = train_test_split(
                temp_files, test_size=test_ratio/(test_ratio + val_ratio), random_state=42
            )
            
            # Copy files to respective directories
            for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
                split_dir = output_path / split / person_name
                split_dir.mkdir(parents=True, exist_ok=True)
                
                for img_file in files:
                    shutil.copy2(img_file, split_dir / img_file.name)
        
        self.logger.info("Dataset split completed")
    
    def augment_data(self, data_dir, output_dir, augment_factor=2):
        """Augment dataset with data augmentation."""
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        for img_file in data_path.rglob("*.jpg"):
            relative_path = img_file.relative_to(data_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, output_file)
        
        # Apply augmentations
        for person_dir in data_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                output_person_dir = output_path / person_name
                output_person_dir.mkdir(parents=True, exist_ok=True)
                
                for img_file in person_dir.glob("*.jpg"):
                    # Load image
                    image = cv2.imread(str(img_file))
                    
                    # Apply different augmentations
                    for i in range(augment_factor):
                        # Random brightness
                        brightness = np.random.uniform(0.8, 1.2)
                        enhanced = self.image_processor.enhance_image(
                            image, brightness=brightness
                        )
                        
                        # Random rotation
                        angle = np.random.uniform(-15, 15)
                        h, w = enhanced.shape[:2]
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated = cv2.warpAffine(enhanced, rotation_matrix, (w, h))
                        
                        # Save augmented image
                        aug_filename = f"{img_file.stem}_aug_{i}.jpg"
                        aug_path = output_person_dir / aug_filename
                        cv2.imwrite(str(aug_path), rotated)
        
        self.logger.info(f"Data augmentation completed with factor {augment_factor}")
    
    def validate_dataset(self, data_dir):
        """Validate dataset quality."""
        data_path = Path(data_dir)
        
        stats = {
            'total_images': 0,
            'total_persons': 0,
            'images_per_person': {},
            'corrupted_images': 0,
            'no_face_images': 0
        }
        
        for person_dir in data_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                stats['total_persons'] += 1
                
                person_images = 0
                for img_file in person_dir.glob("*.jpg"):
                    try:
                        # Check if image can be loaded
                        image = cv2.imread(str(img_file))
                        if image is None:
                            stats['corrupted_images'] += 1
                            continue
                        
                        # Check if face is present
                        faces = self.face_extractor.extract_all_faces(image)
                        if not faces:
                            stats['no_face_images'] += 1
                        
                        person_images += 1
                        stats['total_images'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error validating {img_file}: {e}")
                        stats['corrupted_images'] += 1
                
                stats['images_per_person'][person_name] = person_images
        
        # Generate report
        self.logger.info("Dataset Validation Report:")
        self.logger.info(f"  Total persons: {stats['total_persons']}")
        self.logger.info(f"  Total images: {stats['total_images']}")
        self.logger.info(f"  Corrupted images: {stats['corrupted_images']}")
        self.logger.info(f"  Images without faces: {stats['no_face_images']}")
        self.logger.info(f"  Average images per person: {stats['total_images'] / stats['total_persons']:.1f}")
        
        return stats


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess face recognition data')
    parser.add_argument('--input-dir', required=True, help='Input directory containing raw images')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed data')
    parser.add_argument('--mode', choices=['extract', 'split', 'augment', 'validate'], 
                       required=True, help='Preprocessing mode')
    parser.add_argument('--target-size', nargs=2, type=int, default=[224, 224],
                       help='Target image size (width height)')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--augment-factor', type=int, default=2, help='Augmentation factor')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting data preprocessing")
    
    try:
        # Load configuration
        config = {
            'target_size': tuple(args.target_size),
            'test_ratio': args.test_ratio,
            'val_ratio': args.val_ratio,
            'augment_factor': args.augment_factor
        }
        
        if args.config:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        
        # Execute preprocessing based on mode
        if args.mode == 'extract':
            preprocessor.create_face_dataset(args.input_dir, args.output_dir)
        elif args.mode == 'split':
            preprocessor.split_dataset(args.input_dir, args.output_dir, 
                                     args.test_ratio, args.val_ratio)
        elif args.mode == 'augment':
            preprocessor.augment_data(args.input_dir, args.output_dir, args.augment_factor)
        elif args.mode == 'validate':
            stats = preprocessor.validate_dataset(args.input_dir)
            # Save validation report
            with open(Path(args.output_dir) / 'validation_report.json', 'w') as f:
                json.dump(stats, f, indent=2)
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
