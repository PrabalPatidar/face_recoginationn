"""
Image processing utilities for face detection and recognition.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter


class ImageProcessor:
    """Image processing utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, target_size)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def enhance_image(self, image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, sharpness: float = 1.0) -> np.ndarray:
        """
        Enhance image quality.
        
        Args:
            image: Input image
            brightness: Brightness factor
            contrast: Contrast factor
            sharpness: Sharpness factor
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(sharpness)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def apply_gaussian_blur(self, image: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            radius: Blur radius
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, (0, 0), radius)
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def crop_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> np.ndarray:
        """
        Crop face from image with padding.
        
        Args:
            image: Input image
            face_location: Face location (x, y, width, height)
            padding: Padding factor
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_location
        
        # Add padding
        padding_h = int(h * padding)
        padding_w = int(w * padding)
        
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(image.shape[1], x + w + padding_w)
        y2 = min(image.shape[0], y + h + padding_h)
        
        return image[y1:y2, x1:x2]
    
    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h))
    
    def flip_image(self, image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        Flip image horizontally or vertically.
        
        Args:
            image: Input image
            horizontal: True for horizontal flip, False for vertical
            
        Returns:
            Flipped image
        """
        flip_code = 1 if horizontal else 0
        return cv2.flip(image, flip_code)
    
    def adjust_brightness_contrast(self, image: np.ndarray, alpha: float = 1.0, 
                                 beta: int = 0) -> np.ndarray:
        """
        Adjust brightness and contrast.
        
        Args:
            image: Input image
            alpha: Contrast control (1.0 = no change)
            beta: Brightness control (0 = no change)
            
        Returns:
            Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization.
        
        Args:
            image: Input image
            
        Returns:
            Equalized image
        """
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def detect_and_correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation.
        
        Args:
            image: Input image
            
        Returns:
            Corrected image
        """
        # This is a simplified version - in practice, you might use EXIF data
        # or more sophisticated orientation detection
        return image
    
    def preprocess_for_face_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for face detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale for some detectors
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        return equalized
