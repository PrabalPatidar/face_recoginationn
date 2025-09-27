"""
Image utility functions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return image
    except Exception:
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        output_path: Output file path
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(output_path, image)
        return success
    except Exception:
        return False


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop image to specified region.
    
    Args:
        image: Input image
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of crop region
        height: Height of crop region
        
    Returns:
        Cropped image
    """
    return image[y:y+height, x:x+width]


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by specified angle.
    
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


def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
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


def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Adjust brightness and contrast of image.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0 = no change)
        beta: Brightness control (0 = no change)
        
    Returns:
        Adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: Input image
        kernel_size: Size of the Gaussian kernel
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
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


def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to RGB.
    
    Args:
        image: Input BGR image
        
    Returns:
        RGB image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to BGR.
    
    Args:
        image: Input RGB image
        
    Returns:
        BGR image
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
