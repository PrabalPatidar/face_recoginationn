"""
Face utility functions.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging


def calculate_face_center(face_location: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a face bounding box.
    
    Args:
        face_location: Face bounding box (x, y, width, height)
        
    Returns:
        Center point (x, y)
    """
    x, y, w, h = face_location
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y


def calculate_face_area(face_location: Tuple[int, int, int, int]) -> int:
    """
    Calculate the area of a face bounding box.
    
    Args:
        face_location: Face bounding box (x, y, width, height)
        
    Returns:
        Face area in pixels
    """
    x, y, w, h = face_location
    return w * h


def calculate_face_aspect_ratio(face_location: Tuple[int, int, int, int]) -> float:
    """
    Calculate the aspect ratio of a face bounding box.
    
    Args:
        face_location: Face bounding box (x, y, width, height)
        
    Returns:
        Aspect ratio (width / height)
    """
    x, y, w, h = face_location
    return w / h if h > 0 else 0.0


def is_face_valid(face_location: Tuple[int, int, int, int], 
                 min_size: int = 30, max_size: int = 1000) -> bool:
    """
    Check if a face bounding box is valid.
    
    Args:
        face_location: Face bounding box (x, y, width, height)
        min_size: Minimum face size
        max_size: Maximum face size
        
    Returns:
        True if face is valid, False otherwise
    """
    x, y, w, h = face_location
    
    # Check if coordinates are positive
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    
    # Check if face is too small or too large
    if w < min_size or h < min_size or w > max_size or h > max_size:
        return False
    
    # Check aspect ratio (faces should be roughly square)
    aspect_ratio = w / h
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    return True


def filter_valid_faces(face_locations: List[Tuple[int, int, int, int]], 
                      min_size: int = 30, max_size: int = 1000) -> List[Tuple[int, int, int, int]]:
    """
    Filter out invalid face bounding boxes.
    
    Args:
        face_locations: List of face bounding boxes
        min_size: Minimum face size
        max_size: Maximum face size
        
    Returns:
        List of valid face bounding boxes
    """
    return [face for face in face_locations if is_face_valid(face, min_size, max_size)]


def sort_faces_by_size(face_locations: List[Tuple[int, int, int, int]], 
                      reverse: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    Sort faces by their area (size).
    
    Args:
        face_locations: List of face bounding boxes
        reverse: If True, sort in descending order (largest first)
        
    Returns:
        Sorted list of face bounding boxes
    """
    return sorted(face_locations, key=calculate_face_area, reverse=reverse)


def sort_faces_by_position(face_locations: List[Tuple[int, int, int, int]], 
                          sort_by: str = 'y') -> List[Tuple[int, int, int, int]]:
    """
    Sort faces by their position.
    
    Args:
        face_locations: List of face bounding boxes
        sort_by: Sort by 'x' (left to right) or 'y' (top to bottom)
        
    Returns:
        Sorted list of face bounding boxes
    """
    if sort_by == 'x':
        return sorted(face_locations, key=lambda face: face[0])
    elif sort_by == 'y':
        return sorted(face_locations, key=lambda face: face[1])
    else:
        return face_locations


def calculate_face_distance(face1: Tuple[int, int, int, int], 
                           face2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the distance between two face centers.
    
    Args:
        face1: First face bounding box
        face2: Second face bounding box
        
    Returns:
        Distance between face centers
    """
    center1 = calculate_face_center(face1)
    center2 = calculate_face_center(face2)
    
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    
    return np.sqrt(dx * dx + dy * dy)


def find_closest_faces(face_locations: List[Tuple[int, int, int, int]], 
                      target_face: Tuple[int, int, int, int], 
                      max_distance: float = 100.0) -> List[Tuple[int, int, int, int]]:
    """
    Find faces that are close to a target face.
    
    Args:
        face_locations: List of face bounding boxes
        target_face: Target face bounding box
        max_distance: Maximum distance to consider
        
    Returns:
        List of faces within the maximum distance
    """
    close_faces = []
    
    for face in face_locations:
        if face != target_face:
            distance = calculate_face_distance(target_face, face)
            if distance <= max_distance:
                close_faces.append(face)
    
    return close_faces


def merge_overlapping_faces(face_locations: List[Tuple[int, int, int, int]], 
                           overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """
    Merge overlapping face bounding boxes.
    
    Args:
        face_locations: List of face bounding boxes
        overlap_threshold: Minimum overlap ratio to merge faces
        
    Returns:
        List of merged face bounding boxes
    """
    if not face_locations:
        return []
    
    # Sort faces by area (largest first)
    sorted_faces = sort_faces_by_size(face_locations, reverse=True)
    merged_faces = []
    
    for face in sorted_faces:
        is_overlapping = False
        
        for merged_face in merged_faces:
            if calculate_overlap_ratio(face, merged_face) > overlap_threshold:
                is_overlapping = True
                break
        
        if not is_overlapping:
            merged_faces.append(face)
    
    return merged_faces


def calculate_overlap_ratio(face1: Tuple[int, int, int, int], 
                           face2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the overlap ratio between two face bounding boxes.
    
    Args:
        face1: First face bounding box
        face2: Second face bounding box
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def extract_face_region(image: np.ndarray, face_location: Tuple[int, int, int, int], 
                       padding: float = 0.2) -> np.ndarray:
    """
    Extract face region from image with padding.
    
    Args:
        image: Input image
        face_location: Face bounding box (x, y, width, height)
        padding: Padding factor (0.2 = 20% padding)
        
    Returns:
        Extracted face region
    """
    x, y, w, h = face_location
    
    # Calculate padding
    padding_w = int(w * padding)
    padding_h = int(h * padding)
    
    # Calculate new coordinates with padding
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(image.shape[1], x + w + padding_w)
    y2 = min(image.shape[0], y + h + padding_h)
    
    # Extract face region
    face_region = image[y1:y2, x1:x2]
    
    return face_region


def resize_face_to_standard(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize face image to standard size.
    
    Args:
        image: Face image
        target_size: Target size (width, height)
        
    Returns:
        Resized face image
    """
    return cv2.resize(image, target_size)


def normalize_face_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize face image for better recognition.
    
    Args:
        image: Face image
        
    Returns:
        Normalized face image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to 3-channel if needed
    if len(image.shape) == 3:
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return equalized
