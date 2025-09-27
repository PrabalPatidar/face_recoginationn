"""
Camera manager for handling camera operations.
"""

import cv2
import logging
from typing import Optional, Tuple
import numpy as np

# Add config to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import DEFAULT_CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class CameraManager:
    """Manager for camera operations."""
    
    def __init__(self, camera_index: int = None):
        """
        Initialize camera manager.
        
        Args:
            camera_index: Camera index to use (default from config)
        """
        self.camera_index = camera_index or DEFAULT_CAMERA_INDEX
        self.logger = logging.getLogger(__name__)
        self.cap = None
        self.is_active = False
        
    def start_camera(self) -> bool:
        """
        Start the camera.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            self.is_active = True
            self.logger.info(f"Camera {self.camera_index} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.is_active = False
            self.logger.info("Camera stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping camera: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            Captured frame or None if failed
        """
        try:
            if not self.is_active or self.cap is None:
                return None
            
            ret, frame = self.cap.read()
            
            if ret:
                return frame
            else:
                self.logger.warning("Failed to capture frame")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if camera is available.
        
        Returns:
            True if camera is available, False otherwise
        """
        return self.is_active and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Dictionary containing camera information
        """
        info = {
            'camera_index': self.camera_index,
            'is_active': self.is_active,
            'is_available': self.is_available()
        }
        
        if self.cap is not None:
            info.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'format': self.cap.get(cv2.CAP_PROP_FORMAT)
            })
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        self.start_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_camera()
