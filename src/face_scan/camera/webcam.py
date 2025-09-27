"""
Webcam interface for face scanning.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Callable
import threading
import time

# Add config to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import DEFAULT_CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


class Webcam:
    """Webcam interface for real-time face scanning."""
    
    def __init__(self, camera_index: int = None):
        """
        Initialize webcam.
        
        Args:
            camera_index: Camera index to use
        """
        self.camera_index = camera_index or DEFAULT_CAMERA_INDEX
        self.logger = logging.getLogger(__name__)
        self.cap = None
        self.is_running = False
        self.frame_callback = None
        self.thread = None
        
    def start(self, frame_callback: Callable[[np.ndarray], None] = None) -> bool:
        """
        Start webcam capture.
        
        Args:
            frame_callback: Callback function for frame processing
            
        Returns:
            True if started successfully, False otherwise
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
            
            self.frame_callback = frame_callback
            self.is_running = True
            
            # Start capture thread
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info(f"Webcam started on camera {self.camera_index}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting webcam: {e}")
            return False
    
    def stop(self):
        """Stop webcam capture."""
        try:
            self.is_running = False
            
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.logger.info("Webcam stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping webcam: {e}")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_running and self.cap:
            try:
                ret, frame = self.cap.read()
                
                if ret and self.frame_callback:
                    self.frame_callback(frame)
                elif not ret:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.
        
        Returns:
            Captured frame or None if failed
        """
        try:
            if not self.cap or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            return frame if ret else None
            
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if webcam is available.
        
        Returns:
            True if webcam is available, False otherwise
        """
        try:
            cap = cv2.VideoCapture(self.camera_index)
            available = cap.isOpened()
            cap.release()
            return available
        except Exception:
            return False
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Dictionary containing camera information
        """
        info = {
            'camera_index': self.camera_index,
            'is_running': self.is_running,
            'is_available': self.is_available()
        }
        
        if self.cap and self.cap.isOpened():
            info.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'brightness': int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS)),
                'contrast': int(self.cap.get(cv2.CAP_PROP_CONTRAST)),
                'saturation': int(self.cap.get(cv2.CAP_PROP_SATURATION))
            })
        
        return info
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """
        Set camera property.
        
        Args:
            property_id: Camera property ID
            value: Property value
            
        Returns:
            True if property set successfully, False otherwise
        """
        try:
            if self.cap and self.cap.isOpened():
                return self.cap.set(property_id, value)
            return False
        except Exception as e:
            self.logger.error(f"Error setting camera property: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
