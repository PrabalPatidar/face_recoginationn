"""
IP camera interface for face scanning.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any
import requests
import time
from urllib.parse import urlparse


class IPCamera:
    """IP camera interface for remote face scanning."""
    
    def __init__(self, camera_url: str, username: str = None, password: str = None):
        """
        Initialize IP camera.
        
        Args:
            camera_url: IP camera URL (rtsp://, http://, etc.)
            username: Camera username (optional)
            password: Camera password (optional)
        """
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        self.cap = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Connect to IP camera.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Build URL with credentials if provided
            url = self._build_url_with_credentials()
            
            # Try to open camera
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to connect to IP camera: {url}")
                return False
            
            # Test connection by reading a frame
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to read frame from IP camera")
                self.cap.release()
                return False
            
            self.is_connected = True
            self.logger.info(f"Connected to IP camera: {url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to IP camera: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IP camera."""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            self.is_connected = False
            self.logger.info("Disconnected from IP camera")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from IP camera: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from IP camera.
        
        Returns:
            Captured frame or None if failed
        """
        try:
            if not self.is_connected or not self.cap:
                return None
            
            ret, frame = self.cap.read()
            
            if ret:
                return frame
            else:
                self.logger.warning("Failed to capture frame from IP camera")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information.
        
        Returns:
            Dictionary containing camera information
        """
        info = {
            'camera_url': self.camera_url,
            'is_connected': self.is_connected,
            'has_credentials': bool(self.username and self.password)
        }
        
        if self.cap and self.cap.isOpened():
            info.update({
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'format': self.cap.get(cv2.CAP_PROP_FORMAT)
            })
        
        return info
    
    def test_connection(self) -> bool:
        """
        Test camera connection.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            if not self.is_connected:
                return False
            
            # Try to capture a frame
            frame = self.capture_frame()
            return frame is not None
            
        except Exception as e:
            self.logger.error(f"Error testing connection: {e}")
            return False
    
    def _build_url_with_credentials(self) -> str:
        """
        Build URL with credentials if provided.
        
        Returns:
            URL with credentials
        """
        if not self.username or not self.password:
            return self.camera_url
        
        # Parse URL
        parsed = urlparse(self.camera_url)
        
        # Build new URL with credentials
        if parsed.scheme in ['rtsp', 'http', 'https']:
            return f"{parsed.scheme}://{self.username}:{self.password}@{parsed.netloc}{parsed.path}"
        else:
            return self.camera_url
    
    def get_snapshot_url(self) -> Optional[str]:
        """
        Get snapshot URL for HTTP cameras.
        
        Returns:
            Snapshot URL or None if not applicable
        """
        try:
            parsed = urlparse(self.camera_url)
            
            if parsed.scheme in ['http', 'https']:
                # Common snapshot endpoints
                snapshot_paths = [
                    '/snapshot.jpg',
                    '/snapshot.cgi',
                    '/image.jpg',
                    '/video.cgi',
                    '/cgi-bin/snapshot.cgi'
                ]
                
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                
                for path in snapshot_paths:
                    snapshot_url = base_url + path
                    if self._test_snapshot_url(snapshot_url):
                        return snapshot_url
                
                return base_url + '/snapshot.jpg'  # Default
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting snapshot URL: {e}")
            return None
    
    def _test_snapshot_url(self, url: str) -> bool:
        """
        Test if snapshot URL is accessible.
        
        Args:
            url: Snapshot URL to test
            
        Returns:
            True if accessible, False otherwise
        """
        try:
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def capture_snapshot(self) -> Optional[np.ndarray]:
        """
        Capture snapshot from HTTP camera.
        
        Returns:
            Snapshot image or None if failed
        """
        try:
            snapshot_url = self.get_snapshot_url()
            if not snapshot_url:
                return None
            
            # Build URL with credentials
            if self.username and self.password:
                parsed = urlparse(snapshot_url)
                snapshot_url = f"{parsed.scheme}://{self.username}:{self.password}@{parsed.netloc}{parsed.path}"
            
            # Download snapshot
            response = requests.get(snapshot_url, timeout=10)
            response.raise_for_status()
            
            # Convert to OpenCV image
            nparr = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error capturing snapshot: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
