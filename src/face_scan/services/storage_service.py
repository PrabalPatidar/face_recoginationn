"""
Storage service for managing files and data.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List
import uuid
from datetime import datetime

# Add config to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import UPLOAD_FOLDER, PROCESSED_FOLDER


class StorageService:
    """Service for file storage operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.upload_folder = Path(UPLOAD_FOLDER)
        self.processed_folder = Path(PROCESSED_FOLDER)
        
        # Create directories if they don't exist
        self.upload_folder.mkdir(parents=True, exist_ok=True)
        self.processed_folder.mkdir(parents=True, exist_ok=True)
    
    def save_image(self, image_data: bytes, filename: str = None, subfolder: str = None) -> str:
        """
        Save image data to storage.
        
        Args:
            image_data: Image data as bytes
            filename: Optional filename (will generate if not provided)
            subfolder: Optional subfolder to save in
            
        Returns:
            Path to saved file
        """
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"image_{timestamp}_{unique_id}.jpg"
            
            # Determine save directory
            if subfolder:
                save_dir = self.upload_folder / subfolder
            else:
                save_dir = self.upload_folder
            
            # Create directory if it doesn't exist
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = save_dir / filename
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            self.logger.info(f"Saved image to: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            raise
    
    def load_image(self, file_path: str) -> Optional[bytes]:
        """
        Load image data from storage.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Image data as bytes or None if failed
        """
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to load image: {e}")
            return None
    
    def delete_image(self, file_path: str) -> bool:
        """
        Delete image from storage.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Deleted image: {file_path}")
                return True
            else:
                self.logger.warning(f"Image not found: {file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to delete image: {e}")
            return False
    
    def list_images(self, subfolder: str = None) -> List[str]:
        """
        List all images in storage.
        
        Args:
            subfolder: Optional subfolder to list
            
        Returns:
            List of image file paths
        """
        try:
            if subfolder:
                search_dir = self.upload_folder / subfolder
            else:
                search_dir = self.upload_folder
            
            if not search_dir.exists():
                return []
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images = []
            
            for file_path in search_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    images.append(str(file_path))
            
            return sorted(images)
            
        except Exception as e:
            self.logger.error(f"Failed to list images: {e}")
            return []
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            self.logger.error(f"Failed to get file size: {e}")
            return 0
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        try:
            stat = os.stat(file_path)
            return {
                'path': file_path,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'exists': True
            }
        except Exception as e:
            self.logger.error(f"Failed to get file info: {e}")
            return {
                'path': file_path,
                'size': 0,
                'created': None,
                'modified': None,
                'exists': False
            }
    
    def cleanup_old_files(self, days: int = 7) -> int:
        """
        Clean up old files.
        
        Args:
            days: Number of days to keep files
            
        Returns:
            Number of files deleted
        """
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            for file_path in self.upload_folder.rglob('*'):
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old files: {e}")
            return 0
    
    def get_storage_info(self) -> dict:
        """
        Get storage information.
        
        Returns:
            Dictionary containing storage information
        """
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.upload_folder.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'upload_folder': str(self.upload_folder),
                'processed_folder': str(self.processed_folder),
                'total_files': file_count,
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {
                'upload_folder': str(self.upload_folder),
                'processed_folder': str(self.processed_folder),
                'total_files': 0,
                'total_size': 0,
                'total_size_mb': 0
            }
