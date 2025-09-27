"""
File utility functions.
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import mimetypes
from datetime import datetime


def get_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash or None if failed
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logging.error(f"Error calculating file hash: {e}")
        return None


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    try:
        path = Path(file_path)
        stat = path.stat()
        
        info = {
            'path': str(path),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'exists': path.exists(),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'mime_type': mimetypes.guess_type(file_path)[0],
            'hash_md5': get_file_hash(file_path, 'md5'),
            'hash_sha256': get_file_hash(file_path, 'sha256')
        }
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting file info: {e}")
        return {
            'path': file_path,
            'exists': False,
            'error': str(e)
        }


def create_directory(dir_path: str, parents: bool = True) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        dir_path: Directory path
        parents: Create parent directories if needed
        
    Returns:
        True if directory created or exists, False otherwise
    """
    try:
        Path(dir_path).mkdir(parents=parents, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Error creating directory: {e}")
        return False


def copy_file(source: str, destination: str) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if copy successful, False otherwise
    """
    try:
        # Create destination directory if needed
        dest_dir = Path(destination).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(source, destination)
        return True
        
    except Exception as e:
        logging.error(f"Error copying file: {e}")
        return False


def move_file(source: str, destination: str) -> bool:
    """
    Move file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if move successful, False otherwise
    """
    try:
        # Create destination directory if needed
        dest_dir = Path(destination).parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.move(source, destination)
        return True
        
    except Exception as e:
        logging.error(f"Error moving file: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if deletion successful, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
        
    except Exception as e:
        logging.error(f"Error deleting file: {e}")
        return False


def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        recursive: Search recursively
        
    Returns:
        List of file paths
    """
    try:
        path = Path(directory)
        
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        return [str(f) for f in files if f.is_file()]
        
    except Exception as e:
        logging.error(f"Error listing files: {e}")
        return []


def list_directories(directory: str, recursive: bool = False) -> List[str]:
    """
    List directories in a directory.
    
    Args:
        directory: Directory path
        recursive: Search recursively
        
    Returns:
        List of directory paths
    """
    try:
        path = Path(directory)
        
        if recursive:
            dirs = list(path.rglob("*"))
        else:
            dirs = list(path.glob("*"))
        
        return [str(d) for d in dirs if d.is_dir()]
        
    except Exception as e:
        logging.error(f"Error listing directories: {e}")
        return []


def get_directory_size(directory: str) -> int:
    """
    Get total size of a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    try:
        total_size = 0
        path = Path(directory)
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
        
    except Exception as e:
        logging.error(f"Error getting directory size: {e}")
        return 0


def clean_directory(directory: str, pattern: str = "*", 
                   older_than_days: int = None) -> int:
    """
    Clean files from a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        older_than_days: Only delete files older than this many days
        
    Returns:
        Number of files deleted
    """
    try:
        deleted_count = 0
        path = Path(directory)
        
        if not path.exists():
            return 0
        
        cutoff_time = None
        if older_than_days:
            cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        
        for file_path in path.rglob(pattern):
            if file_path.is_file():
                should_delete = True
                
                if cutoff_time:
                    file_mtime = file_path.stat().st_mtime
                    should_delete = file_mtime < cutoff_time
                
                if should_delete:
                    file_path.unlink()
                    deleted_count += 1
        
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning directory: {e}")
        return 0


def backup_file(file_path: str, backup_dir: str = None) -> Optional[str]:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file
        backup_dir: Backup directory (default: same directory)
        
    Returns:
        Backup file path or None if failed
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        if backup_dir is None:
            backup_dir = path.parent
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
        backup_path = Path(backup_dir) / backup_name
        
        # Create backup directory if needed
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, backup_path)
        
        return str(backup_path)
        
    except Exception as e:
        logging.error(f"Error creating backup: {e}")
        return None


def find_duplicate_files(directory: str, algorithm: str = 'md5') -> Dict[str, List[str]]:
    """
    Find duplicate files in a directory.
    
    Args:
        directory: Directory path
        algorithm: Hash algorithm to use
        
    Returns:
        Dictionary mapping hashes to file paths
    """
    try:
        hash_to_files = {}
        path = Path(directory)
        
        for file_path in path.rglob("*"):
            if file_path.is_file():
                file_hash = get_file_hash(str(file_path), algorithm)
                
                if file_hash:
                    if file_hash not in hash_to_files:
                        hash_to_files[file_hash] = []
                    hash_to_files[file_hash].append(str(file_path))
        
        # Return only duplicates
        return {h: files for h, files in hash_to_files.items() if len(files) > 1}
        
    except Exception as e:
        logging.error(f"Error finding duplicates: {e}")
        return {}


def get_file_extension(file_path: str) -> str:
    """
    Get file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (including dot)
    """
    return Path(file_path).suffix.lower()


def is_image_file(file_path: str) -> bool:
    """
    Check if file is an image.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is an image, False otherwise
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    return get_file_extension(file_path) in image_extensions


def is_video_file(file_path: str) -> bool:
    """
    Check if file is a video.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file is a video, False otherwise
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return get_file_extension(file_path) in video_extensions


def get_available_space(directory: str) -> int:
    """
    Get available disk space for a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        Available space in bytes
    """
    try:
        stat = shutil.disk_usage(directory)
        return stat.free
    except Exception as e:
        logging.error(f"Error getting available space: {e}")
        return 0
