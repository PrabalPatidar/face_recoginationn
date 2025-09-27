"""
Database service for managing database operations.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from face_scan.database.connection import get_db_connection
from face_scan.database.models import FaceModel, ScanResultModel


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_model = FaceModel()
        self.scan_result_model = ScanResultModel()
    
    def initialize_database(self) -> bool:
        """
        Initialize the database with required tables.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            conn = get_db_connection()
            from face_scan.database.connection import create_tables
            create_tables(conn)
            conn.close()
            
            self.logger.info("Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return False
    
    def create_face(self, name: str, encoding_id: str) -> Optional[int]:
        """
        Create a new face record.
        
        Args:
            name: Name of the person
            encoding_id: Unique encoding identifier
            
        Returns:
            Face ID if created successfully, None otherwise
        """
        try:
            face_id = self.face_model.create_face(name, encoding_id)
            if face_id:
                self.logger.info(f"Created face record: {name} (ID: {face_id})")
            return face_id
            
        except Exception as e:
            self.logger.error(f"Failed to create face: {e}")
            return None
    
    def get_face(self, face_id: int) -> Optional[Dict[str, Any]]:
        """
        Get face by ID.
        
        Args:
            face_id: Face ID
            
        Returns:
            Face data or None if not found
        """
        try:
            return self.face_model.get_face(face_id)
        except Exception as e:
            self.logger.error(f"Failed to get face: {e}")
            return None
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """
        Get all faces.
        
        Returns:
            List of all faces
        """
        try:
            return self.face_model.get_all_faces()
        except Exception as e:
            self.logger.error(f"Failed to get all faces: {e}")
            return []
    
    def update_face(self, face_id: int, name: str) -> bool:
        """
        Update face name.
        
        Args:
            face_id: Face ID
            name: New name
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            success = self.face_model.update_face(face_id, name)
            if success:
                self.logger.info(f"Updated face ID {face_id} to name: {name}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update face: {e}")
            return False
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete face by ID.
        
        Args:
            face_id: Face ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            success = self.face_model.delete_face(face_id)
            if success:
                self.logger.info(f"Deleted face ID {face_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete face: {e}")
            return False
    
    def create_scan_result(self, image_path: str, scan_type: str, 
                          faces_detected: int, processing_time: float) -> Optional[int]:
        """
        Create a new scan result record.
        
        Args:
            image_path: Path to the scanned image
            scan_type: Type of scan (detection/recognition)
            faces_detected: Number of faces detected
            processing_time: Processing time in seconds
            
        Returns:
            Scan result ID if created successfully, None otherwise
        """
        try:
            result_id = self.scan_result_model.create_scan_result(
                image_path, scan_type, faces_detected, processing_time
            )
            if result_id:
                self.logger.info(f"Created scan result ID {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Failed to create scan result: {e}")
            return None
    
    def get_scan_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent scan results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of scan results
        """
        try:
            return self.scan_result_model.get_scan_results(limit)
        except Exception as e:
            self.logger.error(f"Failed to get scan results: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            faces = self.get_all_faces()
            scan_results = self.get_scan_results(1000)  # Get more results for stats
            
            return {
                'total_faces': len(faces),
                'total_scan_results': len(scan_results),
                'recent_scans': len([r for r in scan_results if r.get('created_at')]),
                'database_status': 'active'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {
                'total_faces': 0,
                'total_scan_results': 0,
                'recent_scans': 0,
                'database_status': 'error'
            }
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Backup the database.
        
        Args:
            backup_path: Path to save the backup
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            # This is a simplified backup - in production, use proper backup methods
            import shutil
            from config.settings import DATABASE_URL
            
            if DATABASE_URL.startswith('sqlite'):
                db_path = DATABASE_URL.replace('sqlite:///', '')
                shutil.copy2(db_path, backup_path)
                self.logger.info(f"Database backed up to: {backup_path}")
                return True
            else:
                self.logger.warning("Backup not implemented for non-SQLite databases")
                return False
                
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information.
        
        Returns:
            Dictionary containing service information
        """
        stats = self.get_database_stats()
        
        return {
            'service_name': 'DatabaseService',
            'version': '1.0.0',
            'database_type': 'sqlite',
            'status': 'active',
            'stats': stats
        }
