"""
Database models for the face scan application.
"""

from .connection import get_db_connection, create_tables
import logging
from typing import List, Optional, Dict, Any
import json


class FaceModel:
    """Model for face data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_face(self, name: str, encoding_id: str) -> Optional[int]:
        """Create a new face record."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO faces (name, encoding_id)
                VALUES (?, ?)
            ''', (name, encoding_id))
            
            face_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created face: {name} with ID {face_id}")
            return face_id
            
        except Exception as e:
            self.logger.error(f"Error creating face: {e}")
            return None
    
    def get_face(self, face_id: int) -> Optional[Dict[str, Any]]:
        """Get face by ID."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, encoding_id, created_at, updated_at
                FROM faces WHERE id = ?
            ''', (face_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'encoding_id': row[2],
                    'created_at': row[3],
                    'updated_at': row[4]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting face: {e}")
            return None
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """Get all faces."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, encoding_id, created_at, updated_at
                FROM faces ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            faces = []
            for row in rows:
                faces.append({
                    'id': row[0],
                    'name': row[1],
                    'encoding_id': row[2],
                    'created_at': row[3],
                    'updated_at': row[4]
                })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Error getting all faces: {e}")
            return []
    
    def update_face(self, face_id: int, name: str) -> bool:
        """Update face name."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE faces SET name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (name, face_id))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                self.logger.info(f"Updated face ID {face_id} to name: {name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating face: {e}")
            return False
    
    def delete_face(self, face_id: int) -> bool:
        """Delete face by ID."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                self.logger.info(f"Deleted face ID {face_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error deleting face: {e}")
            return False


class ScanResultModel:
    """Model for scan results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_scan_result(self, image_path: str, scan_type: str, 
                          faces_detected: int, processing_time: float) -> Optional[int]:
        """Create a new scan result record."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO scan_results (image_path, scan_type, faces_detected, processing_time)
                VALUES (?, ?, ?, ?)
            ''', (image_path, scan_type, faces_detected, processing_time))
            
            result_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created scan result ID {result_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Error creating scan result: {e}")
            return None
    
    def get_scan_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent scan results."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, image_path, scan_type, faces_detected, processing_time, created_at
                FROM scan_results ORDER BY created_at DESC LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    'id': row[0],
                    'image_path': row[1],
                    'scan_type': row[2],
                    'faces_detected': row[3],
                    'processing_time': row[4],
                    'created_at': row[5]
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting scan results: {e}")
            return []
