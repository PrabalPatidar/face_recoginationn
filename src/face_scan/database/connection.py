"""
Database connection utilities.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

# Add config to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "config"))

from config.settings import DATABASE_URL


def get_db_connection():
    """Get database connection."""
    try:
        if DATABASE_URL.startswith('sqlite'):
            # Extract database path from SQLite URL
            db_path = DATABASE_URL.replace('sqlite:///', '')
            # Create directory if it doesn't exist
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return sqlite3.connect(db_path)
        else:
            # For other database types, you would implement appropriate connections
            raise NotImplementedError(f"Database type not supported: {DATABASE_URL}")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise


def create_tables(conn):
    """Create database tables."""
    cursor = conn.cursor()
    
    # Create faces table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding_id TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create face_encodings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encoding_id TEXT NOT NULL,
            encoding_data BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (encoding_id) REFERENCES faces (encoding_id)
        )
    ''')
    
    # Create scan_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            scan_type TEXT NOT NULL,
            faces_detected INTEGER DEFAULT 0,
            processing_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    logging.info("Database tables created successfully")
