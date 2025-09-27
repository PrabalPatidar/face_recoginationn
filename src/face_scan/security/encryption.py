"""
Data encryption and decryption module.

Provides encryption for sensitive data, file encryption, and secure key management.
"""

import os
import base64
import hashlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, Optional
import json


class DataEncryption:
    """Handles encryption and decryption of sensitive data."""
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryption with optional password.
        If no password provided, generates a random key.
        """
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_string(self, data: str) -> str:
        """Encrypt a string and return base64 encoded result."""
        encrypted_data = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_string(self, encrypted_data: str) -> str:
        """Decrypt a base64 encoded string."""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt data: {e}")
    
    def encrypt_dict(self, data: dict) -> str:
        """Encrypt a dictionary by converting to JSON first."""
        json_data = json.dumps(data, sort_keys=True)
        return self.encrypt_string(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> dict:
        """Decrypt data and parse as JSON dictionary."""
        decrypted_json = self.decrypt_string(encrypted_data)
        return json.loads(decrypted_json)
    
    def get_key(self) -> str:
        """Get the encryption key as base64 string."""
        return base64.urlsafe_b64encode(self.key).decode()


class FileEncryption:
    """Handles encryption and decryption of files."""
    
    def __init__(self, password: Optional[str] = None):
        """Initialize file encryption with optional password."""
        if password:
            self.key = self._derive_key_from_password(password)
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_file(self, input_file_path: str, output_file_path: str) -> bool:
        """
        Encrypt a file and save to output path.
        Returns True if successful, False otherwise.
        """
        try:
            with open(input_file_path, 'rb') as file:
                file_data = file.read()
            
            encrypted_data = self.cipher.encrypt(file_data)
            
            with open(output_file_path, 'wb') as file:
                file.write(encrypted_data)
            
            return True
        except Exception as e:
            print(f"Error encrypting file: {e}")
            return False
    
    def decrypt_file(self, input_file_path: str, output_file_path: str) -> bool:
        """
        Decrypt a file and save to output path.
        Returns True if successful, False otherwise.
        """
        try:
            with open(input_file_path, 'rb') as file:
                encrypted_data = file.read()
            
            decrypted_data = self.cipher.decrypt(encrypted_data)
            
            with open(output_file_path, 'wb') as file:
                file.write(decrypted_data)
            
            return True
        except Exception as e:
            print(f"Error decrypting file: {e}")
            return False
    
    def encrypt_file_inplace(self, file_path: str) -> bool:
        """Encrypt a file in place (overwrites original)."""
        temp_path = file_path + '.tmp'
        if self.encrypt_file(file_path, temp_path):
            os.replace(temp_path, file_path)
            return True
        return False
    
    def decrypt_file_inplace(self, file_path: str) -> bool:
        """Decrypt a file in place (overwrites original)."""
        temp_path = file_path + '.tmp'
        if self.decrypt_file(file_path, temp_path):
            os.replace(temp_path, file_path)
            return True
        return False


class SecureKeyManager:
    """Manages secure key storage and rotation."""
    
    def __init__(self, key_storage_path: str):
        self.key_storage_path = key_storage_path
        self.keys = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load keys from storage."""
        if os.path.exists(self.key_storage_path):
            try:
                with open(self.key_storage_path, 'r') as f:
                    self.keys = json.load(f)
            except Exception as e:
                print(f"Error loading keys: {e}")
                self.keys = {}
    
    def _save_keys(self):
        """Save keys to storage."""
        try:
            with open(self.key_storage_path, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            print(f"Error saving keys: {e}")
    
    def generate_key(self, key_name: str) -> str:
        """Generate a new encryption key."""
        key = Fernet.generate_key()
        key_str = base64.urlsafe_b64encode(key).decode()
        self.keys[key_name] = {
            'key': key_str,
            'created_at': str(os.path.getctime(self.key_storage_path) if os.path.exists(self.key_storage_path) else 0)
        }
        self._save_keys()
        return key_str
    
    def get_key(self, key_name: str) -> Optional[str]:
        """Get an existing key."""
        return self.keys.get(key_name, {}).get('key')
    
    def rotate_key(self, key_name: str) -> str:
        """Rotate (generate new) key and keep old one for decryption."""
        old_key = self.get_key(key_name)
        new_key = self.generate_key(f"{key_name}_new")
        
        # Keep old key for backward compatibility
        if old_key:
            self.keys[f"{key_name}_old"] = self.keys[key_name]
        
        return new_key
    
    def delete_key(self, key_name: str) -> bool:
        """Delete a key."""
        if key_name in self.keys:
            del self.keys[key_name]
            self._save_keys()
            return True
        return False


class HashManager:
    """Manages hashing operations for data integrity."""
    
    @staticmethod
    def sha256_hash(data: Union[str, bytes]) -> str:
        """Generate SHA-256 hash of data."""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def md5_hash(data: Union[str, bytes]) -> str:
        """Generate MD5 hash of data."""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.md5(data).hexdigest()
    
    @staticmethod
    def file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """Generate hash of a file."""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    @staticmethod
    def verify_file_integrity(file_path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify file integrity against expected hash."""
        actual_hash = HashManager.file_hash(file_path, algorithm)
        return actual_hash == expected_hash
