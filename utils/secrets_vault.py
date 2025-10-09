"""
Secrets Vault - Secure credential storage using Windows DPAPI or Fernet encryption.

This module provides a secure way to store and retrieve sensitive information
like API keys, passwords, and other credentials. It uses Windows DPAPI when
available, falling back to Fernet encryption for cross-platform compatibility.

Security Features:
- Windows DPAPI integration for maximum security on Windows
- Fernet fallback for cross-platform support
- User-only file permissions
- Encrypted storage file with metadata
- No logging of secret values (only key names)
- Atomic writes via temp file + replace
- Rotation between encryption methods
- Export/import with passphrase protection
"""

import os
import json
import base64
import logging
import tempfile
import shutil
import tarfile
import gzip
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import stat

# Configure logging - never log secret values
logger = logging.getLogger(__name__)

# DPAPI constants
_DESC = b"ShivXVault"      # constant description
_ENTROPY = b"shivx-entropy-v1"  # static optional entropy (public, not secret)

try:
    import win32crypt
    DPAPI_AVAILABLE = True
    logger.info("DPAPI available - using Windows Data Protection API")
except ImportError:
    DPAPI_AVAILABLE = False
    logger.info("DPAPI not available - using Fernet encryption")


def _dpapi_encrypt(plaintext: bytes) -> str:
    """Encrypt data using Windows DPAPI."""
    try:
        blob = win32crypt.CryptProtectData(plaintext, _DESC, _ENTROPY, None, None, 0)
        return base64.b64encode(blob).decode("utf-8")
    except Exception as e:
        logger.error(f"DPAPI encryption failed: {e}")
        raise


def _dpapi_decrypt(blob_b64: str) -> bytes:
    """Decrypt data using Windows DPAPI."""
    try:
        blob = base64.b64decode(blob_b64.encode("utf-8"))
        result = win32crypt.CryptUnprotectData(blob, _ENTROPY, None, None, 0)
        
        # Handle different return formats from pywin32
        if isinstance(result, tuple):
            # Some pywin32 builds return (data, description)
            plaintext = result[0]
        else:
            # Some return just the data
            plaintext = result
            
        return plaintext
    except Exception as e:
        logger.error(f"DPAPI decryption failed: {e}")
        raise


class SecretsVault:
    """Secure secrets storage with DPAPI (Windows) or Fernet fallback."""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the secrets vault.
        
        Args:
            storage_path: Path to storage file (defaults to var/secrets/kv.json)
        """
        if storage_path is None:
            self.storage_path = Path("var/secrets/kv.json")
        else:
            self.storage_path = Path(storage_path)
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption
        self._encryption_method = self._get_preferred_encryption_method()
        
        if self._encryption_method == "fernet":
            self._fernet_key = self._get_or_create_fernet_key()
            logger.info("Using Fernet encryption (cross-platform)")
        else:
            # Initialize Fernet key for potential fallback
            self._fernet_key = self._get_or_create_fernet_key()
            logger.info("Using Windows DPAPI for encryption")
        
        # Load existing secrets
        self._secrets = self._load_secrets()
        
        # Create empty storage file if it doesn't exist
        if not self.storage_path.exists():
            self._save_secrets()
    
    def _get_preferred_encryption_method(self) -> str:
        """Determine the preferred encryption method."""
        if DPAPI_AVAILABLE and os.name == "nt":
            return "dpapi"
        return "fernet"
    
    def _get_or_create_fernet_key(self) -> bytes:
        """Get existing Fernet key or create a new one."""
        key_file = self.storage_path.parent / ".fernet_key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    key = f.read()
                # Set user-only permissions on key file
                if hasattr(os, 'chmod'):
                    os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
                return key
            except Exception as e:
                logger.warning(f"Could not read existing Fernet key: {e}")
        
        # Create new key
        key = Fernet.generate_key()
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set user-only permissions on key file
            if hasattr(os, 'chmod'):
                os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)
            logger.info("Generated new Fernet encryption key")
        except Exception as e:
            logger.error(f"Could not save Fernet key: {e}")
        
        return key
    
    def _encrypt_value(self, value: str, method: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt a value using the specified or preferred method."""
        if method is None:
            method = self._encryption_method
        
        now = datetime.now(timezone.utc).isoformat()
        
        if method == "dpapi":
            try:
                # Ensure value is bytes for DPAPI
                if isinstance(value, str):
                    value_bytes = value.encode('utf-8')
                else:
                    value_bytes = value
                encrypted = _dpapi_encrypt(value_bytes)
                return {
                    "method": "dpapi",
                    "value": encrypted,
                    "created": now,
                    "rotated": None
                }
            except Exception as e:
                logger.error(f"DPAPI encryption failed: {e}")
                # Fallback to Fernet
                logger.info("Falling back to Fernet encryption")
                return self._encrypt_value(value, "fernet")
        else:
            # Fernet encryption
            try:
                fernet = Fernet(self._fernet_key)
                encrypted = fernet.encrypt(value.encode('utf-8'))
                return {
                    "method": "fernet",
                    "value": base64.b64encode(encrypted).decode('utf-8'),
                    "created": now,
                    "rotated": None
                }
            except Exception as e:
                logger.error(f"Fernet encryption failed: {e}")
                raise
    
    def _decrypt_value(self, entry: Dict[str, Any]) -> str:
        """Decrypt a value from an entry."""
        method = entry.get("method", "fernet")
        encrypted_value = entry.get("value", "")
        
        if method == "dpapi":
            try:
                decrypted = _dpapi_decrypt(encrypted_value)
                return decrypted.decode('utf-8')
            except Exception as e:
                logger.warning(f"DPAPI decryption failed for entry: {e}")
                # Try Fernet as fallback if available
                if self._encryption_method == "fernet":
                    logger.info("Attempting Fernet fallback decryption")
                    try:
                        fernet = Fernet(self._fernet_key)
                        encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
                        decrypted = fernet.decrypt(encrypted_bytes)
                        return decrypted.decode('utf-8')
                    except Exception as fernet_e:
                        logger.error(f"Fernet fallback also failed: {fernet_e}")
                        raise
                else:
                    raise
        else:
            # Fernet decryption
            try:
                fernet = Fernet(self._fernet_key)
                encrypted_bytes = base64.b64decode(encrypted_value.encode('utf-8'))
                decrypted = fernet.decrypt(encrypted_bytes)
                return decrypted.decode('utf-8')
            except Exception as e:
                logger.error(f"Fernet decryption failed: {e}")
                raise
    
    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from storage file."""
        if not self.storage_path.exists():
            return {}
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Handle both old format (direct key-value) and new format (with metadata)
            decrypted_secrets = {}
            for key, entry in data.items():
                try:
                    if isinstance(entry, dict) and "value" in entry:
                        # New format with metadata
                        decrypted_value = self._decrypt_value(entry)
                    else:
                        # Old format - try to decrypt directly
                        logger.debug(f"Attempting legacy decryption for key '{key}'")
                        if self._encryption_method == "dpapi":
                            try:
                                decrypted_value = _dpapi_decrypt(entry)
                                if isinstance(decrypted_value, bytes):
                                    decrypted_value = decrypted_value.decode('utf-8')
                            except Exception:
                                # Try Fernet fallback
                                fernet = Fernet(self._fernet_key)
                                encrypted_bytes = base64.b64decode(entry.encode('utf-8'))
                                decrypted_value = fernet.decrypt(encrypted_bytes).decode('utf-8')
                        else:
                            fernet = Fernet(self._fernet_key)
                            encrypted_bytes = base64.b64decode(entry.encode('utf-8'))
                            decrypted_value = fernet.decrypt(encrypted_bytes).decode('utf-8')
                    
                    decrypted_secrets[key] = decrypted_value
                except Exception as e:
                    logger.error(f"Failed to decrypt secret '{key}': {e}")
                    # Skip corrupted secrets
                    continue
            
            logger.info(f"Loaded {len(decrypted_secrets)} secrets from vault")
            return decrypted_secrets
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return {}
    
    def _save_secrets(self):
        """Save secrets to storage file with atomic write."""
        try:
            # Prepare encrypted data
            encrypted_data = {}
            for key, value in self._secrets.items():
                # Check if we need to re-encrypt (e.g., after rotation)
                if key not in self._metadata_cache:
                    encrypted_data[key] = self._encrypt_value(value)
                else:
                    encrypted_data[key] = self._metadata_cache[key]
            
            # Atomic write via temp file
            temp_file = tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.storage_path.parent,
                delete=False,
                suffix='.tmp'
            )
            
            try:
                json.dump(encrypted_data, temp_file, indent=2)
                temp_file.close()
                
                # Atomic replace
                shutil.move(temp_file.name, self.storage_path)
                
                # Update metadata cache
                self._metadata_cache = encrypted_data
                
                # Set user-only permissions on storage file
                if hasattr(os, 'chmod'):
                    os.chmod(self.storage_path, stat.S_IRUSR | stat.S_IWUSR)
                
                logger.info(f"Saved {len(self._secrets)} secrets to vault")
                
            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                raise e
                
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise
    
    def put(self, name: str, value: str) -> bool:
        """
        Store a secret value.
        
        Args:
            name: Secret name/key
            value: Secret value to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not name or not value:
                logger.error("Secret name and value cannot be empty")
                return False
            
            # Log only the key name, never the value
            logger.info(f"Storing secret: {name}")
            
            self._secrets[name] = value
            self._save_secrets()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store secret '{name}': {e}")
            return False
    
    def get(self, name: str) -> Optional[str]:
        """
        Retrieve a secret value.
        
        Args:
            name: Secret name/key
            
        Returns:
            Secret value if found, None otherwise
        """
        try:
            if not name:
                logger.error("Secret name cannot be empty")
                return None
            
            # Log only the key name, never the value
            logger.info(f"Retrieving secret: {name}")
            
            value = self._secrets.get(name)
            if value is None:
                logger.warning(f"Secret '{name}' not found")
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret '{name}': {e}")
            return None
    
    def delete(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name/key to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not name:
                logger.error("Secret name cannot be empty")
                return False
            
            # Log only the key name, never the value
            logger.info(f"Deleting secret: {name}")
            
            if name in self._secrets:
                del self._secrets[name]
                self._save_secrets()
                return True
            else:
                logger.warning(f"Secret '{name}' not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete secret '{name}': {e}")
            return False
    
    def list(self) -> List[str]:
        """
        List all secret names.
        
        Returns:
            List of secret names
        """
        try:
            secret_names = list(self._secrets.keys())
            logger.info(f"Listed {len(secret_names)} secrets")
            return secret_names
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def exists(self, name: str) -> bool:
        """
        Check if a secret exists.
        
        Args:
            name: Secret name/key to check
            
        Returns:
            True if secret exists, False otherwise
        """
        return name in self._secrets
    
    def clear(self) -> bool:
        """
        Clear all secrets.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Clearing all secrets")
            self._secrets.clear()
            self._save_secrets()
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear secrets: {e}")
            return False
    
    def rotate(self, method: str = None) -> bool:
        """
        Rotate all secrets to a different encryption method.
        
        Args:
            method: Target encryption method ("dpapi" or "fernet")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if method is None:
                method = self._encryption_method
            
            if method not in ["dpapi", "fernet"]:
                logger.error(f"Invalid encryption method: {method}")
                return False
            
            if method == self._encryption_method:
                logger.info(f"Already using {method} encryption")
                return True
            
            logger.info(f"Rotating secrets from {self._encryption_method} to {method}")
            
            # Re-encrypt all secrets with new method
            now = datetime.now(timezone.utc).isoformat()
            for key, value in self._secrets.items():
                encrypted_entry = self._encrypt_value(value, method)
                encrypted_entry["rotated"] = now
                self._metadata_cache[key] = encrypted_entry
            
            # Update encryption method
            self._encryption_method = method
            if method == "fernet":
                self._fernet_key = self._get_or_create_fernet_key()
            
            # Save with new encryption
            self._save_secrets()
            
            logger.info(f"Successfully rotated {len(self._secrets)} secrets to {method}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rotate secrets: {e}")
            return False
    
    def export(self, filepath: str, passphrase: str) -> bool:
        """
        Export secrets to an encrypted archive.
        
        Args:
            filepath: Path to export file
            passphrase: Passphrase for encryption
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_path = Path(filepath)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary directory for export
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy current vault file
                vault_copy = temp_path / "kv.json"
                shutil.copy2(self.storage_path, vault_copy)
                
                # Create tar.gz archive
                archive_path = temp_path / "secrets.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(vault_copy, arcname="kv.json")
                
                # Encrypt archive with passphrase
                salt = os.urandom(16)
                kdf = Scrypt(
                    salt=salt,
                    length=32,
                    n=2**14,
                    r=8,
                    p=1,
                )
                key = kdf.derive(passphrase.encode())
                
                # Use Fernet for the export encryption
                fernet = Fernet(base64.b64encode(key))
                
                with open(archive_path, 'rb') as f:
                    archive_data = f.read()
                
                encrypted_data = fernet.encrypt(archive_data)
                
                # Write encrypted export with salt
                with open(export_path, 'wb') as f:
                    f.write(salt)
                    f.write(encrypted_data)
                
                logger.info(f"Exported {len(self._secrets)} secrets to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export secrets: {e}")
            return False
    
    def import_secrets(self, filepath: str, passphrase: str, merge_policy: str = "skip") -> bool:
        """
        Import secrets from an encrypted archive.
        
        Args:
            filepath: Path to import file
            passphrase: Passphrase for decryption
            merge_policy: How to handle conflicts ("overwrite" or "skip")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import_path = Path(filepath)
            if not import_path.exists():
                logger.error(f"Import file not found: {filepath}")
                return False
            
            with open(import_path, 'rb') as f:
                salt = f.read(16)
                encrypted_data = f.read()
            
            # Decrypt with passphrase
            kdf = Scrypt(
                salt=salt,
                length=32,
                n=2**14,
                r=8,
                p=1,
            )
            key = kdf.derive(passphrase.encode())
            
            fernet = Fernet(base64.b64encode(key))
            archive_data = fernet.decrypt(encrypted_data)
            
            # Extract archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                archive_path = temp_path / "secrets.tar.gz"
                
                with open(archive_path, 'wb') as f:
                    f.write(archive_data)
                
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(temp_path)
                
                # Load imported secrets
                imported_vault = temp_path / "kv.json"
                if not imported_vault.exists():
                    logger.error("Invalid export file format")
                    return False
                
                with open(imported_vault, 'r') as f:
                    imported_data = json.load(f)
                
                # Merge secrets according to policy
                imported_count = 0
                skipped_count = 0
                
                for key, entry in imported_data.items():
                    if key in self._secrets and merge_policy == "skip":
                        skipped_count += 1
                        continue
                    
                    try:
                        # Decrypt imported secret
                        if isinstance(entry, dict) and "value" in entry:
                            decrypted_value = self._decrypt_value(entry)
                        else:
                            # Handle legacy format
                            if self._encryption_method == "dpapi":
                                decrypted_value = _dpapi_decrypt(entry).decode('utf-8')
                            else:
                                fernet = Fernet(self._fernet_key)
                                encrypted_bytes = base64.b64decode(entry.encode('utf-8'))
                                decrypted_value = fernet.decrypt(encrypted_bytes).decode('utf-8')
                        
                        # Store with current encryption method
                        self._secrets[key] = decrypted_value
                        imported_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to import secret '{key}': {e}")
                        continue
                
                # Save merged secrets
                self._save_secrets()
                
                logger.info(f"Imported {imported_count} secrets, skipped {skipped_count}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to import secrets: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get vault information.
        
        Returns:
            Dictionary with vault metadata
        """
        return {
            "encryption_method": self._encryption_method,
            "storage_path": str(self.storage_path),
            "secret_count": len(self._secrets),
            "dpapi_available": DPAPI_AVAILABLE,
            "storage_file_exists": self.storage_path.exists(),
            "storage_file_size": self.storage_path.stat().st_size if self.storage_path.exists() else 0
        }
    
    @property
    def _metadata_cache(self) -> Dict[str, Any]:
        """Get or create metadata cache."""
        if not hasattr(self, '_metadata_cache_store'):
            self._metadata_cache_store = {}
        return self._metadata_cache_store
    
    @_metadata_cache.setter
    def _metadata_cache(self, value: Dict[str, Any]):
        """Set metadata cache."""
        self._metadata_cache_store = value


# Convenience functions for easy access
def get_secret(name: str) -> Optional[str]:
    """Get a secret value using default vault."""
    vault = SecretsVault()
    return vault.get(name)


def put_secret(name: str, value: str) -> bool:
    """Store a secret value using default vault."""
    vault = SecretsVault()
    return vault.put(name, value)


def delete_secret(name: str) -> bool:
    """Delete a secret using default vault."""
    vault = SecretsVault()
    return vault.delete(name)


def list_secrets() -> List[str]:
    """List all secret names using default vault."""
    vault = SecretsVault()
    return vault.list()
