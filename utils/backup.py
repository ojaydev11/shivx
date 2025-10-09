"""
Backup utility for ShivX system.
Creates timestamped zip archives of critical system components.
"""

import os
import zipfile
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backup creation, listing, and restoration of ShivX system."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize backup manager with project root path."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.backup_dir = self.project_root / "var" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Define backup components and their paths
        self.backup_components = {
            "configs": [
                "config/*.yaml",
                "config/*.json",
                "config/policy/*.yaml"
            ],
            "runs": ["var/runs/"],
            "reports": ["var/reports/"],
            "security": ["var/security/goal_execution_audit.jsonl"],
            "logs": ["logs/"],
            "memory": ["memory/"],
            "templates": ["templates/"]
        }
    
    def create_backup(self, description: str = "", include_logs: bool = True) -> Tuple[Path, Dict]:
        """
        Create a timestamped backup of critical system components.
        
        Args:
            description: Optional description for the backup
            include_logs: Whether to include log files
            
        Returns:
            Tuple of (backup_path, backup_metadata)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"shivx_backup_{timestamp}"
        if description:
            backup_name += f"_{description.replace(' ', '_')}"
        
        backup_path = self.backup_dir / f"{backup_name}.zip"
        temp_dir = self.project_root / "temp" / f"backup_{timestamp}"
        
        try:
            # Create temporary directory for backup preparation
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare backup metadata
            metadata = {
                "backup_id": backup_name,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "project_root": str(self.project_root),
                "components": {},
                "file_count": 0,
                "total_size": 0
            }
            
            # Create zip file
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for component_name, patterns in self.backup_components.items():
                    if component_name == "logs" and not include_logs:
                        continue
                    
                    component_files = []
                    for pattern in patterns:
                        pattern_path = self.project_root / pattern
                        
                        if pattern.endswith('/'):
                            # Directory pattern
                            if pattern_path.exists():
                                for file_path in pattern_path.rglob('*'):
                                    if file_path.is_file():
                                        component_files.append(file_path)
                        else:
                            # File pattern
                            if '*' in pattern:
                                # Glob pattern
                                for file_path in self.project_root.glob(pattern):
                                    if file_path.is_file():
                                        component_files.append(file_path)
                            else:
                                # Single file
                                if pattern_path.exists():
                                    component_files.append(pattern_path)
                    
                    # Add files to zip
                    for file_path in component_files:
                        try:
                            # Calculate relative path for zip
                            rel_path = file_path.relative_to(self.project_root)
                            zipf.write(file_path, rel_path)
                            
                            # Update metadata
                            file_size = file_path.stat().st_size
                            metadata["total_size"] += file_size
                            metadata["file_count"] += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to backup {file_path}: {e}")
                    
                    metadata["components"][component_name] = {
                        "files": len(component_files),
                        "patterns": patterns
                    }
                
                # Add metadata to zip
                zipf.writestr("backup_metadata.json", json.dumps(metadata, indent=2))
            
            logger.info(f"Backup created successfully: {backup_path}")
            logger.info(f"Files: {metadata['file_count']}, Size: {metadata['total_size']} bytes")
            
            return backup_path, metadata
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            # Cleanup on failure
            if backup_path.exists():
                backup_path.unlink()
            raise
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups with metadata.
        
        Returns:
            List of backup metadata dictionaries
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.zip"):
            try:
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "backup_metadata.json" in zipf.namelist():
                        metadata_content = zipf.read("backup_metadata.json")
                        metadata = json.loads(metadata_content.decode('utf-8'))
                        metadata["file_path"] = str(backup_file)
                        metadata["file_size"] = backup_file.stat().st_size
                        metadata["created_date"] = datetime.fromtimestamp(
                            backup_file.stat().st_mtime
                        ).isoformat()
                        backups.append(metadata)
                    else:
                        # Legacy backup without metadata
                        backups.append({
                            "file_path": str(backup_file),
                            "file_size": backup_file.stat().st_size,
                            "created_date": datetime.fromtimestamp(
                                backup_file.stat().st_mtime
                            ).isoformat(),
                            "backup_id": backup_file.stem,
                            "timestamp": "unknown",
                            "description": "Legacy backup",
                            "components": {},
                            "file_count": 0,
                            "total_size": 0
                        })
            except Exception as e:
                logger.warning(f"Failed to read backup {backup_file}: {e}")
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x["created_date"], reverse=True)
        return backups
    
    def get_backup_info(self, backup_path: Path) -> Optional[Dict]:
        """
        Get detailed information about a specific backup.
        
        Args:
            backup_path: Path to the backup zip file
            
        Returns:
            Backup metadata dictionary or None if invalid
        """
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                if "backup_metadata.json" in zipf.namelist():
                    metadata_content = zipf.read("backup_metadata.json")
                    metadata = json.loads(metadata_content.decode('utf-8'))
                    metadata["file_path"] = str(backup_path)
                    metadata["file_size"] = backup_path.stat().st_size
                    metadata["file_list"] = zipf.namelist()
                    return metadata
                else:
                    return None
        except Exception as e:
            logger.error(f"Failed to read backup info: {e}")
            return None
    
    def validate_backup(self, backup_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate a backup file for integrity and completeness.
        
        Args:
            backup_path: Path to the backup zip file
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                # Check if zip file is valid
                zipf.testzip()
                
                # Check for required metadata
                if "backup_metadata.json" not in zipf.namelist():
                    issues.append("Missing backup metadata")
                
                # Check for critical components
                critical_components = ["configs", "runs", "reports"]
                metadata = self.get_backup_info(backup_path)
                
                if metadata:
                    for component in critical_components:
                        if component not in metadata.get("components", {}):
                            issues.append(f"Missing critical component: {component}")
                
                # Check file count
                if metadata and metadata.get("file_count", 0) == 0:
                    issues.append("Backup contains no files")
                
        except Exception as e:
            issues.append(f"Backup file is corrupted: {e}")
        
        return len(issues) == 0, issues
    
    def cleanup_old_backups(self, max_backups: int = 10, max_age_days: int = 30) -> List[Path]:
        """
        Clean up old backups based on count and age limits.
        
        Args:
            max_backups: Maximum number of backups to keep
            max_age_days: Maximum age of backups in days
            
        Returns:
            List of deleted backup file paths
        """
        backups = self.list_backups()
        deleted_files = []
        
        # Sort by creation date
        backups.sort(key=lambda x: x["created_date"])
        
        # Remove old backups by age
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        for backup in backups:
            try:
                backup_date = datetime.fromisoformat(backup["created_date"]).timestamp()
                if backup_date < cutoff_date:
                    backup_path = Path(backup["file_path"])
                    if backup_path.exists():
                        backup_path.unlink()
                        deleted_files.append(backup_path)
                        logger.info(f"Deleted old backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to process backup for cleanup: {e}")
        
        # Remove excess backups by count
        if len(backups) > max_backups:
            excess_backups = backups[:-max_backups]
            for backup in excess_backups:
                backup_path = Path(backup["file_path"])
                if backup_path.exists():
                    backup_path.unlink()
                    deleted_files.append(backup_path)
                    logger.info(f"Deleted excess backup: {backup_path}")
        
        return deleted_files


def create_backup(description: str = "", include_logs: bool = True) -> Tuple[Path, Dict]:
    """Convenience function to create a backup."""
    manager = BackupManager()
    return manager.create_backup(description, include_logs)


def list_backups() -> List[Dict]:
    """Convenience function to list backups."""
    manager = BackupManager()
    return manager.list_backups()


if __name__ == "__main__":
    # Test backup creation
    try:
        backup_path, metadata = create_backup("test_backup")
        print(f"Backup created: {backup_path}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        backups = list_backups()
        print(f"Total backups: {len(backups)}")
        
    except Exception as e:
        print(f"Backup test failed: {e}")
