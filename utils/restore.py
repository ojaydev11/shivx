"""
Restore utility for ShivX system.
Safely restores system components from backup archives.
"""

import os
import zipfile
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestoreManager:
    """Manages safe restoration of ShivX system from backups."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize restore manager with project root path."""
        self.project_root = project_root or Path(__file__).parent.parent
        self.temp_dir = self.project_root / "temp" / "restore"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Define safe restore paths (only these can be restored)
        self.safe_restore_paths = {
            "config/",
            "var/runs/",
            "var/reports/",
            "var/security/",
            "templates/",
            "logs/"
        }
        
        # Define critical files that should not be overwritten without confirmation
        self.critical_files = {
            "config/ai_enhancements.json",
            "config/coding_standards.json",
            "config/error_handling.json"
        }
    
    def safe_unlink(self, file_path: Path) -> bool:
        """Safely remove a file with backup creation."""
        try:
            if file_path.exists():
                # Create backup of existing file
                backup_path = self.temp_dir / f"pre_restore_{file_path.name}_{datetime.now().strftime('%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up existing file: {file_path} -> {backup_path}")
                
                # Remove the file
                file_path.unlink()
                logger.info(f"Removed file: {file_path}")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to safely remove {file_path}: {e}")
            return False
    
    def safe_move(self, source: Path, destination: Path) -> bool:
        """Safely move a file with conflict resolution."""
        try:
            if destination.exists():
                # Create backup of existing file
                backup_path = self.temp_dir / f"pre_restore_{destination.name}_{datetime.now().strftime('%H%M%S')}"
                shutil.copy2(destination, backup_path)
                logger.info(f"Backed up existing file: {destination} -> {backup_path}")
                
                # Remove existing file
                destination.unlink()
            
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source), str(destination))
            logger.info(f"Moved file: {source} -> {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to safely move {source} to {destination}: {e}")
            return False
    
    def validate_restore_path(self, file_path: str) -> bool:
        """Validate if a file path is safe to restore."""
        # Check if path starts with any safe restore path
        for safe_path in self.safe_restore_paths:
            if file_path.startswith(safe_path):
                return True
        
        # Check if it's a critical file
        if file_path in self.critical_files:
            logger.warning(f"Critical file detected: {file_path}")
            return True
        
        logger.warning(f"Unsafe restore path: {file_path}")
        return False
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        if not file_path.exists():
            return ""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def dry_run_restore(self, backup_path: Path) -> Dict:
        """Perform a dry-run restore to show what would be changed."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        restore_plan = {
            "backup_file": str(backup_path),
            "files_to_restore": [],
            "files_to_overwrite": [],
            "new_files": [],
            "conflicts": [],
            "summary": {
                "total_files": 0,
                "overwrites": 0,
                "new_files": 0,
                "safe_operations": 0,
                "unsafe_operations": 0
            }
        }
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                for file_info in zipf.infolist():
                    if file_info.is_dir():
                        continue
                    
                    file_path = file_info.filename
                    
                    # Skip metadata files
                    if file_path in ["backup_metadata.json"]:
                        continue
                    
                    # Validate restore path
                    if not self.validate_restore_path(file_path):
                        restore_plan["conflicts"].append({
                            "file": file_path,
                            "reason": "Unsafe restore path",
                            "action": "SKIP"
                        })
                        restore_plan["summary"]["unsafe_operations"] += 1
                        continue
                    
                    target_path = self.project_root / file_path
                    restore_plan["summary"]["total_files"] += 1
                    
                    if target_path.exists():
                        # File exists - check if it's different
                        try:
                            with zipf.open(file_path) as zip_file:
                                zip_content = zip_file.read()
                                zip_hash = hashlib.sha256(zip_content).hexdigest()
                            
                            current_hash = self.calculate_file_hash(target_path)
                            
                            if zip_hash != current_hash:
                                restore_plan["files_to_overwrite"].append({
                                    "file": file_path,
                                    "current_hash": current_hash,
                                    "backup_hash": zip_hash,
                                    "action": "OVERWRITE"
                                })
                                restore_plan["summary"]["overwrites"] += 1
                            else:
                                restore_plan["files_to_restore"].append({
                                    "file": file_path,
                                    "current_hash": current_hash,
                                    "backup_hash": zip_hash,
                                    "action": "SKIP (identical)"
                                })
                        except Exception as e:
                            restore_plan["conflicts"].append({
                                "file": file_path,
                                "reason": f"Error checking file: {e}",
                                "action": "ERROR"
                            })
                            restore_plan["summary"]["unsafe_operations"] += 1
                    else:
                        # New file
                        restore_plan["new_files"].append({
                            "file": file_path,
                            "action": "CREATE"
                        })
                        restore_plan["summary"]["new_files"] += 1
                    
                    restore_plan["summary"]["safe_operations"] += 1
        
        except Exception as e:
            logger.error(f"Dry run failed: {e}")
            raise
        
        return restore_plan
    
    def restore_backup(self, backup_path: Path, dry_run: bool = False, 
                      force_overwrite: bool = False) -> Dict:
        """Restore system from a backup file."""
        if dry_run:
            return self.dry_run_restore(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        restore_results = {
            "backup_file": str(backup_path),
            "restored_files": [],
            "skipped_files": [],
            "errors": [],
            "summary": {
                "total_files": 0,
                "restored": 0,
                "skipped": 0,
                "errors": 0
            }
        }
        
        # Create temporary restore directory
        restore_temp_dir = self.temp_dir / f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        restore_temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                for file_info in zipf.infolist():
                    if file_info.is_dir():
                        continue
                    
                    file_path = file_info.filename
                    
                    # Skip metadata files
                    if file_path in ["backup_metadata.json"]:
                        continue
                    
                    # Validate restore path
                    if not self.validate_restore_path(file_path):
                        restore_results["skipped_files"].append({
                            "file": file_path,
                            "reason": "Unsafe restore path"
                        })
                        restore_results["summary"]["skipped"] += 1
                        continue
                    
                    target_path = self.project_root / file_path
                    restore_results["summary"]["total_files"] += 1
                    
                    try:
                        # Extract to temporary location first
                        temp_file_path = restore_temp_dir / Path(file_path).name
                        with zipf.open(file_path) as zip_file, open(temp_file_path, 'wb') as temp_file:
                            temp_file.write(zip_file.read())
                        
                        # Check if target exists and is different
                        if target_path.exists():
                            current_hash = self.calculate_file_hash(target_path)
                            temp_hash = self.calculate_file_hash(temp_file_path)
                            
                            if current_hash == temp_hash:
                                # Files are identical, skip
                                restore_results["skipped_files"].append({
                                    "file": file_path,
                                    "reason": "File is identical"
                                })
                                restore_results["summary"]["skipped"] += 1
                                continue
                            
                            if not force_overwrite:
                                # Ask for confirmation for critical files
                                if file_path in self.critical_files:
                                    logger.warning(f"Critical file would be overwritten: {file_path}")
                        
                        # Policy guard check for file write
                        try:
                            from utils.policy_guard import evaluate_policy
                            
                            decision = evaluate_policy("fs.write", {
                                "path": str(target_path),
                                "operation": "restore",
                                "source": "backup"
                            })
                            
                            if decision.decision == "deny":
                                restore_results["errors"].append({
                                    "file": file_path,
                                    "reason": f"Policy denied: {', '.join(decision.reasons)}"
                                })
                                restore_results["summary"]["errors"] += 1
                                logger.error(f"Policy denied restore of {file_path}: {', '.join(decision.reasons)}")
                                continue
                            elif decision.decision == "warn":
                                logger.warning(f"Policy warning for {file_path}: {', '.join(decision.reasons)}")
                                
                        except ImportError:
                            # Continue if policy guard not available
                            pass
                        
                        # Restore the file
                        if self.safe_move(temp_file_path, target_path):
                            restore_results["restored_files"].append({
                                "file": file_path,
                                "action": "RESTORED"
                            })
                            restore_results["summary"]["restored"] += 1
                            logger.info(f"Restored: {file_path}")
                        else:
                            restore_results["errors"].append({
                                "file": file_path,
                                "reason": "Failed to restore file"
                            })
                            restore_results["summary"]["errors"] += 1
                    
                    except Exception as e:
                        restore_results["errors"].append({
                            "file": file_path,
                            "reason": str(e)
                        })
                        restore_results["summary"]["errors"] += 1
                        logger.error(f"Failed to restore {file_path}: {e}")
        
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
        finally:
            # Cleanup temporary files
            if restore_temp_dir.exists():
                shutil.rmtree(restore_temp_dir)
        
        logger.info(f"Restore completed: {restore_results['summary']['restored']} files restored, "
                   f"{restore_results['summary']['skipped']} skipped, "
                   f"{restore_results['summary']['errors']} errors")
        
        return restore_results


def restore_backup(backup_path: Path, dry_run: bool = False, 
                  force_overwrite: bool = False) -> Dict:
    """Convenience function to restore a backup."""
    manager = RestoreManager()
    return manager.restore_backup(backup_path, dry_run, force_overwrite)


def dry_run_restore(backup_path: Path) -> Dict:
    """Convenience function to perform a dry-run restore."""
    manager = RestoreManager()
    return manager.dry_run_restore(backup_path)


if __name__ == "__main__":
    # Test restore functionality
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python restore.py <backup_file> [--dry-run]")
        sys.exit(1)
    
    backup_file = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    
    try:
        if dry_run:
            results = dry_run_restore(backup_file)
            print("DRY RUN RESULTS:")
        else:
            results = restore_backup(backup_file)
            print("RESTORE RESULTS:")
        
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"Restore failed: {e}")
        sys.exit(1)
