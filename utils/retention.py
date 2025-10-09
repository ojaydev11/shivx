"""
Retention management for ShivX run artifacts.
Handles cleanup of old runs based on age and size limits.
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ArtifactRetention:
    """Manages retention of run artifacts"""
    
    def __init__(self, runs_dir: str = "var/runs", config: Optional[Dict] = None):
        self.runs_dir = Path(runs_dir)
        self.config = config or {
            "artifacts": {
                "max_days": 14,
                "max_bytes": 1_000_000_000  # 1GB
            }
        }
        self.index_file = self.runs_dir / "index.json"
        
        # Ensure runs directory exists
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
    def get_run_info(self, run_id: str) -> Optional[Dict]:
        """Get information about a specific run"""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None
            
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read summary for run {run_id}: {e}")
                
        # Fallback: get basic info from directory
        try:
            stat = run_dir.stat()
            return {
                "run_id": run_id,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size_bytes": self._get_dir_size(run_dir)
            }
        except Exception as e:
            logger.warning(f"Failed to get basic info for run {run_id}: {e}")
            return None
    
    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Failed to calculate size for {path}: {e}")
        return total_size
    
    def update_index(self, run_id: str, run_info: Dict):
        """Update the runs index with new run information"""
        try:
            # Read existing index
            index_data = []
            if self.index_file.exists():
                try:
                    with open(self.index_file, 'r') as f:
                        index_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read index file: {e}")
            
            # Add/update run info
            run_entry = {
                "run_id": run_id,
                "created": run_info.get("created", datetime.now().isoformat()),
                "size_bytes": run_info.get("size_bytes", 0),
                "status": run_info.get("status", "unknown")
            }
            
            # Remove existing entry if present
            index_data = [entry for entry in index_data if entry["run_id"] != run_id]
            
            # Add new entry
            index_data.append(run_entry)
            
            # Sort by creation time (newest first)
            index_data.sort(key=lambda x: x.get("created", ""), reverse=True)
            
            # Write index atomically
            temp_file = self.index_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            temp_file.replace(self.index_file)
            
        except Exception as e:
            logger.error(f"Failed to update index for run {run_id}: {e}")
    
    def get_runs_to_cleanup(self) -> List[Dict]:
        """Get list of runs that should be cleaned up"""
        if not self.index_file.exists():
            return []
            
        try:
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read index file: {e}")
            return []
        
        now = datetime.now()
        max_age = timedelta(days=self.config["artifacts"]["max_days"])
        max_bytes = self.config["artifacts"]["max_bytes"]
        
        runs_to_cleanup = []
        total_size = 0
        
        for entry in index_data:
            try:
                created = datetime.fromisoformat(entry.get("created", ""))
                age = now - created
                size = entry.get("size_bytes", 0)
                
                # Check age limit
                if age > max_age:
                    runs_to_cleanup.append(entry)
                    continue
                
                # Check size limit
                total_size += size
                if total_size > max_bytes:
                    runs_to_cleanup.append(entry)
                    
            except Exception as e:
                logger.warning(f"Failed to process index entry: {e}")
                continue
        
        return runs_to_cleanup
    
    def cleanup_run(self, run_id: str) -> int:
        """Clean up a specific run and return bytes freed"""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return 0
            
        try:
            # Calculate size before deletion
            size_freed = self._get_dir_size(run_dir)
            
            # Remove directory
            shutil.rmtree(run_dir)
            
            # Update index
            self._remove_from_index(run_id)
            
            logger.info(f"Cleaned up run {run_id}, freed {size_freed} bytes")
            return size_freed
            
        except Exception as e:
            logger.error(f"Failed to cleanup run {run_id}: {e}")
            return 0
    
    def _remove_from_index(self, run_id: str):
        """Remove a run from the index"""
        if not self.index_file.exists():
            return
            
        try:
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
            
            # Remove entry
            index_data = [entry for entry in index_data if entry["run_id"] != run_id]
            
            # Write updated index
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to remove run {run_id} from index: {e}")
    
    def prune_artifacts(self) -> Dict[str, int]:
        """Prune old artifacts and return cleanup summary"""
        runs_to_cleanup = self.get_runs_to_cleanup()
        
        total_bytes_freed = 0
        runs_cleaned = 0
        
        for run_entry in runs_to_cleanup:
            run_id = run_entry["run_id"]
            bytes_freed = self.cleanup_run(run_id)
            total_bytes_freed += bytes_freed
            runs_cleaned += 1
        
        result = {
            "runs_cleaned": runs_cleaned,
            "bytes_freed": total_bytes_freed,
            "runs_remaining": len(self.get_runs_to_cleanup())
        }
        
        logger.info(f"Pruned {runs_cleaned} runs, freed {total_bytes_freed} bytes")
        return result
    
    def get_retention_stats(self) -> Dict:
        """Get current retention statistics"""
        if not self.index_file.exists():
            return {
                "total_runs": 0,
                "total_size_bytes": 0,
                "oldest_run": None,
                "newest_run": None
            }
        
        try:
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
            
            if not index_data:
                return {
                    "total_runs": 0,
                    "total_size_bytes": 0,
                    "oldest_run": None,
                    "newest_run": None
                }
            
            total_size = sum(entry.get("size_bytes", 0) for entry in index_data)
            created_times = [entry.get("created", "") for entry in index_data if entry.get("created")]
            
            return {
                "total_runs": len(index_data),
                "total_size_bytes": total_size,
                "oldest_run": min(created_times) if created_times else None,
                "newest_run": max(created_times) if created_times else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get retention stats: {e}")
            return {
                "total_runs": 0,
                "total_size_bytes": 0,
                "oldest_run": None,
                "newest_run": None
            }
