"""
Artifacts Manager: Save, organize, and manage run artifacts with retention policies.
"""

import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict

from utils.run_id import generate_run_id, is_valid_run_id

logger = logging.getLogger("artifacts")

# Default retention settings
DEFAULT_MAX_DAYS = 14
DEFAULT_MAX_BYTES = 1024 * 1024 * 1024  # 1GB


@dataclass
class ArtifactRecord:
    """Record of an artifact with metadata."""
    id: str
    run_id: str
    kind: str
    path: str
    size_bytes: int
    created_at: datetime
    metadata: Dict[str, Any]
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArtifactRecord':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ArtifactsManager:
    """Manages run artifacts with retention policies."""
    
    def __init__(self, base_path: str = "var/runs", config: Optional[Dict[str, Any]] = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = config or {}
        self.max_days = self.config.get('artifacts', {}).get('max_days', DEFAULT_MAX_DAYS)
        self.max_bytes = self.config.get('artifacts', {}).get('max_bytes', DEFAULT_MAX_BYTES)
        
        # Create necessary directories
        self._ensure_directories()
        
        # Load existing artifact index
        self.artifacts_index = self._load_artifacts_index()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        (self.base_path / "latest").mkdir(exist_ok=True)
        (self.base_path / "artifacts").mkdir(exist_ok=True)
        (self.base_path / "index").mkdir(exist_ok=True)
    
    def _load_artifacts_index(self) -> Dict[str, ArtifactRecord]:
        """Load artifacts index from disk."""
        index_file = self.base_path / "index" / "artifacts.json"
        if not index_file.exists():
            return {}
        
        try:
            with open(index_file, 'r') as f:
                data = json.load(f)
            
            artifacts = {}
            for artifact_id, artifact_data in data.items():
                try:
                    artifacts[artifact_id] = ArtifactRecord.from_dict(artifact_data)
                except Exception as e:
                    logger.warning(f"Failed to load artifact {artifact_id}: {e}")
            
            return artifacts
        except Exception as e:
            logger.error(f"Failed to load artifacts index: {e}")
            return {}
    
    def _save_artifacts_index(self):
        """Save artifacts index to disk."""
        index_file = self.base_path / "index" / "artifacts.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(
                    {aid: artifact.to_dict() for aid, artifact in self.artifacts_index.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save artifacts index: {e}")
    
    def save_artifact(self, content: Union[str, bytes, Path], kind: str, 
                     metadata: Optional[Dict[str, Any]] = None, 
                     run_id: Optional[str] = None, tags: Optional[List[str]] = None) -> ArtifactRecord:
        """Save an artifact and return its record.
        
        Args:
            content: Content to save (string, bytes, or file path)
            kind: Type of artifact (e.g., 'blackboard', 'audit', 'output', 'log')
            metadata: Additional metadata about the artifact
            run_id: Associated run ID (generated if not provided)
            tags: List of tags for categorization
            
        Returns:
            ArtifactRecord with artifact information
        """
        # Generate run ID if not provided
        if not run_id:
            run_id = generate_run_id()
        
        # Validate run ID
        if not is_valid_run_id(run_id):
            raise ValueError(f"Invalid run ID: {run_id}")
        
        # Create run directory
        run_dir = self.base_path / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Generate artifact ID
        artifact_id = f"{run_id}_{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine file path and extension
        if isinstance(content, str):
            file_path = run_dir / f"{kind}.json"
            content_bytes = content.encode('utf-8')
        elif isinstance(content, bytes):
            file_path = run_dir / f"{kind}.bin"
            content_bytes = content
        elif isinstance(content, Path):
            if content.is_file():
                file_path = run_dir / f"{kind}_{content.name}"
                content_bytes = content.read_bytes()
            else:
                raise ValueError(f"Path does not exist: {content}")
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
        
        # Save content
        try:
            if isinstance(content, str) and kind in ['blackboard', 'audit', 'config']:
                # Pretty-print JSON for human-readable files
                with open(file_path, 'w') as f:
                    json.dump(json.loads(content), f, indent=2)
            else:
                # Save as binary
                with open(file_path, 'wb') as f:
                    f.write(content_bytes)
        except Exception as e:
            logger.error(f"Failed to save artifact {artifact_id}: {e}")
            raise
        
        # Create artifact record
        artifact = ArtifactRecord(
            id=artifact_id,
            run_id=run_id,
            kind=kind,
            path=str(file_path),
            size_bytes=len(content_bytes),
            created_at=datetime.now(),
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Add to index
        self.artifacts_index[artifact_id] = artifact
        self._save_artifacts_index()
        
        # Create symlink to latest
        latest_link = self.base_path / "latest" / kind
        try:
            if latest_link.exists():
                from core.security_sandbox import safe_unlink
                safe_unlink(latest_link, [str(self.base_path)])
            latest_link.symlink_to(file_path.relative_to(self.base_path))
        except Exception as e:
            logger.warning(f"Failed to create latest symlink for {kind}: {e}")
        
        logger.info(f"Saved artifact {artifact_id} ({kind}) for run {run_id}")
        return artifact
    
    def get_artifact(self, artifact_id: str) -> Optional[ArtifactRecord]:
        """Get artifact record by ID."""
        return self.artifacts_index.get(artifact_id)
    
    def get_run_artifacts(self, run_id: str) -> List[ArtifactRecord]:
        """Get all artifacts for a specific run."""
        return [
            artifact for artifact in self.artifacts_index.values()
            if artifact.run_id == run_id
        ]
    
    def get_artifacts_by_kind(self, kind: str) -> List[ArtifactRecord]:
        """Get all artifacts of a specific kind."""
        return [
            artifact for artifact in self.artifacts_index.values()
            if artifact.kind == kind
        ]
    
    def get_artifacts_by_tag(self, tag: str) -> List[ArtifactRecord]:
        """Get all artifacts with a specific tag."""
        return [
            artifact for artifact in self.artifacts_index.values()
            if tag in artifact.tags
        ]
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact and its file."""
        if artifact_id not in self.artifacts_index:
            return False
        
        artifact = self.artifacts_index[artifact_id]
        
        try:
            # Delete file
            file_path = Path(artifact.path)
            if file_path.exists():
                from core.security_sandbox import safe_unlink
                safe_unlink(file_path, [str(self.base_path)])
            
            # Remove from index
            del self.artifacts_index[artifact_id]
            self._save_artifacts_index()
            
            logger.info(f"Deleted artifact {artifact_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False
    
    def cleanup_expired_artifacts(self) -> Dict[str, int]:
        """Clean up artifacts based on retention policies."""
        now = datetime.now()
        cutoff_date = now - timedelta(days=self.max_days)
        
        deleted_count = 0
        freed_bytes = 0
        
        # Find expired artifacts
        expired_artifacts = [
            artifact for artifact in self.artifacts_index.values()
            if artifact.created_at < cutoff_date
        ]
        
        # Delete expired artifacts
        for artifact in expired_artifacts:
            if self.delete_artifact(artifact.id):
                deleted_count += 1
                freed_bytes += artifact.size_bytes
        
        # Check size-based cleanup if still over limit
        total_size = sum(artifact.size_bytes for artifact in self.artifacts_index.values())
        if total_size > self.max_bytes:
            # Sort by creation date (oldest first) and delete until under limit
            sorted_artifacts = sorted(
                self.artifacts_index.values(),
                key=lambda x: x.created_at
            )
            
            for artifact in sorted_artifacts:
                if total_size <= self.max_bytes:
                    break
                
                if self.delete_artifact(artifact.id):
                    deleted_count += 1
                    freed_bytes += artifact.size_bytes
                    total_size -= artifact.size_bytes
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} artifacts, freed {freed_bytes} bytes")
        
        return {
            "deleted_count": deleted_count,
            "freed_bytes": freed_bytes,
            "total_artifacts": len(self.artifacts_index)
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(artifact.size_bytes for artifact in self.artifacts_index.values())
        total_count = len(self.artifacts_index)
        
        # Group by kind
        by_kind = {}
        for artifact in self.artifacts_index.values():
            if artifact.kind not in by_kind:
                by_kind[artifact.kind] = {"count": 0, "size_bytes": 0}
            by_kind[artifact.kind]["count"] += 1
            by_kind[artifact.kind]["size_bytes"] += artifact.size_bytes
        
        # Group by run
        by_run = {}
        for artifact in self.artifacts_index.values():
            if artifact.run_id not in by_run:
                by_run[artifact.run_id] = {"count": 0, "size_bytes": 0}
            by_run[artifact.run_id]["count"] += 1
            by_run[artifact.run_id]["size_bytes"] += artifact.size_bytes
        
        return {
            "total_count": total_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "by_kind": by_kind,
            "by_run": by_run,
            "retention_policy": {
                "max_days": self.max_days,
                "max_bytes": self.max_bytes,
                "max_bytes_mb": self.max_bytes / (1024 * 1024)
            }
        }
    
    def export_run(self, run_id: str, export_path: Optional[str] = None) -> str:
        """Export all artifacts for a run to a zip file."""
        import zipfile
        
        if not export_path:
            export_path = f"var/exports/{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        run_artifacts = self.get_run_artifacts(run_id)
        if not run_artifacts:
            raise ValueError(f"No artifacts found for run {run_id}")
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for artifact in run_artifacts:
                file_path = Path(artifact.path)
                if file_path.exists():
                    zipf.write(file_path, f"{run_id}/{file_path.name}")
            
            # Add metadata
            metadata = {
                "run_id": run_id,
                "exported_at": datetime.now().isoformat(),
                "artifacts": [artifact.to_dict() for artifact in run_artifacts]
            }
            
            zipf.writestr(f"{run_id}/metadata.json", json.dumps(metadata, indent=2))
        
        logger.info(f"Exported run {run_id} to {export_path}")
        return str(export_path)


# Global artifacts manager instance
artifacts_manager = ArtifactsManager()


def save_artifact(content: Union[str, bytes, Path], kind: str, 
                 metadata: Optional[Dict[str, Any]] = None, 
                 run_id: Optional[str] = None, tags: Optional[List[str]] = None) -> ArtifactRecord:
    """Convenience function to save an artifact."""
    return artifacts_manager.save_artifact(content, kind, metadata, run_id, tags)


def get_run_artifacts(run_id: str) -> List[ArtifactRecord]:
    """Convenience function to get run artifacts."""
    return artifacts_manager.get_run_artifacts(run_id)


if __name__ == "__main__":
    # Test the artifacts manager
    print("Testing Artifacts Manager:")
    print("-" * 40)
    
    # Create test artifacts
    test_run_id = generate_run_id()
    
    # Save some test artifacts
    blackboard_data = {"goal": "test", "status": "running"}
    audit_data = [{"timestamp": datetime.now().isoformat(), "action": "test"}]
    
    try:
        # Save blackboard
        blackboard_artifact = save_artifact(
            json.dumps(blackboard_data, indent=2),
            "blackboard",
            {"test": True},
            test_run_id,
            ["test", "blackboard"]
        )
        print(f"Saved blackboard artifact: {blackboard_artifact.id}")
        
        # Save audit
        audit_artifact = save_artifact(
            json.dumps(audit_data, indent=2),
            "audit",
            {"test": True},
            test_run_id,
            ["test", "audit"]
        )
        print(f"Saved audit artifact: {audit_artifact.id}")
        
        # Get run artifacts
        run_artifacts = get_run_artifacts(test_run_id)
        print(f"Run {test_run_id} has {len(run_artifacts)} artifacts")
        
        # Get storage stats
        stats = artifacts_manager.get_storage_stats()
        print(f"Storage stats: {stats['total_count']} artifacts, {stats['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Error during testing: {e}")
