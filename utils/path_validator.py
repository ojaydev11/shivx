"""
Path validation utility for secure file operations
"""
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any

# Import the validated safe deletion function
from core.security_sandbox import safe_unlink as core_safe_unlink

def load_path_policy() -> Dict[str, Any]:
    """Load path policy from settings.yaml"""
    try:
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('policy', {})
    except Exception:
        pass
    
    # Fallback policy
    return {
        'allow_paths': ["E:/shivx/data", "E:/shivx/var", "E:/shivx/temp"],
        'blocked_paths': ["C:/Windows", "C:/System32", "/system", "/root"]
    }

def validate_deletion_path(file_path: str) -> bool:
    """
    Validate if a path is safe for deletion based on policy
    Returns True if safe to delete, False otherwise
    """
    try:
        policy = load_path_policy()
        allow_paths = policy.get('allow_paths', [])
        blocked_paths = policy.get('blocked_paths', [])
        
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        
        # Check blocked paths first
        for blocked in blocked_paths:
            if abs_path.lower().startswith(blocked.lower()):
                return False
        
        # Check allowed paths
        for allowed in allow_paths:
            if abs_path.lower().startswith(allowed.lower()):
                return True
        
        # Default deny if not in allowed paths
        return False
        
    except Exception:
        # Default deny on error
        return False

def safe_unlink(file_path: str, missing_ok: bool = True) -> Dict[str, Any]:
    """
    Safely delete a file with path validation
    """
    try:
        path_obj = Path(file_path)
        
        # Validate path is safe for deletion
        if not validate_deletion_path(str(path_obj)):
            return {
                'success': False,
                'error': f'Path not allowed for deletion: {file_path}',
                'security_blocked': True
            }
        
        # Use the validated safe deletion function
        policy = load_path_policy()
        allow_paths = policy.get('allow_paths', [])
        core_safe_unlink(path_obj, allow_paths)
        
        return {
            'success': True,
            'path': str(path_obj),
            'validated': True
        }
        
    except FileNotFoundError as e:
        if missing_ok:
            return {'success': True, 'path': file_path, 'already_missing': True}
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': str(e)}