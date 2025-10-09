"""
Tamper-Evident Audit Chain for ShivX
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class AuditChain:
    """Tamper-evident audit log with hash chains."""
    
    def __init__(self, log_file: str, head_file: str):
        self.log_file = Path(log_file)
        self.head_file = Path(head_file)
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure log and head files exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_file.exists():
            self.log_file.touch()
        
        if not self.head_file.exists():
            # Initialize with zero hash
            self.head_file.write_text("0" * 64)
    
    def _get_current_head(self) -> str:
        """Get current head hash."""
        try:
            return self.head_file.read_text().strip()
        except Exception:
            return "0" * 64
    
    def _compute_hash(self, data: str, prev_hash: str) -> str:
        """Compute hash of data + previous hash."""
        combined = f"{data}{prev_hash}".encode('utf-8')
        return hashlib.sha256(combined).hexdigest()
    
    def append(self, entry: Dict[str, Any]) -> str:
        """Append entry to audit log and update hash chain."""
        try:
            # Get current head
            prev_hash = self._get_current_head()
            
            # Add previous hash to entry
            entry['prev_hash'] = prev_hash
            
            # Convert to JSON
            entry_json = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            
            # Compute new hash
            new_hash = self._compute_hash(entry_json, prev_hash)
            
            # Append to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(entry_json + '\n')
            
            # Update head file
            self.head_file.write_text(new_hash)
            
            return new_hash
            
        except Exception as e:
            raise RuntimeError(f"Failed to append to audit chain: {e}")
    
    def verify(self) -> Dict[str, Any]:
        """Verify integrity of audit chain."""
        try:
            if not self.log_file.exists() or self.log_file.stat().st_size == 0:
                return {
                    'valid': True,
                    'entries': 0,
                    'head_hash': self._get_current_head(),
                    'errors': []
                }
            
            errors = []
            entries = []
            current_hash = "0" * 64
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                        
                        # Verify hash chain
                        expected_hash = self._compute_hash(line, current_hash)
                        entry_hash = entry.get('prev_hash')
                        
                        if entry_hash != current_hash:
                            errors.append(f"Line {line_num}: Hash chain broken - expected {current_hash}, got {entry_hash}")
                        
                        current_hash = expected_hash
                        
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    except Exception as e:
                        errors.append(f"Line {line_num}: Error processing - {e}")
            
            # Check final hash matches head
            stored_head = self._get_current_head()
            if current_hash != stored_head:
                errors.append(f"Head hash mismatch - computed: {current_hash}, stored: {stored_head}")
            
            return {
                'valid': len(errors) == 0,
                'entries': len(entries),
                'head_hash': stored_head,
                'computed_hash': current_hash,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'valid': False,
                'entries': 0,
                'head_hash': 'unknown',
                'computed_hash': 'unknown',
                'errors': [f"Verification failed: {e}"]
            }
    
    def get_entries(self, limit: Optional[int] = None) -> list:
        """Get audit log entries."""
        try:
            if not self.log_file.exists():
                return []
            
            entries = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            entries.append(entry)
                        except json.JSONDecodeError:
                            continue
            
            # Return last N entries if limit specified
            if limit:
                return entries[-limit:]
            return entries
            
        except Exception as e:
            raise RuntimeError(f"Failed to read audit entries: {e}")
    
    def get_head_hash(self) -> str:
        """Get current head hash."""
        return self._get_current_head()


# Global audit chain instance
_audit_chains: Dict[str, AuditChain] = {}


def get_audit_chain(log_file: str, head_file: Optional[str] = None) -> AuditChain:
    """Get or create audit chain instance."""
    if log_file not in _audit_chains:
        if head_file is None:
            head_file = str(Path(log_file).parent / "audit_head")
        _audit_chains[log_file] = AuditChain(log_file, head_file)
    
    return _audit_chains[log_file]


def append_jsonl(log_file: str, entry: Dict[str, Any]) -> str:
    """Append entry to audit chain and return new hash."""
    audit_chain = get_audit_chain(log_file)
    return audit_chain.append(entry)


def verify_audit_chain(log_file: str) -> Dict[str, Any]:
    """Verify integrity of audit chain."""
    audit_chain = get_audit_chain(log_file)
    return audit_chain.verify()


def get_audit_entries(log_file: str, limit: Optional[int] = None) -> list:
    """Get audit log entries."""
    audit_chain = get_audit_chain(log_file)
    return audit_chain.get_entries(limit)


def get_audit_head(log_file: str) -> str:
    """Get current audit chain head hash."""
    audit_chain = get_audit_chain(log_file)
    return audit_chain.get_head_hash()
