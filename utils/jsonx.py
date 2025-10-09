"""
JSON Extended Utilities
Compatibility stub for legacy test imports.
Wraps standard json module with same interface.
"""

import json as _json
from typing import Any

# Re-export standard json functions
dumps = _json.dumps
loads = _json.loads
dump = _json.dump
load = _json.load

__all__ = ["dumps", "loads", "dump", "load"]


def dumps_pretty(obj: Any, **kwargs) -> str:
    """Pretty-print JSON with indentation."""
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("sort_keys", False)
    return _json.dumps(obj, **kwargs)


def loads_safe(s: str, default: Any = None) -> Any:
    """Load JSON with fallback to default on error."""
    try:
        return _json.loads(s)
    except (ValueError, TypeError):
        return default
