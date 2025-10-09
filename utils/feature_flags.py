"""
Lightweight feature-flag loader with mtime caching and monotonic TTL.
Flags live in config/settings.yaml under the 'features:' key.
"""
from __future__ import annotations
import threading
import yaml
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional
from utils.metrics import inc_feature_usage

_SETTINGS = Path("config/settings.yaml")
_CACHE_LOCK = threading.RLock()  # More robust for nested calls
_CACHE = {"features": {}, "mtime": 0.0, "loaded_at": 0.0}
_CACHE_TTL = 10.0  # seconds

def _maybe_reload() -> None:
    global _CACHE
    try:
        mtime = _SETTINGS.stat().st_mtime
    except FileNotFoundError:
        with _CACHE_LOCK:
            _CACHE.update(features={}, mtime=0.0, loaded_at=0.0)
        return
    
    with _CACHE_LOCK:
        now = monotonic()
        if (mtime != _CACHE["mtime"]) or (now - _CACHE["loaded_at"] > _CACHE_TTL):
            data = yaml.safe_load(_SETTINGS.read_text(encoding="utf-8")) or {}
            feats = data.get("features", {}) or {}
            
            # Apply environment variable overrides (ops escape hatch)
            import os
            overrides = {k[len("SHIVX_FEATURE_"):].lower(): os.getenv(k) 
                        for k in os.environ if k.startswith("SHIVX_FEATURE_")}
            for k, v in overrides.items():
                if v is not None:
                    feats[k] = {"enabled": v.lower() in ("1", "true", "yes", "on")}
            
            _CACHE.update(features=feats, mtime=mtime, loaded_at=now)
            # Emit one metric per feature snapshot (cheap and useful)
            for name, cfg in feats.items():
                enabled = cfg.get("enabled") if isinstance(cfg, dict) else bool(cfg)
                inc_feature_usage(str(name), bool(enabled))

def all_features() -> Dict[str, Any]:
    # Monotonic TTL so we don't stat/read on every call
    # (kept tiny; adjust if needed)
    now = monotonic()
    if _CACHE["mtime"] == 0.0 or (now - _CACHE["loaded_at"] > _CACHE_TTL):
        _maybe_reload()
    return dict(_CACHE["features"])

def is_feature_enabled(name: str, default: bool = False) -> bool:
    feats = all_features()
    val = feats.get(name, default)
    if isinstance(val, dict):
        return bool(val.get("enabled", default))
    return bool(val)
