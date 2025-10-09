"""
Bootstrap Environment Utilities for ShivX
Handles environment variable setup for noise suppression and UTF-8 encoding.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def apply_noise_guards(enabled: bool) -> None:
    """
    Apply noise suppression guards for TensorFlow and protobuf.
    
    Args:
        enabled: Whether to enable noise suppression features
        
    Note:
        - PYTHONIOENCODING: Forces UTF-8 encoding for all I/O operations
        - TF_CPP_MIN_LOG_LEVEL: Reduces TensorFlow C++ logging noise
        - PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION: Uses pure Python protobuf
    """
    if not enabled:
        logger.debug("Noise suppression guards disabled")
        return
    
    # Set UTF-8 encoding for all Python I/O operations
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    
    # Reduce TensorFlow C++ logging noise (0=all, 1=no INFO, 2=no WARNING, 3=no ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # Use pure Python protobuf implementation (can be slower but more compatible)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    
    logger.info("Noise suppression guards applied: UTF-8 + reduced TF/protobuf noise")


def get_environment_status() -> dict:
    """
    Get current status of noise suppression environment variables.
    
    Returns:
        Dictionary with current environment variable values
    """
    return {
        "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING", "not set"),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "not set"),
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "not set")
    }


def validate_environment() -> bool:
    """
    Validate that noise suppression environment variables are properly set.
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = {
        "PYTHONIOENCODING": "utf-8",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"
    }
    
    for var, expected_value in required_vars.items():
        if os.environ.get(var) != expected_value:
            logger.warning(f"Environment variable {var} not set to expected value {expected_value}")
            return False
    
    logger.info("All noise suppression environment variables properly set")
    return True
