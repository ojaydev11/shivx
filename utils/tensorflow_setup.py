"""
TensorFlow setup and warning suppression for production environments.
"""
import os
import warnings
import logging

def suppress_tensorflow_warnings():
    """
    Suppress TensorFlow and protobuf warnings for cleaner production output.
    
    This suppresses common warnings that don't affect functionality:
    - TensorFlow C++ logging
    - Protobuf version compatibility warnings  
    - oneDNN optimization messages
    """
    # Suppress TensorFlow logging before import
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent output
    
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.runtime_version')
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    warnings.filterwarnings('ignore', message='.*Protobuf gencode version.*')
    
    # Set TensorFlow logging level
    try:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
    except ImportError:
        pass  # TensorFlow not installed, skip

def setup_production_logging():
    """Set up clean logging for production environment."""
    # Suppress verbose libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)