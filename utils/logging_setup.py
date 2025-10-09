"""
JSON Logging Setup for ShivX with Request Tracing
"""

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Context variables for request tracing
trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
span_id: ContextVar[Optional[str]] = ContextVar('span_id', default=None)
request_path: ContextVar[Optional[str]] = ContextVar('request_path', default=None)
request_method: ContextVar[Optional[str]] = ContextVar('request_method', default=None)
request_status: ContextVar[Optional[int]] = ContextVar('request_status', default=None)
request_start_time: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter that includes trace context."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Get trace context from contextvars
        trace_context = {
            'ts': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
            'trace_id': trace_id.get(),
            'span_id': span_id.get(),
            'path': request_path.get(),
            'method': request_method.get(),
            'status': request_status.get(),
        }
        
        # Add duration if we have start time
        if request_start_time.get():
            duration_ms = int((time.time() - request_start_time.get()) * 1000)
            trace_context['dur_ms'] = duration_ms
        
        # Add exception info if present
        if record.exc_info:
            trace_context['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                trace_context[key] = value
        
        return json.dumps(trace_context, ensure_ascii=False)


def init_logging(config: Dict[str, Any]) -> None:
    """Initialize JSON logging with the given configuration."""
    if not config.get('logging', {}).get('enabled', False):
        return
    
    log_config = config['logging']
    
    # Create logs directory if it doesn't exist
    log_file = Path(log_config['file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.get('rotate_megabytes', 20) * 1024 * 1024,
        backupCount=log_config.get('rotate_backups', 7),
        encoding='utf-8'
    )
    
    # Set formatter
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Add console handler for development
    if os.getenv('SHIVX_DEV', 'false').lower() == 'true':
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.WARNING)
    
    logging.info("JSON logging initialized", extra={
        'log_file': str(log_file),
        'rotate_mb': log_config.get('rotate_megabytes', 20),
        'backups': log_config.get('rotate_backups', 7)
    })


def set_trace_context(
    trace_id_val: Optional[str] = None,
    span_id_val: Optional[str] = None,
    path: Optional[str] = None,
    method: Optional[str] = None,
    status: Optional[int] = None,
    start_time: Optional[float] = None
) -> None:
    """Set trace context for the current request."""
    if trace_id_val:
        trace_id.set(trace_id_val)
    if span_id_val:
        span_id.set(span_id_val)
    if path:
        request_path.set(path)
    if method:
        request_method.set(method)
    if status is not None:
        request_status.set(status)
    if start_time:
        request_start_time.set(start_time)


def clear_trace_context() -> None:
    """Clear trace context for the current request."""
    trace_id.set(None)
    span_id.set(None)
    request_path.set(None)
    request_method.set(None)
    request_status.set(None)
    request_start_time.set(None)


def generate_trace_id() -> str:
    """Generate a new trace ID."""
    return str(uuid.uuid4())


def generate_span_id() -> str:
    """Generate a new span ID."""
    return str(uuid.uuid4())[:16]
