"""
Run ID Generation: Snowflake-like IDs with UUID v7 fallback.
"""

import time
import uuid
import threading
from datetime import datetime
from typing import Optional

# Thread-safe counter for snowflake IDs
_counter_lock = threading.Lock()
_last_timestamp = 0
_sequence = 0
_NODE_ID = 1  # Could be configurable per instance


def generate_snowflake_id() -> str:
    """Generate a snowflake-like ID (timestamp + node + sequence)."""
    global _last_timestamp, _sequence
    
    with _counter_lock:
        current_time = int(time.time() * 1000)  # milliseconds since epoch
        
        if current_time == _last_timestamp:
            _sequence = (_sequence + 1) & 0xFFF  # 12-bit sequence
            if _sequence == 0:
                # Wait for next millisecond
                time.sleep(0.001)
                current_time = int(time.time() * 1000)
        else:
            _sequence = 0
        
        _last_timestamp = current_time
        
        # Combine: 41 bits timestamp + 10 bits node + 12 bits sequence
        snowflake_id = (current_time << 22) | (_NODE_ID << 12) | _sequence
        
        return str(snowflake_id)


def generate_uuid_v7() -> str:
    """Generate a UUID v7 (timestamp + random)."""
    try:
        # Python 3.13+ has uuid.uuid7()
        if hasattr(uuid, 'uuid7'):
            return str(uuid.uuid7())
    except AttributeError:
        pass
    
    # Fallback implementation for older Python versions
    now = datetime.now()
    timestamp_ms = int(now.timestamp() * 1000)
    
    # Convert timestamp to 48-bit hex
    timestamp_hex = format(timestamp_ms & 0xFFFFFFFFFFFF, '012x')
    
    # Generate random bytes for the rest
    random_bytes = uuid.uuid4().bytes[6:]  # Last 10 bytes
    
    # Combine timestamp + random + version + variant
    uuid_bytes = bytes.fromhex(timestamp_hex[:12]) + random_bytes
    uuid_bytes = uuid_bytes[:16]  # Ensure 16 bytes
    
    # Set version (7) and variant bits
    uuid_bytes = bytearray(uuid_bytes)
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x70  # Version 7
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80  # Variant
    
    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def generate_run_id(use_snowflake: bool = True) -> str:
    """Generate a unique run ID.
    
    Args:
        use_snowflake: If True, use snowflake format; otherwise use UUID v7
        
    Returns:
        Unique run ID string
    """
    try:
        if use_snowflake:
            return generate_snowflake_id()
        else:
            return generate_uuid_v7()
    except Exception:
        # Fallback to UUID v4 if both methods fail
        return str(uuid.uuid4())


def parse_run_id(run_id: str) -> Optional[dict]:
    """Parse a run ID to extract information.
    
    Args:
        run_id: The run ID to parse
        
    Returns:
        Dictionary with parsed information or None if invalid
    """
    try:
        # Try to parse as snowflake ID
        if run_id.isdigit() and len(run_id) <= 19:
            snowflake_int = int(run_id)
            timestamp_ms = (snowflake_int >> 22) & 0x1FFFFFFFFFF
            node_id = (snowflake_int >> 12) & 0x3FF
            sequence = snowflake_int & 0xFFF
            
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            return {
                "type": "snowflake",
                "timestamp": timestamp,
                "timestamp_ms": timestamp_ms,
                "node_id": node_id,
                "sequence": sequence,
                "original": run_id
            }
        
        # Try to parse as UUID
        try:
            uuid_obj = uuid.UUID(run_id)
            timestamp_ms = None
            
            # Extract timestamp from UUID v7 if possible
            if uuid_obj.version == 7:
                # First 48 bits contain timestamp
                timestamp_bytes = uuid_obj.bytes[:6]
                timestamp_ms = int.from_bytes(timestamp_bytes, 'big')
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            else:
                timestamp = None
            
            return {
                "type": "uuid",
                "version": uuid_obj.version,
                "timestamp": timestamp,
                "timestamp_ms": timestamp_ms,
                "original": run_id
            }
            
        except ValueError:
            return None
            
    except Exception:
        return None


def is_valid_run_id(run_id: str) -> bool:
    """Check if a run ID is valid.
    
    Args:
        run_id: The run ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return parse_run_id(run_id) is not None


def get_run_id_info(run_id: str) -> str:
    """Get human-readable information about a run ID.
    
    Args:
        run_id: The run ID to analyze
        
    Returns:
        Formatted string with run ID information
    """
    parsed = parse_run_id(run_id)
    if not parsed:
        return f"Invalid run ID: {run_id}"
    
    if parsed["type"] == "snowflake":
        return (f"Snowflake ID: {run_id}\n"
                f"  Timestamp: {parsed['timestamp']}\n"
                f"  Node ID: {parsed['node_id']}\n"
                f"  Sequence: {parsed['sequence']}")
    
    elif parsed["type"] == "uuid":
        info = f"UUID v{parsed['version']}: {run_id}"
        if parsed["timestamp"]:
            info += f"\n  Timestamp: {parsed['timestamp']}"
        return info
    
    return f"Unknown format: {run_id}"


if __name__ == "__main__":
    # Test the run ID generation
    print("Testing Run ID Generation:")
    print("-" * 40)
    
    # Generate some IDs
    snowflake_id = generate_run_id(use_snowflake=True)
    uuid_id = generate_run_id(use_snowflake=False)
    
    print(f"Snowflake ID: {snowflake_id}")
    print(f"UUID v7 ID: {uuid_id}")
    print()
    
    # Parse them
    print("Parsing Results:")
    print("-" * 40)
    print(get_run_id_info(snowflake_id))
    print()
    print(get_run_id_info(uuid_id))
    print()
    
    # Validation
    print("Validation:")
    print("-" * 40)
    print(f"Snowflake ID valid: {is_valid_run_id(snowflake_id)}")
    print(f"UUID ID valid: {is_valid_run_id(uuid_id)}")
    print(f"Invalid ID valid: {is_valid_run_id('invalid')}")
