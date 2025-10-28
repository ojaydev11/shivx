"""
Week 11: Security Hardening Module

Implements security best practices:
- Input validation and sanitization
- Authentication and authorization
- Encryption for sensitive data
- API key management
- Security auditing
- Protection against common attacks
- Secrets management
"""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from cryptography.fernet import Fernet
import functools

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permission types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class User:
    """User entity"""
    user_id: str
    username: str
    password_hash: str
    permissions: Set[Permission] = field(default_factory=set)
    roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True


@dataclass
class APIKey:
    """API key for external access"""
    key_id: str
    key_hash: str  # Hashed key
    name: str
    permissions: Set[Permission]
    rate_limit: int  # Calls per minute
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    is_active: bool = True


@dataclass
class SecurityAuditEntry:
    """Security audit log entry"""
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    success: bool
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """
    Input validation and sanitization.

    Protects against injection attacks and malformed input.
    """

    # Regex patterns for common inputs
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,32}$')
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )

    # Dangerous patterns (potential injection)
    SQL_INJECTION_PATTERNS = [
        r"(\bOR\b|\bAND\b).*=.*",
        r";\s*DROP\s+TABLE",
        r"UNION\s+SELECT",
        r"--",
        r"/\*.*\*/",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onclick\s*=",
    ]

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        return cls.EMAIL_PATTERN.match(email) is not None

    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username format"""
        return cls.USERNAME_PATTERN.match(username) is not None

    @classmethod
    def validate_uuid(cls, uuid_str: str) -> bool:
        """Validate UUID format"""
        return cls.UUID_PATTERN.match(uuid_str) is not None

    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        # Trim whitespace
        value = value.strip()

        # Truncate to max length
        if len(value) > max_length:
            value = value[:max_length]

        # Remove null bytes
        value = value.replace('\x00', '')

        return value

    @classmethod
    def check_sql_injection(cls, value: str) -> bool:
        """Check for SQL injection patterns"""
        value_upper = value.upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected: {value[:100]}")
                return True
        return False

    @classmethod
    def check_xss(cls, value: str) -> bool:
        """Check for XSS patterns"""
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"XSS attempt detected: {value[:100]}")
                return True
        return False

    @classmethod
    def validate_input(
        cls,
        value: Any,
        input_type: str,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        allow_none: bool = False,
    ) -> bool:
        """
        Comprehensive input validation.

        Args:
            value: Value to validate
            input_type: Type of input (email, username, uuid, string, int, float, bool)
            min_length: Minimum length (for strings)
            max_length: Maximum length (for strings)
            allow_none: Whether None is allowed

        Returns:
            True if valid, False otherwise
        """
        if value is None:
            return allow_none

        if input_type == "email":
            return isinstance(value, str) and cls.validate_email(value)

        elif input_type == "username":
            return isinstance(value, str) and cls.validate_username(value)

        elif input_type == "uuid":
            return isinstance(value, str) and cls.validate_uuid(value)

        elif input_type == "string":
            if not isinstance(value, str):
                return False
            if min_length and len(value) < min_length:
                return False
            if max_length and len(value) > max_length:
                return False
            # Check for injection attempts
            if cls.check_sql_injection(value) or cls.check_xss(value):
                return False
            return True

        elif input_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)

        elif input_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        elif input_type == "bool":
            return isinstance(value, bool)

        else:
            logger.warning(f"Unknown input type: {input_type}")
            return False


class EncryptionManager:
    """
    Encryption manager for sensitive data.

    Uses Fernet (symmetric encryption) for data encryption.
    """

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            # Generate new key
            master_key = Fernet.generate_key()
            logger.warning("Generated new master key - should be persisted securely")

        self.cipher = Fernet(master_key)
        self.master_key = master_key

        logger.info("Encryption manager initialized")

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext to ciphertext"""
        plaintext_bytes = plaintext.encode('utf-8')
        ciphertext_bytes = self.cipher.encrypt(plaintext_bytes)
        # Return base64 encoded for storage
        return base64.b64encode(ciphertext_bytes).decode('utf-8')

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt ciphertext to plaintext"""
        ciphertext_bytes = base64.b64decode(ciphertext.encode('utf-8'))
        plaintext_bytes = self.cipher.decrypt(ciphertext_bytes)
        return plaintext_bytes.decode('utf-8')

    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash password using PBKDF2.

        Returns: (hash, salt) as base64 encoded strings
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with 100k iterations
        hash_bytes = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )

        hash_b64 = base64.b64encode(hash_bytes).decode('utf-8')
        salt_b64 = base64.b64encode(salt).decode('utf-8')

        return hash_b64, salt_b64

    def verify_password(self, password: str, hash_b64: str, salt_b64: str) -> bool:
        """Verify password against hash"""
        salt = base64.b64decode(salt_b64.encode('utf-8'))
        computed_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, hash_b64)


class AuthenticationManager:
    """
    Authentication and authorization manager.

    Handles user authentication, API keys, and permissions.
    """

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}

        # Session management
        self.sessions: Dict[str, Tuple[str, datetime]] = {}  # token -> (user_id, expires)
        self.session_duration = timedelta(hours=24)

        logger.info("Authentication manager initialized")

    def create_user(
        self,
        username: str,
        password: str,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        """Create a new user"""
        # Validate username
        if not InputValidator.validate_username(username):
            raise ValueError(f"Invalid username: {username}")

        # Check if exists
        if username in self.users:
            raise ValueError(f"User {username} already exists")

        # Hash password
        password_hash, salt = self.encryption.hash_password(password)
        password_hash_with_salt = f"{password_hash}:{salt}"

        # Create user
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            password_hash=password_hash_with_salt,
            permissions=permissions or set(),
        )

        self.users[username] = user
        logger.info(f"Created user: {username}")

        return user

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate user with username/password.

        Returns: session token if successful, None otherwise
        """
        user = self.users.get(username)
        if not user or not user.is_active:
            logger.warning(f"Authentication failed: user {username} not found or inactive")
            return None

        # Parse hash and salt
        password_hash, salt = user.password_hash.split(':')

        # Verify password
        if not self.encryption.verify_password(password, password_hash, salt):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None

        # Create session
        session_token = secrets.token_urlsafe(32)
        expires = datetime.now() + self.session_duration
        self.sessions[session_token] = (user.user_id, expires)

        # Update last login
        user.last_login = datetime.now()

        logger.info(f"User {username} authenticated successfully")
        return session_token

    def validate_session(self, session_token: str) -> Optional[str]:
        """
        Validate session token.

        Returns: user_id if valid, None otherwise
        """
        if session_token not in self.sessions:
            return None

        user_id, expires = self.sessions[session_token]

        # Check if expired
        if datetime.now() > expires:
            del self.sessions[session_token]
            logger.info(f"Session expired for user {user_id}")
            return None

        return user_id

    def create_api_key(
        self,
        name: str,
        permissions: Set[Permission],
        rate_limit: int = 1000,
        expires_at: Optional[datetime] = None,
    ) -> Tuple[str, APIKey]:
        """
        Create API key.

        Returns: (raw_key, api_key_object)
        Note: raw_key is only returned once and should be stored by client
        """
        # Generate random key
        raw_key = secrets.token_urlsafe(32)

        # Hash key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )

        self.api_keys[key_hash] = api_key
        logger.info(f"Created API key: {name}")

        return raw_key, api_key

    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate API key.

        Returns: APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self.api_keys.get(key_hash)

        if not api_key or not api_key.is_active:
            logger.warning("Invalid or inactive API key")
            return None

        # Check if expired
        if api_key.expires_at and datetime.now() > api_key.expires_at:
            logger.warning(f"API key {api_key.name} expired")
            return None

        # Update last used
        api_key.last_used = datetime.now()

        return api_key

    def check_permission(
        self,
        user_id: str,
        required_permission: Permission,
    ) -> bool:
        """Check if user has required permission"""
        # Find user
        user = None
        for u in self.users.values():
            if u.user_id == user_id:
                user = u
                break

        if not user:
            return False

        # Admin has all permissions
        if Permission.ADMIN in user.permissions:
            return True

        return required_permission in user.permissions


class SecurityAuditor:
    """
    Security auditing system.

    Logs all security-relevant events for compliance and investigation.
    """

    def __init__(self):
        self.audit_log: List[SecurityAuditEntry] = []
        self.event_counts: Dict[str, int] = {}

        logger.info("Security auditor initialized")

    def log_event(
        self,
        event_type: str,
        resource: str,
        action: str,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security event"""
        entry = SecurityAuditEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            ip_address=ip_address,
            details=details or {},
        )

        self.audit_log.append(entry)
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        # Log to file/system
        log_level = logging.INFO if success else logging.WARNING
        logger.log(
            log_level,
            f"[AUDIT] {event_type}: {action} on {resource} by {user_id or 'anonymous'} - "
            f"{'SUCCESS' if success else 'FAILURE'}"
        )

    def get_audit_log(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SecurityAuditEntry]:
        """Get audit log entries"""
        filtered = self.audit_log

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        # Sort by timestamp descending
        filtered.sort(key=lambda e: e.timestamp, reverse=True)

        return filtered[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics"""
        total_events = len(self.audit_log)
        success_count = sum(1 for e in self.audit_log if e.success)
        failure_count = total_events - success_count

        # Recent activity (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_events = [e for e in self.audit_log if e.timestamp >= one_hour_ago]

        return {
            "total_events": total_events,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / total_events if total_events > 0 else 0.0,
            "events_by_type": dict(self.event_counts),
            "recent_events": len(recent_events),
        }


class SecurityHardeningEngine:
    """
    Complete security hardening engine.

    Integrates all security components:
    - Input validation
    - Encryption
    - Authentication/authorization
    - Security auditing
    """

    def __init__(self, master_key: Optional[bytes] = None):
        self.validator = InputValidator()
        self.encryption = EncryptionManager(master_key)
        self.auth = AuthenticationManager(self.encryption)
        self.auditor = SecurityAuditor()

        logger.info("Security hardening engine initialized")

    # ========== Decorators ==========

    def require_authentication(self, func: Callable) -> Callable:
        """Decorator to require authentication"""
        @functools.wraps(func)
        async def wrapper(*args, session_token: Optional[str] = None, **kwargs):
            if not session_token:
                self.auditor.log_event(
                    "authentication",
                    func.__name__,
                    "access",
                    False,
                    details={"reason": "no_token"},
                )
                raise PermissionError("Authentication required")

            user_id = self.auth.validate_session(session_token)
            if not user_id:
                self.auditor.log_event(
                    "authentication",
                    func.__name__,
                    "access",
                    False,
                    details={"reason": "invalid_token"},
                )
                raise PermissionError("Invalid or expired session")

            # Add user_id to kwargs
            kwargs['user_id'] = user_id

            # Log successful authentication
            self.auditor.log_event(
                "authentication",
                func.__name__,
                "access",
                True,
                user_id=user_id,
            )

            return await func(*args, **kwargs)

        return wrapper

    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, user_id: Optional[str] = None, **kwargs):
                if not user_id:
                    raise PermissionError("User ID required")

                if not self.auth.check_permission(user_id, permission):
                    self.auditor.log_event(
                        "authorization",
                        func.__name__,
                        permission.value,
                        False,
                        user_id=user_id,
                    )
                    raise PermissionError(f"Permission {permission.value} required")

                # Log successful authorization
                self.auditor.log_event(
                    "authorization",
                    func.__name__,
                    permission.value,
                    True,
                    user_id=user_id,
                )

                return await func(*args, user_id=user_id, **kwargs)

            return wrapper
        return decorator

    def validate_inputs(self, **input_specs):
        """
        Decorator to validate function inputs.

        Example:
            @validate_inputs(email=("email", {}), username=("username", {}))
            async def create_user(email: str, username: str):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Validate each specified input
                for param_name, (input_type, validation_kwargs) in input_specs.items():
                    if param_name in kwargs:
                        value = kwargs[param_name]
                        if not self.validator.validate_input(value, input_type, **validation_kwargs):
                            raise ValueError(f"Invalid {param_name}: {value}")

                return await func(*args, **kwargs)

            return wrapper
        return decorator

    # ========== Statistics ==========

    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        return {
            "users": len(self.auth.users),
            "active_sessions": len(self.auth.sessions),
            "api_keys": len(self.auth.api_keys),
            "audit_log": self.auditor.get_statistics(),
        }


# ========== Testing Functions ==========

async def test_security_hardening():
    """
    Test security hardening capabilities

    NOTE: For security testing, use environment variables or the test suite in tests/test_security_hardening.py
    This demo uses environment variables with safe defaults for demonstration only.
    """
    import os

    print("\n" + "="*60)
    print("Testing Security Hardening Engine")
    print("="*60)

    engine = SecurityHardeningEngine()

    # Test 1: Input Validation
    print("\n1. Testing input validation...")

    test_inputs = [
        ("email", "user@example.com", True),
        ("email", "invalid-email", False),
        ("username", "valid_user123", True),
        ("username", "invalid user!", False),
        ("string", "SELECT * FROM users", False),  # SQL injection
        ("string", "<script>alert('xss')</script>", False),  # XSS
    ]

    for input_type, value, expected_valid in test_inputs:
        is_valid = engine.validator.validate_input(value, input_type)
        status = "PASS" if is_valid == expected_valid else "FAIL"
        print(f"  {status} {input_type}: '{value}' - {'valid' if is_valid else 'invalid'}")

    # Test 2: Encryption
    print("\n2. Testing encryption...")

    plaintext = "sensitive_data_12345"
    ciphertext = engine.encryption.encrypt(plaintext)
    decrypted = engine.encryption.decrypt(ciphertext)

    print(f"  Plaintext: {plaintext}")
    print(f"  Ciphertext: {ciphertext[:50]}...")
    print(f"  Decrypted: {decrypted}")
    print(f"  Match: {'PASS' if plaintext == decrypted else 'FAIL'}")

    # Test 3: User Authentication
    print("\n3. Testing user authentication...")

    # Use environment variables for credentials (safe for demo)
    test_username = os.getenv("SHIVX_DEMO_USERNAME", "demo_user")
    test_password = os.getenv("SHIVX_DEMO_PASSWORD", "demo_password_secure_456")

    # Create user
    user = engine.auth.create_user(
        test_username,
        test_password,
        permissions={Permission.READ, Permission.WRITE},
    )
    print(f"  Created user: {user.username} (ID: {user.user_id})")

    # Authenticate
    session_token = engine.auth.authenticate_user(test_username, test_password)
    print(f"  Authenticated: {'PASS' if session_token else 'FAIL'}")

    # Validate session
    user_id = engine.auth.validate_session(session_token)
    print(f"  Session valid: {'PASS' if user_id else 'FAIL'}")

    # Wrong password
    failed_token = engine.auth.authenticate_user(test_username, "intentionally_wrong_password")
    print(f"  Wrong password rejected: {'PASS' if not failed_token else 'FAIL'}")

    # Test 4: API Keys
    print("\n4. Testing API keys...")

    raw_key, api_key = engine.auth.create_api_key(
        "test_api",
        permissions={Permission.READ},
        rate_limit=100,
    )
    print(f"  Created API key: {api_key.name}")
    print(f"  Raw key: {raw_key[:20]}...")

    # Validate key
    validated = engine.auth.validate_api_key(raw_key)
    print(f"  Key valid: {'PASS' if validated else 'FAIL'}")

    # Invalid key
    invalid_validated = engine.auth.validate_api_key("invalid_key_abc123")
    print(f"  Invalid key rejected: {'PASS' if not invalid_validated else 'FAIL'}")

    # Test 5: Permissions
    print("\n5. Testing permissions...")

    has_read = engine.auth.check_permission(user_id, Permission.READ)
    has_admin = engine.auth.check_permission(user_id, Permission.ADMIN)

    print(f"  User has READ: {'PASS' if has_read else 'FAIL'}")
    print(f"  User has ADMIN: {'PASS' if not has_admin else 'FAIL'} (should not)")

    # Test 6: Security Auditing
    print("\n6. Testing security auditing...")

    engine.auditor.log_event(
        "authentication",
        "test_resource",
        "login",
        True,
        user_id=user_id,
        ip_address="127.0.0.1",
    )

    engine.auditor.log_event(
        "authorization",
        "secret_data",
        "access",
        False,
        user_id=user_id,
    )

    audit_stats = engine.auditor.get_statistics()
    print(f"  Total audit events: {audit_stats['total_events']}")
    print(f"  Success rate: {audit_stats['success_rate']:.1%}")
    print(f"  Events by type: {audit_stats['events_by_type']}")

    # Test 7: Protected Function
    print("\n7. Testing protected functions...")

    @engine.require_authentication
    @engine.require_permission(Permission.READ)
    async def protected_function(user_id: str):
        return f"Access granted for user {user_id}"

    try:
        result = await protected_function(session_token=session_token)
        print(f"  PASS Protected function accessed: {result}")
    except Exception as e:
        print(f"  FAIL Access denied: {e}")

    # Try without authentication
    try:
        await protected_function()
        print("  FAIL Unauthorized access allowed (should not happen)")
    except PermissionError as e:
        print(f"  PASS Unauthorized access blocked: {e}")

    # Final Statistics
    print("\n" + "="*60)
    print("Security Statistics")
    print("="*60)

    stats = engine.get_security_stats()
    print(f"Users: {stats['users']}")
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"API keys: {stats['api_keys']}")
    print(f"Audit events: {stats['audit_log']['total_events']}")
    print(f"Audit success rate: {stats['audit_log']['success_rate']:.1%}")

    return engine


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_security_hardening())
