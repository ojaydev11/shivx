"""
Data Loss Prevention (DLP) for ShivX

Provides comprehensive PII and secrets detection and redaction:
- PII: SSN, email, phone, credit cards
- Secrets: API keys, JWT tokens, passwords, private keys
- Automatic redaction
- Logging integration

CVSS Score: 8.7 (High)
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SensitiveDataType(Enum):
    """Types of sensitive data"""
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    AWS_KEY = "aws_key"
    GITHUB_TOKEN = "github_token"
    STRIPE_KEY = "stripe_key"
    GENERIC_SECRET = "generic_secret"


@dataclass
class DetectionResult:
    """Result of sensitive data detection"""
    found_sensitive_data: bool
    detections: List[Tuple[SensitiveDataType, str]]
    redacted_text: str
    original_length: int
    redacted_count: int
    metadata: Dict[str, any]


class DataLossPreventionFilter:
    """
    Comprehensive DLP filter for detecting and redacting sensitive data

    Detects:
    - PII: SSN, email addresses, phone numbers, credit cards
    - API Keys: AWS, Stripe, GitHub, OpenAI, Google, etc.
    - JWT tokens
    - Private keys (RSA, SSH, PGP)
    - Passwords in common formats
    """

    # PII Patterns
    PII_PATTERNS = [
        # US SSN: 123-45-6789 or 123 45 6789 or 123456789
        (
            SensitiveDataType.SSN,
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            "SSN"
        ),
        # Email addresses
        (
            SensitiveDataType.EMAIL,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "EMAIL"
        ),
        # US Phone numbers: (123) 456-7890, 123-456-7890, 1234567890
        (
            SensitiveDataType.PHONE,
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "PHONE"
        ),
        # Credit card numbers (basic pattern, supports major cards)
        (
            SensitiveDataType.CREDIT_CARD,
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "CREDIT_CARD"
        ),
    ]

    # API Key and Secret Patterns
    SECRET_PATTERNS = [
        # OpenAI API Keys
        (
            SensitiveDataType.API_KEY,
            r"sk-[a-zA-Z0-9]{32,}",
            "OPENAI_API_KEY"
        ),
        # AWS Access Keys
        (
            SensitiveDataType.AWS_KEY,
            r"AKIA[0-9A-Z]{16}",
            "AWS_ACCESS_KEY"
        ),
        # AWS Secret Keys (more complex)
        (
            SensitiveDataType.AWS_KEY,
            r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key[\"'\s:=]+([a-zA-Z0-9/+=]{40})",
            "AWS_SECRET_KEY"
        ),
        # GitHub Personal Access Tokens
        (
            SensitiveDataType.GITHUB_TOKEN,
            r"ghp_[a-zA-Z0-9]{36}",
            "GITHUB_PAT"
        ),
        # GitHub OAuth Tokens
        (
            SensitiveDataType.GITHUB_TOKEN,
            r"gho_[a-zA-Z0-9]{36}",
            "GITHUB_OAUTH"
        ),
        # Stripe API Keys
        (
            SensitiveDataType.STRIPE_KEY,
            r"(?:r|s)k_live_[a-zA-Z0-9]{24,}",
            "STRIPE_API_KEY"
        ),
        # Google API Keys
        (
            SensitiveDataType.API_KEY,
            r"AIza[0-9A-Za-z_-]{35}",
            "GOOGLE_API_KEY"
        ),
        # Google OAuth Access Tokens
        (
            SensitiveDataType.API_KEY,
            r"ya29\.[0-9A-Za-z_-]{68,}",
            "GOOGLE_OAUTH_TOKEN"
        ),
        # JWT Tokens
        (
            SensitiveDataType.JWT_TOKEN,
            r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
            "JWT_TOKEN"
        ),
        # Generic API key patterns
        (
            SensitiveDataType.API_KEY,
            r"(?i)(?:api[_\-]?key|apikey)[\"'\s:=]+([a-zA-Z0-9_\-]{16,})",
            "API_KEY"
        ),
        # Slack tokens
        (
            SensitiveDataType.API_KEY,
            r"xox[baprs]-[a-zA-Z0-9]{10,72}",
            "SLACK_TOKEN"
        ),
        # Facebook Access Tokens
        (
            SensitiveDataType.API_KEY,
            r"EAACEdEose0cBA[0-9A-Za-z]+",
            "FACEBOOK_ACCESS_TOKEN"
        ),
        # Twitter API Keys
        (
            SensitiveDataType.API_KEY,
            r"[1-9][0-9]+-[0-9a-zA-Z]{40}",
            "TWITTER_API_KEY"
        ),
    ]

    # Private Key Patterns
    PRIVATE_KEY_PATTERNS = [
        # RSA Private Keys
        (
            SensitiveDataType.PRIVATE_KEY,
            r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA )?PRIVATE KEY-----",
            "RSA_PRIVATE_KEY"
        ),
        # SSH Private Keys
        (
            SensitiveDataType.PRIVATE_KEY,
            r"-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]+?-----END OPENSSH PRIVATE KEY-----",
            "SSH_PRIVATE_KEY"
        ),
        # PGP Private Keys
        (
            SensitiveDataType.PRIVATE_KEY,
            r"-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]+?-----END PGP PRIVATE KEY BLOCK-----",
            "PGP_PRIVATE_KEY"
        ),
    ]

    # Password Patterns (common formats)
    PASSWORD_PATTERNS = [
        # password=... or password: ...
        (
            SensitiveDataType.PASSWORD,
            r"(?i)(?:password|passwd|pwd)[\"'\s:=]+([^\s\"']{8,})",
            "PASSWORD"
        ),
        # Common database connection strings
        (
            SensitiveDataType.PASSWORD,
            r"(?i)(?:mysql|postgresql|mongodb)://[^:]+:([^@]+)@",
            "DB_PASSWORD"
        ),
    ]

    def __init__(self, enable_logging: bool = True):
        """
        Initialize DLP filter

        Args:
            enable_logging: If True, log all detections
        """
        self.enable_logging = enable_logging

        # Compile all patterns
        self.all_patterns = []

        for data_type, pattern, name in (
            self.PII_PATTERNS +
            self.SECRET_PATTERNS +
            self.PRIVATE_KEY_PATTERNS +
            self.PASSWORD_PATTERNS
        ):
            self.all_patterns.append((
                data_type,
                re.compile(pattern, re.IGNORECASE | re.MULTILINE),
                name
            ))

        logger.info("DLP filter initialized with {} patterns".format(len(self.all_patterns)))

    def scan(self, text: str) -> DetectionResult:
        """
        Scan text for sensitive data

        Args:
            text: Text to scan

        Returns:
            DetectionResult with findings and redacted text
        """
        if not text or not text.strip():
            return DetectionResult(
                found_sensitive_data=False,
                detections=[],
                redacted_text=text,
                original_length=len(text) if text else 0,
                redacted_count=0,
                metadata={}
            )

        detections = []
        redacted_text = text
        redaction_count = 0

        # Scan for all patterns
        for data_type, pattern, name in self.all_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                matched_text = match.group(0)
                detections.append((data_type, name))

                # Redact the sensitive data
                redaction_label = f"[REDACTED-{name}]"
                redacted_text = redacted_text.replace(matched_text, redaction_label)
                redaction_count += 1

                # Log detection
                if self.enable_logging:
                    logger.warning(
                        f"DLP DETECTION: {name} found in text (type={data_type.value})"
                    )

        found_sensitive = len(detections) > 0

        # Additional metadata
        metadata = {
            "scan_length": len(text),
            "detection_types": list(set(d[0].value for d in detections)),
            "unique_types_count": len(set(d[1] for d in detections)),
        }

        return DetectionResult(
            found_sensitive_data=found_sensitive,
            detections=detections,
            redacted_text=redacted_text,
            original_length=len(text),
            redacted_count=redaction_count,
            metadata=metadata
        )

    def redact(self, text: str) -> str:
        """
        Redact sensitive data from text (convenience method)

        Args:
            text: Text to redact

        Returns:
            Redacted text
        """
        result = self.scan(text)
        return result.redacted_text

    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains PII

        Args:
            text: Text to check

        Returns:
            True if PII detected
        """
        result = self.scan(text)
        pii_types = {SensitiveDataType.SSN, SensitiveDataType.EMAIL,
                     SensitiveDataType.PHONE, SensitiveDataType.CREDIT_CARD}
        return any(d[0] in pii_types for d in result.detections)

    def contains_secrets(self, text: str) -> bool:
        """
        Check if text contains secrets/credentials

        Args:
            text: Text to check

        Returns:
            True if secrets detected
        """
        result = self.scan(text)
        secret_types = {SensitiveDataType.API_KEY, SensitiveDataType.AWS_KEY,
                       SensitiveDataType.GITHUB_TOKEN, SensitiveDataType.STRIPE_KEY,
                       SensitiveDataType.JWT_TOKEN, SensitiveDataType.PASSWORD,
                       SensitiveDataType.PRIVATE_KEY}
        return any(d[0] in secret_types for d in result.detections)

    def get_detection_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of detected sensitive data types

        Args:
            text: Text to scan

        Returns:
            Dictionary mapping data type to count
        """
        result = self.scan(text)
        summary = {}
        for data_type, name in result.detections:
            key = data_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary


# Global DLP instance
_dlp_instance: Optional[DataLossPreventionFilter] = None


def get_dlp_filter(enable_logging: bool = True) -> DataLossPreventionFilter:
    """Get or create global DLP filter instance"""
    global _dlp_instance
    if _dlp_instance is None:
        _dlp_instance = DataLossPreventionFilter(enable_logging=enable_logging)
    return _dlp_instance


# Convenience functions
def scan_for_sensitive_data(text: str) -> DetectionResult:
    """Quick scan for sensitive data"""
    dlp = get_dlp_filter()
    return dlp.scan(text)


def redact_sensitive_data(text: str) -> str:
    """Quick redaction of sensitive data"""
    dlp = get_dlp_filter()
    return dlp.redact(text)


def contains_pii(text: str) -> bool:
    """Quick check for PII"""
    dlp = get_dlp_filter()
    return dlp.contains_pii(text)


def contains_secrets(text: str) -> bool:
    """Quick check for secrets"""
    dlp = get_dlp_filter()
    return dlp.contains_secrets(text)
