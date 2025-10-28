"""
Prompt Injection Filter for ShivX

Provides comprehensive protection against prompt injection attacks:
- Input filtering with keyword detection
- Encoding detection (Base64, ROT13, Unicode)
- Output validation for leaked secrets
- Pattern-based detection

CVSS Score: 9.3 (Critical)
"""

import re
import logging
import base64
import codecs
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat level classification"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    """Result of prompt filtering"""
    is_safe: bool
    threat_level: ThreatLevel
    detected_threats: List[str]
    sanitized_input: Optional[str]
    confidence: float
    metadata: Dict[str, any]


class PromptInjectionFilter:
    """
    Advanced prompt injection filter with multi-layer detection

    Protects against:
    - Direct instruction overrides
    - Role manipulation attempts
    - System prompt extraction
    - Encoded payloads (Base64, ROT13, Unicode)
    - Jailbreak attempts
    - Secret exfiltration
    """

    # Critical patterns that indicate prompt injection
    INJECTION_PATTERNS = [
        # Direct override attempts
        r"ignore\s+(previous|all|earlier)\s+(instructions|prompts|rules)",
        r"disregard\s+(previous|all|earlier)\s+(instructions|prompts|rules)",
        r"forget\s+(previous|all|earlier)\s+(instructions|prompts|rules)",
        r"override\s+(previous|all|system)\s+(instructions|prompts|rules)",

        # Role manipulation
        r"you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode",
        r"act\s+as\s+(a\s+)?(\w+)",
        r"pretend\s+to\s+be\s+(a\s+)?(\w+)",
        r"simulate\s+(a\s+)?(\w+)",
        r"roleplay\s+as\s+(a\s+)?(\w+)",

        # System prompt extraction
        r"(print|show|reveal|display|tell\s+me|what\s+is)\s+(your|the)\s+system\s+prompt",
        r"(print|show|reveal|display)\s+(all|everything)",
        r"repeat\s+(your|the)\s+(instructions|prompt|rules)",
        r"what\s+(are|were)\s+your\s+(original|initial)\s+instructions",

        # Jailbreak patterns
        r"DAN\s+mode",
        r"developer\s+mode",
        r"god\s+mode",
        r"admin\s+mode",
        r"root\s+access",
        r"bypass\s+(safety|security|filters|restrictions)",
        r"disable\s+(safety|security|filters|restrictions)",

        # Prompt ending/continuation
        r"---\s*END\s+OF\s+(PROMPT|INSTRUCTIONS|RULES)",
        r"<\|endoftext\|>",
        r"<\|im_end\|>",
        r"###\s+NEW\s+INSTRUCTIONS",

        # Output manipulation
        r"print\s+.*?(api[_-]?key|password|secret|token)",
        r"reveal\s+.*?(api[_-]?key|password|secret|token)",
        r"show\s+.*?(api[_-]?key|password|secret|token)",

        # Multi-language injection attempts
        r"traduire|翻译|traducir|übersetzen",  # translate in multiple languages
        r"ejecutar|ausführen|执行|実行",  # execute in multiple languages
    ]

    # Encoding detection patterns
    ENCODING_PATTERNS = {
        "base64": r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
        "rot13": r"[a-zA-Z]{20,}",  # Long strings of only letters
        "unicode_escape": r"\\u[0-9a-fA-F]{4}",
        "hex_escape": r"\\x[0-9a-fA-F]{2}",
    }

    # Output patterns to detect leaked secrets
    OUTPUT_SECRET_PATTERNS = [
        # API Keys
        (r"sk-[a-zA-Z0-9]{32,}", "OPENAI_API_KEY"),
        (r"pk_live_[a-zA-Z0-9]{24,}", "STRIPE_API_KEY"),
        (r"rk_live_[a-zA-Z0-9]{24,}", "STRIPE_RESTRICTED_KEY"),
        (r"AIza[0-9A-Za-z_-]{35}", "GOOGLE_API_KEY"),
        (r"ya29\.[0-9A-Za-z_-]{68,}", "GOOGLE_OAUTH_TOKEN"),
        (r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY"),
        (r"ghp_[a-zA-Z0-9]{36}", "GITHUB_PAT"),
        (r"gho_[a-zA-Z0-9]{36}", "GITHUB_OAUTH"),

        # JWT Tokens
        (r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+", "JWT_TOKEN"),

        # Private Keys
        (r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----", "PRIVATE_KEY"),
        (r"-----BEGIN (OPENSSH|PGP) PRIVATE KEY-----", "SSH_PRIVATE_KEY"),

        # Generic secrets
        (r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?[^'\"\s]{8,}", "PASSWORD"),
        (r"(?i)(secret|api[_-]?key)\s*[:=]\s*['\"]?[^'\"\s]{16,}", "SECRET"),
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize prompt injection filter

        Args:
            strict_mode: If True, be more aggressive with detection
        """
        self.strict_mode = strict_mode
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.INJECTION_PATTERNS
        ]
        self.output_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in self.OUTPUT_SECRET_PATTERNS
        ]

        logger.info(f"Prompt injection filter initialized (strict_mode={strict_mode})")

    def filter_input(self, user_input: str) -> FilterResult:
        """
        Filter and validate user input for prompt injection attempts

        Args:
            user_input: User-provided input to validate

        Returns:
            FilterResult with detection results
        """
        if not user_input or not user_input.strip():
            return FilterResult(
                is_safe=True,
                threat_level=ThreatLevel.SAFE,
                detected_threats=[],
                sanitized_input=user_input,
                confidence=1.0,
                metadata={}
            )

        detected_threats = []
        threat_scores = []

        # 1. Check for direct injection patterns
        pattern_threats = self._check_injection_patterns(user_input)
        detected_threats.extend(pattern_threats)
        if pattern_threats:
            threat_scores.append(0.9)

        # 2. Check for encoding attempts
        encoding_threats = self._check_encoding(user_input)
        detected_threats.extend(encoding_threats)
        if encoding_threats:
            threat_scores.append(0.7)

        # 3. Check for suspicious character patterns
        char_threats = self._check_suspicious_chars(user_input)
        detected_threats.extend(char_threats)
        if char_threats:
            threat_scores.append(0.6)

        # 4. Check input length and complexity
        complexity_threats = self._check_complexity(user_input)
        detected_threats.extend(complexity_threats)
        if complexity_threats:
            threat_scores.append(0.5)

        # Calculate overall threat level
        if not detected_threats:
            threat_level = ThreatLevel.SAFE
            confidence = 1.0
            is_safe = True
        else:
            avg_score = sum(threat_scores) / len(threat_scores) if threat_scores else 0
            if avg_score >= 0.8:
                threat_level = ThreatLevel.CRITICAL
                is_safe = False
                confidence = avg_score
            elif avg_score >= 0.6:
                threat_level = ThreatLevel.DANGEROUS
                is_safe = not self.strict_mode
                confidence = avg_score
            elif avg_score >= 0.4:
                threat_level = ThreatLevel.SUSPICIOUS
                is_safe = not self.strict_mode
                confidence = avg_score
            else:
                threat_level = ThreatLevel.SAFE
                is_safe = True
                confidence = 1.0 - avg_score

        # Log security events
        if not is_safe:
            logger.warning(
                f"Prompt injection detected: threats={detected_threats}, "
                f"threat_level={threat_level.value}, confidence={confidence:.2f}"
            )

        return FilterResult(
            is_safe=is_safe,
            threat_level=threat_level,
            detected_threats=detected_threats,
            sanitized_input=self._sanitize_input(user_input) if is_safe else None,
            confidence=confidence,
            metadata={
                "input_length": len(user_input),
                "threat_count": len(detected_threats),
                "strict_mode": self.strict_mode
            }
        )

    def filter_output(self, model_output: str) -> FilterResult:
        """
        Filter model output to detect leaked secrets or sensitive data

        Args:
            model_output: LLM-generated output to validate

        Returns:
            FilterResult with detection results
        """
        if not model_output or not model_output.strip():
            return FilterResult(
                is_safe=True,
                threat_level=ThreatLevel.SAFE,
                detected_threats=[],
                sanitized_input=model_output,
                confidence=1.0,
                metadata={}
            )

        detected_threats = []

        # Check for leaked secrets
        for pattern, secret_type in self.output_patterns:
            matches = pattern.findall(model_output)
            if matches:
                detected_threats.append(f"LEAKED_{secret_type}")
                logger.critical(
                    f"SECRET LEAK DETECTED: {secret_type} found in model output"
                )

        is_safe = len(detected_threats) == 0
        threat_level = ThreatLevel.CRITICAL if detected_threats else ThreatLevel.SAFE

        return FilterResult(
            is_safe=is_safe,
            threat_level=threat_level,
            detected_threats=detected_threats,
            sanitized_input=self._redact_secrets(model_output) if not is_safe else model_output,
            confidence=1.0 if is_safe else 0.0,
            metadata={
                "output_length": len(model_output),
                "leaked_secrets": len(detected_threats)
            }
        )

    def _check_injection_patterns(self, text: str) -> List[str]:
        """Check for known injection patterns"""
        threats = []
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                threats.append(f"INJECTION_PATTERN: {pattern.pattern[:50]}")
        return threats

    def _check_encoding(self, text: str) -> List[str]:
        """Check for encoded payloads"""
        threats = []

        # Check Base64
        if len(text) > 20:
            try:
                # Try to decode as base64
                decoded = base64.b64decode(text, validate=True).decode('utf-8', errors='ignore')
                # If decoded contains injection patterns, flag it
                if any(pattern.search(decoded) for pattern in self.compiled_patterns):
                    threats.append("BASE64_ENCODED_INJECTION")
            except Exception:
                pass

        # Check ROT13
        try:
            rot13_decoded = codecs.decode(text, 'rot_13')
            if any(pattern.search(rot13_decoded) for pattern in self.compiled_patterns):
                threats.append("ROT13_ENCODED_INJECTION")
        except Exception:
            pass

        # Check Unicode escapes
        if '\\u' in text or '\\x' in text:
            try:
                unicode_decoded = text.encode().decode('unicode_escape')
                if any(pattern.search(unicode_decoded) for pattern in self.compiled_patterns):
                    threats.append("UNICODE_ENCODED_INJECTION")
            except Exception:
                pass

        return threats

    def _check_suspicious_chars(self, text: str) -> List[str]:
        """Check for suspicious character patterns"""
        threats = []

        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_char_ratio > 0.3:
            threats.append(f"EXCESSIVE_SPECIAL_CHARS: {special_char_ratio:.2%}")

        # Check for null bytes or control characters
        if '\x00' in text or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
            threats.append("CONTROL_CHARACTERS")

        # Check for repeated patterns (potential obfuscation)
        if re.search(r'(.{3,})\1{3,}', text):
            threats.append("REPEATED_PATTERNS")

        return threats

    def _check_complexity(self, text: str) -> List[str]:
        """Check input complexity for anomalies"""
        threats = []

        # Very long inputs might be attempts to overwhelm
        if len(text) > 10000:
            threats.append(f"EXCESSIVE_LENGTH: {len(text)} chars")

        # Check for very long single words (potential encoding)
        words = text.split()
        max_word_len = max((len(w) for w in words), default=0)
        if max_word_len > 200:
            threats.append(f"EXCESSIVE_WORD_LENGTH: {max_word_len}")

        return threats

    def _sanitize_input(self, text: str) -> str:
        """Sanitize input by removing/escaping dangerous content"""
        # For now, just return the original (already validated as safe)
        # In a stricter mode, we could strip out suspicious patterns
        return text

    def _redact_secrets(self, text: str) -> str:
        """Redact detected secrets from output"""
        redacted = text
        for pattern, secret_type in self.output_patterns:
            redacted = pattern.sub(f"[REDACTED-{secret_type}]", redacted)
        return redacted


# Global filter instance
_filter_instance: Optional[PromptInjectionFilter] = None


def get_prompt_filter(strict_mode: bool = True) -> PromptInjectionFilter:
    """Get or create global prompt filter instance"""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = PromptInjectionFilter(strict_mode=strict_mode)
    return _filter_instance


# Convenience functions
def validate_prompt(user_input: str, strict: bool = True) -> FilterResult:
    """Quick validation of user prompt"""
    filter_obj = get_prompt_filter(strict_mode=strict)
    return filter_obj.filter_input(user_input)


def validate_output(model_output: str) -> FilterResult:
    """Quick validation of model output"""
    filter_obj = get_prompt_filter()
    return filter_obj.filter_output(model_output)
