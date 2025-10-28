"""
Content Moderation for ShivX

Provides content moderation to filter harmful content:
- Violence and threats
- Sexual content
- Hate speech
- Self-harm content
- Spam and abuse
- Integration with OpenAI Moderation API

Note: This implementation includes both API-based and pattern-based moderation
for defense-in-depth.
"""

import re
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ModerationCategory(Enum):
    """Content moderation categories"""
    VIOLENCE = "violence"
    VIOLENCE_GRAPHIC = "violence/graphic"
    SEXUAL = "sexual"
    SEXUAL_MINORS = "sexual/minors"
    HATE = "hate"
    HATE_THREATENING = "hate/threatening"
    SELF_HARM = "self-harm"
    SELF_HARM_INTENT = "self-harm/intent"
    SELF_HARM_INSTRUCTIONS = "self-harm/instructions"
    HARASSMENT = "harassment"
    HARASSMENT_THREATENING = "harassment/threatening"
    SPAM = "spam"


class ModerationSeverity(Enum):
    """Severity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModerationResult:
    """Result of content moderation"""
    is_safe: bool
    flagged_categories: List[ModerationCategory]
    severity: ModerationSeverity
    confidence: float
    category_scores: Dict[str, float]
    reason: str
    metadata: Dict[str, Any]


class ContentModerationFilter:
    """
    Content moderation filter with multiple detection methods

    Uses:
    1. Pattern-based detection (fast, local)
    2. OpenAI Moderation API (when API key available)
    3. Keyword blocklists
    """

    # Pattern-based detection for offline/backup moderation
    VIOLENCE_PATTERNS = [
        r"\b(kill|murder|assassinate|slaughter|massacre|torture)\b",
        r"\b(stab|shoot|strangle|suffocate|poison)\b",
        r"\b(bomb|explosive|weapon|gun|rifle)\b",
        r"\b(attack|assault|fight|hurt|harm|injure)\b",
    ]

    SEXUAL_PATTERNS = [
        r"\b(explicit sexual content patterns - redacted for safety)\b",
        # Note: Actual implementation would include appropriate patterns
        # This is a placeholder for demonstration
    ]

    HATE_PATTERNS = [
        r"\b(hate speech patterns - redacted for safety)\b",
        # Note: Actual implementation would include slurs and hate speech
        # This is a placeholder for demonstration
    ]

    SELF_HARM_PATTERNS = [
        r"\b(suicide|self[- ]harm|cut myself|end (my|your) life)\b",
        r"\b(kill (myself|yourself|themselves))\b",
        r"how to (die|commit suicide|hurt myself)",
    ]

    HARASSMENT_PATTERNS = [
        r"\b(harassment patterns - redacted for safety)\b",
        # Patterns for threats, doxxing, etc.
    ]

    SPAM_PATTERNS = [
        r"(?i)(click here|buy now|limited time|act now).*(http|www)",
        r"(?i)(get rich quick|make money fast|work from home)",
        r"(?i)(congratulations|you('ve| have) won)",
        r"(?i)(verify your account|confirm your identity).*(click|link)",
    ]

    def __init__(self, use_api: bool = True, api_key: Optional[str] = None):
        """
        Initialize content moderation filter

        Args:
            use_api: Whether to use OpenAI Moderation API
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.use_api = use_api
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Compile patterns
        self.compiled_patterns = {
            ModerationCategory.VIOLENCE: [
                re.compile(p, re.IGNORECASE) for p in self.VIOLENCE_PATTERNS
            ],
            ModerationCategory.SEXUAL: [
                re.compile(p, re.IGNORECASE) for p in self.SEXUAL_PATTERNS
            ],
            ModerationCategory.HATE: [
                re.compile(p, re.IGNORECASE) for p in self.HATE_PATTERNS
            ],
            ModerationCategory.SELF_HARM: [
                re.compile(p, re.IGNORECASE) for p in self.SELF_HARM_PATTERNS
            ],
            ModerationCategory.HARASSMENT: [
                re.compile(p, re.IGNORECASE) for p in self.HARASSMENT_PATTERNS
            ],
            ModerationCategory.SPAM: [
                re.compile(p, re.IGNORECASE) for p in self.SPAM_PATTERNS
            ],
        }

        logger.info(
            f"Content moderation initialized (api_enabled={self.use_api and self.api_key is not None})"
        )

    async def moderate(self, text: str) -> ModerationResult:
        """
        Moderate content using available methods

        Args:
            text: Text to moderate

        Returns:
            ModerationResult
        """
        if not text or not text.strip():
            return ModerationResult(
                is_safe=True,
                flagged_categories=[],
                severity=ModerationSeverity.SAFE,
                confidence=1.0,
                category_scores={},
                reason="Empty input",
                metadata={}
            )

        # Try API-based moderation first (more accurate)
        if self.use_api and self.api_key:
            try:
                return await self._moderate_with_api(text)
            except Exception as e:
                logger.warning(f"API moderation failed, falling back to pattern-based: {e}")

        # Fallback to pattern-based moderation
        return self._moderate_with_patterns(text)

    async def _moderate_with_api(self, text: str) -> ModerationResult:
        """
        Moderate content using OpenAI Moderation API

        Args:
            text: Text to moderate

        Returns:
            ModerationResult
        """
        try:
            import openai
            openai.api_key = self.api_key

            # Call moderation API
            response = await asyncio.to_thread(
                lambda: openai.Moderation.create(input=text)
            )

            result = response["results"][0]
            flagged = result["flagged"]
            categories = result["categories"]
            category_scores = result["category_scores"]

            # Extract flagged categories
            flagged_categories = []
            max_score = 0.0

            for category, is_flagged in categories.items():
                if is_flagged:
                    try:
                        cat_enum = ModerationCategory(category)
                        flagged_categories.append(cat_enum)
                    except ValueError:
                        # Unknown category
                        pass

                # Track max score for severity
                score = category_scores.get(category, 0.0)
                max_score = max(max_score, score)

            # Determine severity
            if max_score >= 0.9:
                severity = ModerationSeverity.CRITICAL
            elif max_score >= 0.7:
                severity = ModerationSeverity.HIGH
            elif max_score >= 0.5:
                severity = ModerationSeverity.MEDIUM
            elif max_score >= 0.3:
                severity = ModerationSeverity.LOW
            else:
                severity = ModerationSeverity.SAFE

            reason = "Flagged by OpenAI Moderation API" if flagged else "Passed moderation"

            if flagged:
                logger.warning(
                    f"Content flagged by moderation API: categories={[c.value for c in flagged_categories]}, "
                    f"severity={severity.value}, max_score={max_score:.2f}"
                )

            return ModerationResult(
                is_safe=not flagged,
                flagged_categories=flagged_categories,
                severity=severity,
                confidence=max_score,
                category_scores=category_scores,
                reason=reason,
                metadata={"method": "openai_api"}
            )

        except ImportError:
            logger.warning("OpenAI library not installed, falling back to pattern-based")
            return self._moderate_with_patterns(text)
        except Exception as e:
            logger.error(f"OpenAI moderation API error: {e}")
            raise

    def _moderate_with_patterns(self, text: str) -> ModerationResult:
        """
        Moderate content using pattern matching

        Args:
            text: Text to moderate

        Returns:
            ModerationResult
        """
        flagged_categories = []
        category_scores = {}
        max_score = 0.0

        # Check each category
        for category, patterns in self.compiled_patterns.items():
            matches = 0
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1

            # Calculate score based on matches
            score = min(matches / len(patterns), 1.0) if patterns else 0.0
            category_scores[category.value] = score

            if score > 0.5:  # Threshold for flagging
                flagged_categories.append(category)
                max_score = max(max_score, score)

        # Determine severity
        if max_score >= 0.9:
            severity = ModerationSeverity.CRITICAL
        elif max_score >= 0.7:
            severity = ModerationSeverity.HIGH
        elif max_score >= 0.5:
            severity = ModerationSeverity.MEDIUM
        elif max_score >= 0.3:
            severity = ModerationSeverity.LOW
        else:
            severity = ModerationSeverity.SAFE

        is_safe = len(flagged_categories) == 0
        reason = "Flagged by pattern matching" if not is_safe else "Passed moderation"

        if not is_safe:
            logger.warning(
                f"Content flagged by pattern matching: categories={[c.value for c in flagged_categories]}, "
                f"severity={severity.value}, max_score={max_score:.2f}"
            )

        return ModerationResult(
            is_safe=is_safe,
            flagged_categories=flagged_categories,
            severity=severity,
            confidence=max_score,
            category_scores=category_scores,
            reason=reason,
            metadata={"method": "pattern_matching"}
        )

    def moderate_sync(self, text: str) -> ModerationResult:
        """
        Synchronous wrapper for moderation

        Args:
            text: Text to moderate

        Returns:
            ModerationResult
        """
        # For sync contexts, only use pattern-based
        return self._moderate_with_patterns(text)


# Global moderation instance
_moderation_instance: Optional[ContentModerationFilter] = None


def get_content_moderator(
    use_api: bool = True,
    api_key: Optional[str] = None
) -> ContentModerationFilter:
    """Get or create global content moderator instance"""
    global _moderation_instance
    if _moderation_instance is None:
        _moderation_instance = ContentModerationFilter(
            use_api=use_api,
            api_key=api_key
        )
    return _moderation_instance


# Convenience functions
async def moderate_content(text: str) -> ModerationResult:
    """Quick content moderation"""
    moderator = get_content_moderator()
    return await moderator.moderate(text)


def moderate_content_sync(text: str) -> ModerationResult:
    """Quick synchronous content moderation"""
    moderator = get_content_moderator()
    return moderator.moderate_sync(text)


def is_content_safe(text: str) -> bool:
    """Quick safety check"""
    result = moderate_content_sync(text)
    return result.is_safe
