"""
Content Moderation Tests
Tests for harmful content detection and moderation
"""

import pytest
from utils.content_moderation import ContentModerationFilter, ModerationSeverity


@pytest.fixture
def moderator():
    """Fixture for content moderator"""
    return ContentModerationFilter(use_api=False)  # Use pattern-based for tests


# =============================================================================
# Test Violence Detection
# =============================================================================

@pytest.mark.security
class TestViolenceDetection:
    """Test violence and threat detection"""

    def test_detect_violence_threats(self, moderator):
        """Test: Detect violent threats"""
        texts = [
            "Planning to attack tomorrow",
            "Going to harm someone",
        ]

        for text in texts:
            result = moderator.moderate_sync(text)
            assert not result.is_safe or result.severity != ModerationSeverity.SAFE


# =============================================================================
# Test Safe Content
# =============================================================================

@pytest.mark.security
class TestSafeContent:
    """Test that safe content passes"""

    def test_normal_conversation(self, moderator):
        """Test: Normal conversation is safe"""
        texts = [
            "Hello, how are you?",
            "The weather is nice today",
            "I love programming in Python",
        ]

        for text in texts:
            result = moderator.moderate_sync(text)
            assert result.is_safe
            assert result.severity == ModerationSeverity.SAFE

    def test_professional_content(self, moderator):
        """Test: Professional content is safe"""
        text = "Let's schedule a meeting to discuss the project roadmap"
        result = moderator.moderate_sync(text)

        assert result.is_safe
        assert len(result.flagged_categories) == 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestModerationIntegration:
    """Integration tests for content moderation"""

    def test_batch_moderation(self, moderator):
        """Test: Batch content moderation"""
        texts = [
            "Safe content",
            "Normal question",
            "Professional message",
        ]

        results = [moderator.moderate_sync(text) for text in texts]

        safe_count = sum(1 for r in results if r.is_safe)
        assert safe_count >= 2

    def test_moderation_consistency(self, moderator):
        """Test: Consistent results for same input"""
        text = "Hello, how are you?"

        result1 = moderator.moderate_sync(text)
        result2 = moderator.moderate_sync(text)

        assert result1.is_safe == result2.is_safe
        assert result1.severity == result2.severity
