"""
Data Loss Prevention (DLP) Tests
Tests for PII detection, secret detection, and data redaction
"""

import pytest
from utils.dlp import DataLossPreventionFilter, SensitiveDataType


@pytest.fixture
def dlp_engine():
    """Fixture for DLP engine"""
    return DataLossPreventionFilter(enable_logging=False)


# =============================================================================
# Test SSN Detection
# =============================================================================

@pytest.mark.security
class TestSSNDetection:
    """Test Social Security Number detection"""

    def test_detect_ssn_with_dashes(self, dlp_engine):
        """Test: Detect SSN in format XXX-XX-XXXX"""
        text = "My SSN is 123-45-6789 for verification"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.SSN for d in result.detections)

    def test_multiple_ssns(self, dlp_engine):
        """Test: Detect multiple SSNs"""
        text = "First: 123-45-6789, Second: 987-65-4321"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert result.redacted_count >= 2


# =============================================================================
# Test Email Detection
# =============================================================================

@pytest.mark.security
class TestEmailDetection:
    """Test email address detection"""

    def test_detect_standard_email(self, dlp_engine):
        """Test: Detect standard email"""
        text = "Contact me at john.doe@example.com"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.EMAIL for d in result.detections)

    def test_multiple_emails(self, dlp_engine):
        """Test: Detect multiple emails"""
        text = "Send to alice@test.com and bob@test.com"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert result.redacted_count >= 2


# =============================================================================
# Test Phone Number Detection
# =============================================================================

@pytest.mark.security
class TestPhoneDetection:
    """Test phone number detection"""

    def test_detect_phone_with_dashes(self, dlp_engine):
        """Test: Detect phone with dashes"""
        text = "Call me at 555-123-4567"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.PHONE for d in result.detections)


# =============================================================================
# Test Credit Card Detection (with Luhn validation)
# =============================================================================

@pytest.mark.security
class TestCreditCardDetection:
    """Test credit card number detection with Luhn algorithm"""

    def test_detect_valid_credit_card(self, dlp_engine):
        """Test: Detect valid credit card (passes Luhn check)"""
        # Valid credit card number (passes Luhn)
        text = "Card: 4532 1488 0343 6467"  # Valid test card
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.CREDIT_CARD for d in result.detections)

    def test_skip_invalid_credit_card(self, dlp_engine):
        """Test: Skip invalid credit card (fails Luhn check)"""
        # Invalid credit card number (fails Luhn)
        text = "Number: 1234-5678-9012-3456"  # Invalid
        result = dlp_engine.scan(text)

        # Should not detect as credit card due to Luhn validation
        credit_card_detections = [d for d in result.detections if d[0] == SensitiveDataType.CREDIT_CARD]
        assert len(credit_card_detections) == 0


# =============================================================================
# Test API Key Detection
# =============================================================================

@pytest.mark.security
class TestAPIKeyDetection:
    """Test API key and secret detection"""

    def test_detect_aws_access_key(self, dlp_engine):
        """Test: Detect AWS access key"""
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.AWS_KEY for d in result.detections)

    def test_detect_github_token(self, dlp_engine):
        """Test: Detect GitHub personal access token"""
        text = "GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwx"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.GITHUB_TOKEN for d in result.detections)

    def test_detect_jwt_token(self, dlp_engine):
        """Test: Detect JWT token"""
        text = "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert any(d[0] == SensitiveDataType.JWT_TOKEN for d in result.detections)


# =============================================================================
# Test Redaction Functionality
# =============================================================================

@pytest.mark.security
class TestRedaction:
    """Test PII/secret redaction"""

    def test_redact_ssn(self, dlp_engine):
        """Test: Redact SSN from text"""
        text = "My SSN is 123-45-6789"
        result = dlp_engine.scan(text)

        assert "123-45-6789" not in result.redacted_text
        assert "[REDACTED-" in result.redacted_text
        assert "My SSN is" in result.redacted_text

    def test_redact_email(self, dlp_engine):
        """Test: Redact email from text"""
        text = "Contact: john@example.com"
        result = dlp_engine.scan(text)

        assert "john@example.com" not in result.redacted_text
        assert "Contact:" in result.redacted_text

    def test_redact_multiple_pii(self, dlp_engine):
        """Test: Redact multiple PII items"""
        text = "Email: test@test.com, SSN: 123-45-6789"
        result = dlp_engine.scan(text)

        assert "test@test.com" not in result.redacted_text
        assert "123-45-6789" not in result.redacted_text
        assert result.redacted_count >= 2


# =============================================================================
# Test Convenience Functions
# =============================================================================

@pytest.mark.security
class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_contains_pii(self, dlp_engine):
        """Test: Check if text contains PII"""
        text_with_pii = "My email is john@example.com"
        text_without_pii = "Hello world"

        assert dlp_engine.contains_pii(text_with_pii)
        assert not dlp_engine.contains_pii(text_without_pii)

    def test_contains_secrets(self, dlp_engine):
        """Test: Check if text contains secrets"""
        text_with_secret = "API key: ghp_1234567890abcdefghijklmnopqrstuvwx"
        text_without_secret = "Hello world"

        assert dlp_engine.contains_secrets(text_with_secret)
        assert not dlp_engine.contains_secrets(text_without_secret)


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestDLPIntegration:
    """Integration tests for full DLP workflow"""

    def test_full_dlp_workflow(self, dlp_engine):
        """Test: Complete DLP workflow"""
        text = "Email: admin@company.com, SSN: 123-45-6789, Token: ghp_abcd1234567890abcdefghijklmnopqr"

        # Scan
        result = dlp_engine.scan(text)

        assert result.found_sensitive_data
        assert result.redacted_count >= 3

        # Verify redaction
        assert "admin@company.com" not in result.redacted_text
        assert "123-45-6789" not in result.redacted_text
        assert "ghp_abcd1234567890abcdefghijklmnopqr" not in result.redacted_text
