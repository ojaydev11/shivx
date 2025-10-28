"""
Tests for Emotion Detection and Personality Engine
"""

import pytest
from unittest.mock import Mock, patch

from core.soul.emotion import (
    EmotionDetector,
    Emotion,
    EmotionResult,
    get_emotion_detector,
)
from core.soul.personality import (
    PersonalityEngine,
    PersonalityType,
    ConversationContext,
    get_personality_engine,
)


class TestEmotionDetector:
    """Test emotion detection"""

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_init_vader(self, mock_analyzer):
        """Test initialization with VADER"""
        detector = EmotionDetector(use_transformer=False)

        assert detector.use_transformer is False
        assert detector.sentiment_analyzer is not None

    def test_detect_empty_text(self):
        """Test emotion detection with empty text"""
        with patch("core.soul.emotion.SentimentIntensityAnalyzer"):
            detector = EmotionDetector(use_transformer=False)
            result = detector.detect("")

        assert result.primary_emotion == Emotion.NEUTRAL
        assert result.confidence == 1.0

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_detect_joy(self, mock_analyzer_class):
        """Test joy emotion detection"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": 0.8,
            "pos": 0.8,
            "neg": 0.0,
            "neu": 0.2,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("I am so happy and excited!")

        assert result.primary_emotion in [Emotion.JOY, Emotion.EXCITEMENT]
        assert result.sentiment == "positive"
        assert result.confidence > 0

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_detect_frustration(self, mock_analyzer_class):
        """Test frustration emotion detection"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": -0.5,
            "pos": 0.0,
            "neg": 0.6,
            "neu": 0.4,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("This is not working and I'm frustrated")

        assert result.primary_emotion == Emotion.FRUSTRATION
        assert result.sentiment == "negative"
        assert "frustrated" in result.indicators

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_detect_confusion(self, mock_analyzer_class):
        """Test confusion emotion detection"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "neu": 1.0,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("I don't understand how this works?")

        assert result.primary_emotion == Emotion.CONFUSION
        assert "confused" in result.indicators or "don't understand" in result.indicators

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_sentiment_positive(self, mock_analyzer_class):
        """Test positive sentiment"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": 0.6,
            "pos": 0.7,
            "neg": 0.0,
            "neu": 0.3,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("This is great!")

        assert result.sentiment == "positive"

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_sentiment_negative(self, mock_analyzer_class):
        """Test negative sentiment"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": -0.6,
            "pos": 0.0,
            "neg": 0.7,
            "neu": 0.3,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("This is terrible!")

        assert result.sentiment == "negative"

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_emotion_indicators(self, mock_analyzer_class):
        """Test emotion indicator detection"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": -0.5,
            "pos": 0.0,
            "neg": 0.6,
            "neu": 0.4,
        }
        mock_analyzer_class.return_value = mock_analyzer

        detector = EmotionDetector(use_transformer=False)
        result = detector.detect("I'm angry and frustrated with this error")

        assert len(result.indicators) > 0
        assert any(
            keyword in result.indicators
            for keyword in ["angry", "frustrated", "error"]
        )


class TestPersonalityEngine:
    """Test personality engine"""

    def test_init(self):
        """Test personality engine initialization"""
        engine = PersonalityEngine(personality=PersonalityType.PROFESSIONAL)

        assert engine.personality == PersonalityType.PROFESSIONAL
        assert len(engine.templates) > 0

    def test_get_context(self):
        """Test getting conversation context"""
        engine = PersonalityEngine()
        context = engine.get_context("user123")

        assert context.user_id == "user123"
        assert context.interaction_count == 0
        assert context.frustration_count == 0

    def test_update_context(self):
        """Test updating conversation context"""
        engine = PersonalityEngine()
        engine.update_context("user123", "joy", frustration=False)

        context = engine.get_context("user123")
        assert "joy" in context.recent_emotions
        assert context.interaction_count == 1
        assert context.frustration_count == 0

    def test_frustration_tracking(self):
        """Test frustration count tracking"""
        engine = PersonalityEngine()

        # Increase frustration
        for _ in range(3):
            engine.update_context("user123", "frustration", frustration=True)

        context = engine.get_context("user123")
        assert context.frustration_count == 3

        # Decrease frustration
        engine.update_context("user123", "joy", frustration=False)
        assert context.frustration_count == 2

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_adjust_response_professional(self, mock_analyzer_class):
        """Test response adjustment with professional personality"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": -0.5,
            "pos": 0.0,
            "neg": 0.6,
            "neu": 0.4,
        }
        mock_analyzer_class.return_value = mock_analyzer

        emotion_detector = EmotionDetector(use_transformer=False)
        engine = PersonalityEngine(
            personality=PersonalityType.PROFESSIONAL,
            emotion_detector=emotion_detector,
        )

        original = "Here is your result."
        adjusted = engine.adjust_response(
            original, user_text="I'm frustrated with this error"
        )

        assert len(adjusted) > len(original)
        assert original in adjusted

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_adjust_response_friendly(self, mock_analyzer_class):
        """Test response adjustment with friendly personality"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": 0.8,
            "pos": 0.8,
            "neg": 0.0,
            "neu": 0.2,
        }
        mock_analyzer_class.return_value = mock_analyzer

        emotion_detector = EmotionDetector(use_transformer=False)
        engine = PersonalityEngine(
            personality=PersonalityType.FRIENDLY, emotion_detector=emotion_detector
        )

        original = "Task completed."
        adjusted = engine.adjust_response(original, user_text="This is great!")

        assert len(adjusted) > len(original)

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_adjust_response_playful(self, mock_analyzer_class):
        """Test response adjustment with playful personality"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": 0.8,
            "pos": 0.8,
            "neg": 0.0,
            "neu": 0.2,
        }
        mock_analyzer_class.return_value = mock_analyzer

        emotion_detector = EmotionDetector(use_transformer=False)
        engine = PersonalityEngine(
            personality=PersonalityType.PLAYFUL, emotion_detector=emotion_detector
        )

        original = "Done!"
        adjusted = engine.adjust_response(original, user_text="Awesome!")

        assert len(adjusted) >= len(original)

    def test_adjust_response_concise(self):
        """Test response adjustment with concise personality"""
        engine = PersonalityEngine(personality=PersonalityType.CONCISE)

        original = "This is a long response with multiple sentences. Here is more detail. And even more."
        adjusted = engine.adjust_response(original, user_id="user123")

        # Concise personality should shorten response
        assert len(adjusted) <= len(original)

    def test_consecutive_frustration_handling(self):
        """Test handling of consecutive frustration"""
        engine = PersonalityEngine()

        # Simulate 3+ consecutive frustrations
        for _ in range(4):
            engine.update_context("user123", "frustration", frustration=True)

        adjusted = engine.adjust_response(
            "Result ready.", user_text="Still not working!"
        )

        # Should have empathetic prefix
        assert "help you succeed" in adjusted.lower() or len(adjusted) > len(
            "Result ready."
        )

    def test_set_personality(self):
        """Test changing personality"""
        engine = PersonalityEngine(personality=PersonalityType.PROFESSIONAL)
        engine.set_personality(PersonalityType.FRIENDLY, user_id="user123")

        context = engine.get_context("user123")
        assert context.preferred_personality == PersonalityType.FRIENDLY

    def test_get_greeting(self):
        """Test getting personality-appropriate greeting"""
        engine = PersonalityEngine(personality=PersonalityType.PROFESSIONAL)
        greeting = engine.get_greeting()

        assert len(greeting) > 0
        assert isinstance(greeting, str)

    def test_reset_context(self):
        """Test resetting conversation context"""
        engine = PersonalityEngine()
        engine.update_context("user123", "joy")

        engine.reset_context("user123")

        # Context should be cleared
        assert "user123" not in engine.context_cache


class TestIntegration:
    """Integration tests for emotion detection and personality"""

    @patch("core.soul.emotion.SentimentIntensityAnalyzer")
    def test_emotion_to_personality_pipeline(self, mock_analyzer_class):
        """Test full pipeline from emotion detection to personality adjustment"""
        mock_analyzer = Mock()
        mock_analyzer.polarity_scores.return_value = {
            "compound": -0.6,
            "pos": 0.0,
            "neg": 0.7,
            "neu": 0.3,
        }
        mock_analyzer_class.return_value = mock_analyzer

        # Detect emotion
        detector = EmotionDetector(use_transformer=False)
        emotion = detector.detect("I'm frustrated with these errors")

        # Adjust response based on emotion
        engine = PersonalityEngine(
            personality=PersonalityType.EMPATHETIC, emotion_detector=detector
        )
        adjusted = engine.adjust_response(
            "Error has been logged.", user_emotion=emotion, user_id="user123"
        )

        assert emotion.primary_emotion == Emotion.FRUSTRATION
        assert len(adjusted) > len("Error has been logged.")
        assert adjusted.startswith(
            ("I hear you", "I can sense", "This must be", "I understand")
        )
