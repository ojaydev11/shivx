"""
ShivX Emotion Detection Module
Analyzes user input to detect emotions and sentiment

Features:
- Sentiment analysis (positive/negative/neutral)
- Emotion detection (joy, sadness, anger, fear, surprise)
- Frustration detection
- Confidence scoring
"""

import logging
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Emotion(Enum):
    """Emotion types"""

    NEUTRAL = "neutral"
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    CONFUSION = "confusion"


@dataclass
class EmotionResult:
    """Emotion detection result"""

    primary_emotion: Emotion
    sentiment: str  # positive, negative, neutral
    confidence: float
    emotions: Dict[str, float]  # emotion -> score
    indicators: List[str]  # detected keywords/patterns


class EmotionDetector:
    """Emotion detection using NLP"""

    def __init__(self, use_transformer: bool = False):
        """
        Initialize emotion detector

        Args:
            use_transformer: Use transformer model (more accurate but slower)
        """
        self.use_transformer = use_transformer
        self.sentiment_analyzer = None
        self._load_models()

    def _load_models(self):
        """Load sentiment analysis models"""
        try:
            if self.use_transformer:
                self._load_transformer()
            else:
                self._load_vader()
            logger.info(
                f"Emotion detector initialized (transformer={self.use_transformer})"
            )
        except Exception as e:
            logger.error(f"Failed to load emotion models: {e}")
            raise

    def _load_vader(self):
        """Load VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        except ImportError:
            logger.error(
                "vaderSentiment not installed. Install with: pip install vaderSentiment"
            )
            raise

    def _load_transformer(self):
        """Load transformer-based sentiment model"""
        try:
            from transformers import pipeline

            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
            )
            logger.info("Transformer sentiment model loaded")
        except ImportError:
            logger.error(
                "transformers not installed. Install with: pip install transformers"
            )
            raise

    def detect(self, text: str) -> EmotionResult:
        """
        Detect emotion from text

        Args:
            text: Input text

        Returns:
            EmotionResult with detected emotions
        """
        if not text or not text.strip():
            return EmotionResult(
                primary_emotion=Emotion.NEUTRAL,
                sentiment="neutral",
                confidence=1.0,
                emotions={"neutral": 1.0},
                indicators=[],
            )

        # Get sentiment
        if self.use_transformer:
            sentiment_data = self._analyze_sentiment_transformer(text)
        else:
            sentiment_data = self._analyze_sentiment_vader(text)

        # Detect specific emotions
        emotions = self._detect_emotions(text)

        # Determine primary emotion
        primary_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        confidence = emotions[primary_emotion]

        # Get sentiment label
        sentiment = sentiment_data["label"]

        # Find indicators
        indicators = self._find_indicators(text, primary_emotion)

        return EmotionResult(
            primary_emotion=Emotion(primary_emotion),
            sentiment=sentiment,
            confidence=confidence,
            emotions=emotions,
            indicators=indicators,
        )

    def _analyze_sentiment_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        scores = self.sentiment_analyzer.polarity_scores(text)

        # Determine label
        if scores["compound"] >= 0.05:
            label = "positive"
        elif scores["compound"] <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "scores": scores,
        }

    def _analyze_sentiment_transformer(self, text: str) -> Dict:
        """Analyze sentiment using transformer"""
        result = self.sentiment_analyzer(text)[0]

        # Map labels
        label_map = {"POSITIVE": "positive", "NEGATIVE": "negative"}
        label = label_map.get(result["label"], "neutral")

        return {
            "label": label,
            "score": result["score"],
        }

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detect specific emotions using keyword matching

        Returns:
            Dictionary of emotion -> score
        """
        text_lower = text.lower()

        emotions = {
            "neutral": 0.0,
            "joy": 0.0,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "frustration": 0.0,
            "excitement": 0.0,
            "confusion": 0.0,
        }

        # Joy keywords
        joy_keywords = [
            "happy",
            "great",
            "awesome",
            "wonderful",
            "excellent",
            "love",
            "amazing",
            "fantastic",
            "perfect",
            "best",
        ]
        emotions["joy"] = sum(1 for kw in joy_keywords if kw in text_lower) / len(
            joy_keywords
        )

        # Sadness keywords
        sadness_keywords = [
            "sad",
            "unhappy",
            "disappointed",
            "depressed",
            "down",
            "bad",
            "terrible",
            "awful",
            "worst",
        ]
        emotions["sadness"] = sum(
            1 for kw in sadness_keywords if kw in text_lower
        ) / len(sadness_keywords)

        # Anger keywords
        anger_keywords = [
            "angry",
            "mad",
            "furious",
            "hate",
            "annoyed",
            "irritated",
            "frustrated",
        ]
        emotions["anger"] = sum(1 for kw in anger_keywords if kw in text_lower) / len(
            anger_keywords
        )

        # Fear keywords
        fear_keywords = [
            "afraid",
            "scared",
            "worried",
            "anxious",
            "nervous",
            "concerned",
        ]
        emotions["fear"] = sum(1 for kw in fear_keywords if kw in text_lower) / len(
            fear_keywords
        )

        # Frustration keywords
        frustration_keywords = [
            "frustrated",
            "stuck",
            "confused",
            "lost",
            "don't understand",
            "help",
            "issue",
            "problem",
            "error",
            "fail",
            "not working",
        ]
        emotions["frustration"] = sum(
            1 for kw in frustration_keywords if kw in text_lower
        ) / len(frustration_keywords)

        # Excitement keywords
        excitement_keywords = [
            "excited",
            "can't wait",
            "amazing",
            "wow",
            "incredible",
            "awesome",
        ]
        emotions["excitement"] = sum(
            1 for kw in excitement_keywords if kw in text_lower
        ) / len(excitement_keywords)

        # Confusion keywords
        confusion_keywords = [
            "confused",
            "don't understand",
            "what",
            "how",
            "why",
            "?",
            "unclear",
        ]
        emotions["confusion"] = sum(
            1 for kw in confusion_keywords if kw in text_lower
        ) / len(confusion_keywords)

        # Normalize scores
        max_score = max(emotions.values())
        if max_score > 0:
            emotions = {k: v / max_score for k, v in emotions.items()}
        else:
            emotions["neutral"] = 1.0

        return emotions

    def _find_indicators(self, text: str, emotion: str) -> List[str]:
        """Find keywords that indicate the emotion"""
        indicators = []
        text_lower = text.lower()

        emotion_keywords = {
            "joy": [
                "happy",
                "great",
                "awesome",
                "wonderful",
                "excellent",
                "love",
            ],
            "sadness": ["sad", "unhappy", "disappointed", "terrible"],
            "anger": ["angry", "mad", "furious", "hate"],
            "fear": ["afraid", "scared", "worried", "anxious"],
            "frustration": [
                "frustrated",
                "stuck",
                "confused",
                "error",
                "not working",
            ],
            "excitement": ["excited", "can't wait", "amazing", "wow"],
            "confusion": ["confused", "don't understand", "unclear"],
        }

        keywords = emotion_keywords.get(emotion, [])
        for keyword in keywords:
            if keyword in text_lower:
                indicators.append(keyword)

        return indicators


# Singleton instance
_emotion_detector: Optional[EmotionDetector] = None


def get_emotion_detector(use_transformer: bool = False) -> EmotionDetector:
    """Get or create emotion detector singleton"""
    global _emotion_detector

    if _emotion_detector is None:
        _emotion_detector = EmotionDetector(use_transformer=use_transformer)

    return _emotion_detector
