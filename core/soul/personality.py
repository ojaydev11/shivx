"""
ShivX Personality Engine
Adjusts AI responses based on user emotion and personality settings

Features:
- Multiple personality modes (professional, friendly, playful, empathetic)
- Emotion-aware response adjustment
- Context-aware tone adaptation
- Conversation history tracking
"""

import logging
from typing import Optional, List, Dict
from enum import Enum
from dataclasses import dataclass

from core.soul.emotion import Emotion, EmotionResult, EmotionDetector

logger = logging.getLogger(__name__)


class PersonalityType(Enum):
    """Available personality types"""

    PROFESSIONAL = "professional"  # Formal, concise, business-like
    FRIENDLY = "friendly"  # Warm, conversational, approachable
    PLAYFUL = "playful"  # Humorous, creative, casual
    EMPATHETIC = "empathetic"  # Supportive, understanding, compassionate
    MENTOR = "mentor"  # Educational, guiding, patient
    CONCISE = "concise"  # Brief, to-the-point, minimal fluff


@dataclass
class ConversationContext:
    """Track conversation context"""

    user_id: str
    recent_emotions: List[str]  # Last 5 emotions
    frustration_count: int = 0  # Track consecutive frustration
    interaction_count: int = 0
    preferred_personality: PersonalityType = PersonalityType.PROFESSIONAL


class PersonalityEngine:
    """
    Personality engine for adaptive AI responses
    Adjusts tone and content based on user emotion and personality settings
    """

    def __init__(
        self,
        personality: PersonalityType = PersonalityType.PROFESSIONAL,
        emotion_detector: Optional[EmotionDetector] = None,
    ):
        """
        Initialize personality engine

        Args:
            personality: Default personality type
            emotion_detector: Emotion detector instance (optional)
        """
        self.personality = personality
        self.emotion_detector = emotion_detector
        self.context_cache: Dict[str, ConversationContext] = {}

        # Personality templates
        self.templates = self._load_templates()

        logger.info(f"PersonalityEngine initialized with personality: {personality.value}")

    def _load_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load response templates for each personality and emotion"""
        return {
            PersonalityType.PROFESSIONAL.value: {
                "greeting": ["Hello", "Good day", "Welcome"],
                "frustration": [
                    "I understand this is challenging. Let me help you resolve this.",
                    "I see you're experiencing difficulties. Let's work through this together.",
                    "I apologize for the inconvenience. I'm here to assist you.",
                ],
                "confusion": [
                    "Let me clarify that for you.",
                    "I'll explain this more clearly.",
                    "Let me break this down step by step.",
                ],
                "joy": [
                    "Excellent! I'm glad to help.",
                    "Great to hear!",
                    "Wonderful! Let's continue.",
                ],
                "neutral": [
                    "I understand.",
                    "Let me assist you with that.",
                    "Certainly.",
                ],
            },
            PersonalityType.FRIENDLY.value: {
                "greeting": ["Hey there!", "Hi! How can I help?", "Hello friend!"],
                "frustration": [
                    "Oh no, I can see this is frustrating! Don't worry, we'll figure it out together.",
                    "I totally get why this would be annoying. Let me help you out!",
                    "Ugh, I know right? Let's fix this!",
                ],
                "confusion": [
                    "No worries, let me explain that better!",
                    "Oops, I should have been clearer. Here's what I mean...",
                    "Let me walk you through this step by step!",
                ],
                "joy": [
                    "Awesome! So glad I could help!",
                    "Yay! That's fantastic!",
                    "That's great news! Happy to assist!",
                ],
                "neutral": [
                    "Sure thing!",
                    "Got it! Let me help with that.",
                    "Happy to help!",
                ],
            },
            PersonalityType.PLAYFUL.value: {
                "greeting": [
                    "Hey rockstar! Ready to trade?",
                    "What's up! Let's make some magic happen!",
                    "Hiya! Ready for action?",
                ],
                "frustration": [
                    "Whoa, let's turn that frown upside down! I've got this.",
                    "No sweat! Even the best traders hit snags. Let's crush this!",
                    "Plot twist: we're about to solve this problem! Hang tight.",
                ],
                "confusion": [
                    "Ah, the plot thickens! Let me shed some light on this mystery.",
                    "Confusing, right? Let's unravel this puzzle together!",
                    "Think of it like this...",
                ],
                "joy": [
                    "Woohoo! We're on fire!",
                    "Yes! Victory dance time!",
                    "Boom! That's what I'm talking about!",
                ],
                "neutral": [
                    "Roger that!",
                    "On it like sonic!",
                    "You got it, boss!",
                ],
            },
            PersonalityType.EMPATHETIC.value: {
                "greeting": [
                    "Hello, I'm here to support you.",
                    "Hi there, how are you doing today?",
                    "Welcome, I'm here to listen and help.",
                ],
                "frustration": [
                    "I hear you, and I understand your frustration. Let's work through this together, one step at a time.",
                    "This must be really frustrating for you. I'm here to help make this easier.",
                    "I can sense your frustration, and that's completely valid. Let me help you.",
                ],
                "confusion": [
                    "I can see this might be confusing. Let me explain it in a way that makes more sense.",
                    "That's a great question. Let me break it down for you.",
                    "I understand your confusion. Here's another way to look at it...",
                ],
                "joy": [
                    "I'm so happy for you! This is wonderful!",
                    "Your enthusiasm is contagious! I'm glad I could help.",
                    "That's fantastic! I'm thrilled to see your success.",
                ],
                "neutral": [
                    "I'm here to help you.",
                    "I understand what you need.",
                    "Let me support you with that.",
                ],
            },
            PersonalityType.MENTOR.value: {
                "greeting": [
                    "Welcome! Ready to learn and grow?",
                    "Hello! I'm here to guide you.",
                    "Great to see you! Let's learn together.",
                ],
                "frustration": [
                    "I know this is challenging, but challenges are how we grow. Let's approach this methodically.",
                    "Every expert was once a beginner who faced frustrations. Let me guide you through this.",
                    "This is a common stumbling block. Here's how we'll overcome it...",
                ],
                "confusion": [
                    "Excellent question! Let me teach you the concept behind this.",
                    "This is a great learning opportunity. Here's the key principle...",
                    "Let me explain the 'why' behind this, not just the 'how'.",
                ],
                "joy": [
                    "Well done! You're making excellent progress.",
                    "That's the spirit! You're learning quickly.",
                    "Great work! You've mastered this concept.",
                ],
                "neutral": [
                    "Let me guide you through this.",
                    "Here's what you need to know...",
                    "Let's explore this together.",
                ],
            },
            PersonalityType.CONCISE.value: {
                "greeting": ["Hi.", "Hello.", "Ready."],
                "frustration": ["Understood. Fixing now.", "On it.", "Will resolve."],
                "confusion": ["Clarifying:", "Simply:", "Here's how:"],
                "joy": ["Great.", "Excellent.", "Done."],
                "neutral": ["OK.", "Understood.", "Proceeding."],
            },
        }

    def get_context(self, user_id: str) -> ConversationContext:
        """Get or create conversation context for user"""
        if user_id not in self.context_cache:
            self.context_cache[user_id] = ConversationContext(
                user_id=user_id,
                recent_emotions=[],
                preferred_personality=self.personality,
            )
        return self.context_cache[user_id]

    def update_context(
        self, user_id: str, emotion: str, frustration: bool = False
    ):
        """Update conversation context with new emotion"""
        context = self.get_context(user_id)
        context.recent_emotions.append(emotion)
        context.recent_emotions = context.recent_emotions[-5:]  # Keep last 5
        context.interaction_count += 1

        if frustration:
            context.frustration_count += 1
        else:
            context.frustration_count = max(0, context.frustration_count - 1)

    def adjust_response(
        self,
        response: str,
        user_emotion: Optional[EmotionResult] = None,
        user_text: Optional[str] = None,
        user_id: str = "default",
    ) -> str:
        """
        Adjust response based on user emotion and personality

        Args:
            response: Original AI response
            user_emotion: Detected emotion result (optional)
            user_text: User's input text (optional, for emotion detection)
            user_id: User identifier

        Returns:
            Adjusted response with appropriate tone
        """
        # Detect emotion if not provided
        if user_emotion is None and user_text and self.emotion_detector:
            user_emotion = self.emotion_detector.detect(user_text)

        if user_emotion is None:
            return response  # No adjustment if no emotion info

        # Get context
        context = self.get_context(user_id)

        # Update context
        emotion_str = user_emotion.primary_emotion.value
        is_frustrated = emotion_str in ["frustration", "anger"]
        self.update_context(user_id, emotion_str, is_frustrated)

        # Get personality templates
        personality_key = context.preferred_personality.value
        templates = self.templates.get(personality_key, {})

        # Select appropriate prefix based on emotion
        prefix = ""
        if emotion_str in templates:
            import random

            prefix = random.choice(templates[emotion_str]) + " "

        # Add empathy for consecutive frustration
        if context.frustration_count >= 3:
            prefix = "I really want to help you succeed. " + prefix

        # Adjust response length based on personality
        if context.preferred_personality == PersonalityType.CONCISE:
            # Make response more concise
            response = response.split(".")[0] + "."  # First sentence only

        # Add context-aware adjustments
        adjusted_response = prefix + response

        logger.debug(
            f"Response adjusted: emotion={emotion_str}, personality={personality_key}"
        )

        return adjusted_response

    def set_personality(self, personality: PersonalityType, user_id: str = "default"):
        """
        Set personality for a specific user

        Args:
            personality: Personality type to set
            user_id: User identifier
        """
        context = self.get_context(user_id)
        context.preferred_personality = personality
        logger.info(f"Personality set to {personality.value} for user {user_id}")

    def get_greeting(self, user_id: str = "default") -> str:
        """
        Get personality-appropriate greeting

        Args:
            user_id: User identifier

        Returns:
            Greeting message
        """
        context = self.get_context(user_id)
        personality_key = context.preferred_personality.value
        templates = self.templates.get(personality_key, {})
        greetings = templates.get("greeting", ["Hello"])

        import random

        return random.choice(greetings)

    def reset_context(self, user_id: str):
        """Reset conversation context for user"""
        if user_id in self.context_cache:
            del self.context_cache[user_id]
            logger.info(f"Context reset for user {user_id}")


# Singleton instance
_personality_engine: Optional[PersonalityEngine] = None


def get_personality_engine(
    personality: PersonalityType = PersonalityType.PROFESSIONAL,
) -> PersonalityEngine:
    """Get or create personality engine singleton"""
    global _personality_engine

    if _personality_engine is None:
        from core.soul.emotion import get_emotion_detector

        emotion_detector = get_emotion_detector()
        _personality_engine = PersonalityEngine(
            personality=personality, emotion_detector=emotion_detector
        )

    return _personality_engine
