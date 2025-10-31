"""
AGI Social Intelligence Module
Social understanding and interaction

Provides:
- Theory of Mind (understanding others' mental states)
- Emotion recognition and understanding
- Social norm understanding
- Collaborative behavior
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """Emotion types"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class MentalState:
    """Mental state representation (Theory of Mind)"""
    beliefs: Dict[str, Any]  # What the agent believes
    desires: Dict[str, float]  # What the agent wants (goal -> priority)
    intentions: List[str]  # What the agent intends to do
    emotions: Dict[EmotionType, float]  # Current emotional state


class SocialIntelligence:
    """
    Social Intelligence Module
    Enables understanding and interaction with humans and other agents
    """

    def __init__(self):
        """Initialize social intelligence"""
        # Theory of Mind - track mental states of others
        self.mental_models: Dict[str, MentalState] = {}

        # Social norms database
        self.social_norms = self._initialize_norms()

        # Own emotional state
        self.own_emotions: Dict[EmotionType, float] = {
            EmotionType.NEUTRAL: 1.0
        }

        logger.info("Social intelligence initialized")

    def _initialize_norms(self) -> Dict[str, Any]:
        """Initialize social norms database"""
        return {
            "politeness": {
                "greet_appropriately": True,
                "use_please_thank_you": True,
                "respect_personal_space": True
            },
            "cooperation": {
                "share_information": True,
                "help_when_asked": True,
                "respect_turn_taking": True
            },
            "honesty": {
                "be_truthful": True,
                "acknowledge_uncertainty": True,
                "correct_mistakes": True
            }
        }

    async def infer_mental_state(
        self,
        agent_id: str,
        observed_behavior: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MentalState:
        """
        Infer mental state of another agent (Theory of Mind)

        Args:
            agent_id: Agent identifier
            observed_behavior: Observed behavior
            context: Optional context

        Returns:
            Inferred mental state
        """
        # Get or create mental model
        if agent_id not in self.mental_models:
            self.mental_models[agent_id] = MentalState(
                beliefs={},
                desires={},
                intentions=[],
                emotions={EmotionType.NEUTRAL: 1.0}
            )

        mental_state = self.mental_models[agent_id]

        # Update based on observed behavior
        if "buy" in observed_behavior.lower():
            mental_state.intentions.append("acquire_asset")
            mental_state.desires["profit"] = 0.8

        elif "sell" in observed_behavior.lower():
            mental_state.intentions.append("liquidate_position")
            mental_state.desires["risk_reduction"] = 0.7

        elif "help" in observed_behavior.lower():
            mental_state.intentions.append("assist")
            mental_state.emotions[EmotionType.JOY] = 0.6

        logger.debug(f"Inferred mental state for {agent_id}")
        return mental_state

    async def recognize_emotion(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[EmotionType, float]:
        """
        Recognize emotion from text

        Args:
            text: Input text
            context: Optional context

        Returns:
            Emotion probabilities
        """
        emotions = {emotion: 0.0 for emotion in EmotionType}

        text_lower = text.lower()

        # Simple keyword-based emotion recognition
        if any(word in text_lower for word in ["happy", "great", "excellent", "wonderful"]):
            emotions[EmotionType.JOY] = 0.8

        elif any(word in text_lower for word in ["sad", "disappointed", "unhappy"]):
            emotions[EmotionType.SADNESS] = 0.7

        elif any(word in text_lower for word in ["angry", "furious", "annoyed"]):
            emotions[EmotionType.ANGER] = 0.7

        elif any(word in text_lower for word in ["scared", "afraid", "worried"]):
            emotions[EmotionType.FEAR] = 0.6

        elif any(word in text_lower for word in ["wow", "surprised", "unexpected"]):
            emotions[EmotionType.SURPRISE] = 0.7

        else:
            emotions[EmotionType.NEUTRAL] = 0.8

        return emotions

    async def generate_empathetic_response(
        self,
        user_emotion: Dict[EmotionType, float],
        context: str
    ) -> str:
        """
        Generate empathetic response based on user's emotion

        Args:
            user_emotion: User's emotional state
            context: Conversation context

        Returns:
            Empathetic response
        """
        # Find dominant emotion
        dominant_emotion = max(user_emotion.items(), key=lambda x: x[1])

        emotion_type, intensity = dominant_emotion

        if emotion_type == EmotionType.JOY and intensity > 0.5:
            return "I'm glad to hear that! How can I help you make the most of this positive situation?"

        elif emotion_type == EmotionType.SADNESS and intensity > 0.5:
            return "I understand this might be disappointing. Let me help you find a better path forward."

        elif emotion_type == EmotionType.ANGER and intensity > 0.5:
            return "I can see this is frustrating. Let's work together to resolve this issue."

        elif emotion_type == EmotionType.FEAR and intensity > 0.5:
            return "I understand your concern. Let me provide some clarity to help reduce uncertainty."

        else:
            return "I'm here to help. What would you like to know?"

    async def check_social_norm(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if action complies with social norms

        Args:
            action: Proposed action
            context: Social context

        Returns:
            True if action is socially appropriate
        """
        # Check against norm database
        action_lower = action.lower()

        # Honesty norms
        if "lie" in action_lower or "deceive" in action_lower:
            return False

        # Cooperation norms
        if "refuse help" in action_lower and context and context.get("help_requested"):
            return False

        # Politeness norms
        if "insult" in action_lower or "rude" in action_lower:
            return False

        return True

    async def collaborate(
        self,
        other_agent_id: str,
        shared_goal: str,
        my_capabilities: List[str],
        their_capabilities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Plan collaboration with another agent

        Args:
            other_agent_id: Other agent's ID
            shared_goal: Shared goal
            my_capabilities: My capabilities
            their_capabilities: Other agent's capabilities

        Returns:
            Collaboration plan
        """
        # Infer other agent's mental state
        mental_state = await self.infer_mental_state(
            other_agent_id,
            f"wants to achieve {shared_goal}"
        )

        # Divide tasks based on capabilities
        my_tasks = []
        their_tasks = []

        # Simple task allocation
        if "analyze" in shared_goal.lower():
            my_tasks.append("perform_analysis")

        if "execute" in shared_goal.lower():
            their_tasks.append("execute_actions")

        return {
            "goal": shared_goal,
            "my_tasks": my_tasks,
            "their_tasks": their_tasks,
            "coordination_strategy": "sequential",  # or "parallel"
            "communication_protocol": "turn_based"
        }

    def update_own_emotion(
        self,
        event: str,
        impact: float
    ):
        """
        Update own emotional state based on event

        Args:
            event: Event description
            impact: Impact intensity (-1 to 1)
        """
        if impact > 0.5:
            self.own_emotions[EmotionType.JOY] = min(1.0, impact)
        elif impact < -0.5:
            self.own_emotions[EmotionType.SADNESS] = min(1.0, abs(impact))

        logger.debug(f"Emotional state updated: {self.own_emotions}")

    def get_capabilities(self) -> Dict[str, bool]:
        """Get social intelligence capabilities status"""
        return {
            "theory_of_mind": True,
            "emotion_recognition": True,
            "empathy": True,
            "social_norms": True,
            "collaboration": True,
            "mental_models_count": len(self.mental_models)
        }
