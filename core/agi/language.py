"""
AGI Language Module
Natural Language Understanding & Generation

Integrates with LLMs for:
- Text understanding
- Natural conversation
- Reasoning in natural language
- Multi-turn dialogue
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Chat message"""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime


class LanguageModule:
    """
    Language capabilities for AGI
    Provides natural language understanding and generation
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize language module

        Args:
            model_name: LLM model to use (gpt-4, claude-3, etc.)
            api_key: API key for LLM service
        """
        self.model_name = model_name
        self.api_key = api_key
        self.conversation_history: List[Message] = []

        # Try to import LLM libraries
        self.openai_available = False
        self.anthropic_available = False

        try:
            import openai
            self.openai = openai
            if api_key:
                self.openai.api_key = api_key
            self.openai_available = True
            logger.info("OpenAI integration available")
        except ImportError:
            logger.warning("OpenAI not installed. Install with: pip install openai")

        try:
            import anthropic
            self.anthropic = anthropic
            self.anthropic_available = True
            logger.info("Anthropic integration available")
        except ImportError:
            logger.warning("Anthropic not installed. Install with: pip install anthropic")

        logger.info(f"Language module initialized: {model_name}")

    async def understand(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Understand natural language input

        Args:
            text: Input text
            context: Optional context information

        Returns:
            Understanding result with intent, entities, sentiment
        """
        # Simple implementation for now
        # In production, would use NLU pipeline

        understanding = {
            "text": text,
            "intent": self._extract_intent(text),
            "entities": self._extract_entities(text),
            "sentiment": self._analyze_sentiment(text),
            "confidence": 0.8
        }

        return understanding

    async def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
        max_tokens: int = 500
    ) -> str:
        """
        Generate natural language response

        Args:
            prompt: Input prompt
            context: Conversation context
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if context:
            self.conversation_history = context

        # Try to use real LLM if available
        if self.openai_available and self.api_key:
            try:
                response = await self._generate_openai(prompt, max_tokens)
                return response
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")

        # Fallback to template-based generation
        return self._generate_template(prompt)

    async def _generate_openai(self, prompt: str, max_tokens: int) -> str:
        """Generate using OpenAI API"""
        messages = [
            {"role": "system", "content": "You are ShivX, an advanced AGI trading system with deep learning capabilities."},
            *[{"role": msg.role, "content": msg.content} for msg in self.conversation_history[-10:]],
            {"role": "user", "content": prompt}
        ]

        response = await self.openai.ChatCompletion.acreate(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

        return response.choices[0].message.content

    def _generate_template(self, prompt: str) -> str:
        """Template-based generation (fallback)"""
        if "what" in prompt.lower() or "how" in prompt.lower():
            return f"Based on my analysis, regarding '{prompt}', I can provide detailed insights using my multi-agent reasoning system."
        elif "should" in prompt.lower():
            return f"Analyzing the question '{prompt}' using my metacognitive capabilities, I recommend considering multiple factors."
        else:
            return f"I understand you're asking about '{prompt}'. Let me process this using my reasoning engine."

    def _extract_intent(self, text: str) -> str:
        """Extract intent from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["trade", "buy", "sell"]):
            return "trading"
        elif any(word in text_lower for word in ["analyze", "analysis", "predict"]):
            return "analysis"
        elif any(word in text_lower for word in ["explain", "why", "how"]):
            return "explanation"
        elif any(word in text_lower for word in ["strategy", "optimize"]):
            return "strategy"
        else:
            return "general"

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text"""
        entities = []

        # Simple entity extraction
        tokens = ["SOL", "USDC", "USDT", "RAY", "ORCA", "BTC", "ETH"]
        for token in tokens:
            if token in text.upper():
                entities.append({"type": "token", "value": token})

        return entities

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        positive_words = ["bullish", "good", "great", "profit", "gain", "up"]
        negative_words = ["bearish", "bad", "loss", "down", "crash"]

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    async def chat(self, message: str, user_id: str = "user") -> str:
        """
        Interactive chat interface

        Args:
            message: User message
            user_id: User identifier

        Returns:
            Assistant response
        """
        # Add user message to history
        self.conversation_history.append(
            Message(role="user", content=message, timestamp=datetime.utcnow())
        )

        # Generate response
        response = await self.generate(message)

        # Add assistant response to history
        self.conversation_history.append(
            Message(role="assistant", content=response, timestamp=datetime.utcnow())
        )

        # Trim history to last 20 messages
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def get_capabilities(self) -> Dict[str, bool]:
        """Get language capabilities status"""
        return {
            "understanding": True,
            "generation": True,
            "chat": True,
            "openai_available": self.openai_available,
            "anthropic_available": self.anthropic_available,
            "multilingual": False,  # Would add with translation API
            "voice": False  # Would add with TTS/STT
        }
