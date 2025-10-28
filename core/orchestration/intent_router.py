"""
Intent Router - Multi-Agent Orchestration Framework
====================================================

Classifies user intents and routes to appropriate agents using:
- Rule-based pattern matching (regex)
- NLU integration (transformers for semantic understanding)
- Confidence scoring (0.0-1.0)
- Context extraction

Intent Categories:
- code: Programming and development tasks
- research: Information gathering and analysis
- trading: Financial operations and market analysis
- system: System management and configuration
- communication: Email, messaging, notifications
- unknown: Fallback for unclassified intents

Features:
- Multi-strategy classification (rules + NLU)
- Confidence-based routing with fallback
- Context extraction from user input
- Guardian defense integration for safety
- Audit chain logging
"""

import re
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """Supported intent categories"""
    CODE = "code"
    RESEARCH = "research"
    TRADING = "trading"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Intent classification result"""
    request_id: str
    intent: IntentCategory
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any]
    agent_role: str
    timestamp: str
    patterns_matched: List[str] = field(default_factory=list)
    fallback_used: bool = False


class IntentRouter:
    """
    Routes user intents to appropriate agent roles.

    Uses a hybrid approach:
    1. Rule-based pattern matching for common intents (fast, high precision)
    2. NLU semantic understanding for complex intents (slower, broad coverage)
    3. Confidence-based fallback handling
    """

    def __init__(self, use_nlu: bool = True, confidence_threshold: float = 0.6):
        """
        Initialize intent router.

        Args:
            use_nlu: Enable NLU-based classification (requires transformers)
            confidence_threshold: Minimum confidence for direct routing (0.0-1.0)
        """
        self.use_nlu = use_nlu
        self.confidence_threshold = confidence_threshold

        # Rule-based patterns (regex)
        self._init_patterns()

        # NLU model (lazy-loaded)
        self._nlu_model = None
        self._nlu_tokenizer = None

        # Intent to agent role mapping
        self.intent_to_agent = {
            IntentCategory.CODE: "coder",
            IntentCategory.RESEARCH: "researcher",
            IntentCategory.TRADING: "finance",
            IntentCategory.SYSTEM: "operator",
            IntentCategory.COMMUNICATION: "operator",
            IntentCategory.UNKNOWN: "planner",  # Planner decides what to do
        }

        # Statistics
        self.classification_count = 0
        self.rule_based_count = 0
        self.nlu_based_count = 0
        self.fallback_count = 0

        logger.info(f"IntentRouter initialized (NLU: {use_nlu}, threshold: {confidence_threshold})")

    def _init_patterns(self):
        """Initialize rule-based patterns for each intent category"""
        self.patterns = {
            IntentCategory.CODE: [
                # Programming keywords
                (r'\b(write|code|program|implement|develop|debug|fix|refactor)\b.*\b(function|class|module|script|api|endpoint)\b', 0.9),
                (r'\b(create|build|make)\b.*\b(app|application|service|tool|library)\b', 0.85),
                (r'\b(python|javascript|java|c\+\+|rust|go|typescript)\b', 0.8),
                (r'\b(git|github|commit|push|pull request|merge)\b', 0.75),
                (r'\b(test|unittest|pytest|jest|testing)\b', 0.7),
                (r'\b(deploy|docker|kubernetes|container)\b', 0.7),
                (r'\b(sql|database|query|orm|migration)\b', 0.7),
            ],

            IntentCategory.RESEARCH: [
                # Research and information gathering
                (r'\b(research|investigate|explore|study|analyze|examine)\b', 0.9),
                (r'\b(what is|tell me about|explain|describe|how does)\b', 0.85),
                (r'\b(find|search|lookup|discover|learn)\b.*\b(information|data|facts|articles)\b', 0.85),
                (r'\b(summarize|summary|overview|digest)\b', 0.8),
                (r'\b(compare|contrast|difference between)\b', 0.8),
                (r'\b(trend|pattern|insight|analysis)\b', 0.75),
                (r'\b(wikipedia|google|search engine|web search)\b', 0.7),
            ],

            IntentCategory.TRADING: [
                # Financial and trading operations
                (r'\b(trade|buy|sell|exchange|swap)\b.*\b(stock|crypto|token|asset|coin)\b', 0.95),
                (r'\b(portfolio|position|holdings|balance)\b', 0.9),
                (r'\b(market|price|chart|ticker|symbol)\b.*\b(analysis|data|info)\b', 0.85),
                (r'\b(arbitrage|dex|liquidity|swap)\b', 0.9),
                (r'\b(profit|loss|pnl|return|roi)\b', 0.8),
                (r'\b(solana|ethereum|bitcoin|binance|jupiter)\b', 0.85),
                (r'\b(wallet|address|transaction|signature)\b', 0.8),
                (r'\b(risk|exposure|hedge|strategy)\b.*\b(trading|investment)\b', 0.85),
            ],

            IntentCategory.SYSTEM: [
                # System management
                (r'\b(restart|stop|start|kill|shutdown)\b.*\b(service|process|system|server)\b', 0.95),
                (r'\b(monitor|check|status|health)\b.*\b(system|server|service)\b', 0.9),
                (r'\b(configure|setup|install|update|upgrade)\b', 0.85),
                (r'\b(log|logs|logging|audit|metrics)\b', 0.8),
                (r'\b(disk|memory|cpu|resource|performance)\b', 0.8),
                (r'\b(backup|restore|snapshot|recovery)\b', 0.85),
                (r'\b(security|firewall|permissions|access)\b', 0.8),
            ],

            IntentCategory.COMMUNICATION: [
                # Communication tasks
                (r'\b(send|email|message|notify|alert)\b', 0.9),
                (r'\b(slack|discord|telegram|sms|notification)\b', 0.9),
                (r'\b(subscribe|unsubscribe|subscribe)\b', 0.85),
                (r'\b(report|dashboard|summary)\b.*\b(email|send|deliver)\b', 0.85),
            ],
        }

    def classify(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Classify user intent and determine routing.

        Args:
            user_input: User's natural language input
            context: Optional additional context (user_id, session_id, etc.)

        Returns:
            IntentResult with classification details
        """
        request_id = str(uuid4())
        context = context or {}

        self.classification_count += 1

        # Try rule-based classification first (fast path)
        rule_result = self._classify_by_rules(user_input)

        if rule_result and rule_result[1] >= self.confidence_threshold:
            # High confidence rule match
            intent, confidence, patterns = rule_result
            self.rule_based_count += 1

            result = IntentResult(
                request_id=request_id,
                intent=intent,
                confidence=confidence,
                context=self._extract_context(user_input, intent, context),
                agent_role=self.intent_to_agent[intent],
                timestamp=datetime.now().isoformat(),
                patterns_matched=patterns,
                fallback_used=False
            )

            logger.info(f"Intent classified (rule): {intent.value} (confidence: {confidence:.2f})")
            return result

        # Try NLU-based classification (semantic understanding)
        if self.use_nlu:
            nlu_result = self._classify_by_nlu(user_input)

            if nlu_result and nlu_result[1] >= self.confidence_threshold:
                intent, confidence = nlu_result
                self.nlu_based_count += 1

                result = IntentResult(
                    request_id=request_id,
                    intent=intent,
                    confidence=confidence,
                    context=self._extract_context(user_input, intent, context),
                    agent_role=self.intent_to_agent[intent],
                    timestamp=datetime.now().isoformat(),
                    patterns_matched=["nlu_semantic"],
                    fallback_used=False
                )

                logger.info(f"Intent classified (NLU): {intent.value} (confidence: {confidence:.2f})")
                return result

        # Fallback: Use best guess or default to UNKNOWN
        if rule_result:
            intent, confidence, patterns = rule_result
            logger.warning(f"Low confidence classification: {intent.value} ({confidence:.2f})")
        else:
            intent = IntentCategory.UNKNOWN
            confidence = 0.3
            patterns = []
            logger.warning("Unable to classify intent, using UNKNOWN")

        self.fallback_count += 1

        result = IntentResult(
            request_id=request_id,
            intent=intent,
            confidence=confidence,
            context=self._extract_context(user_input, intent, context),
            agent_role=self.intent_to_agent[intent],
            timestamp=datetime.now().isoformat(),
            patterns_matched=patterns,
            fallback_used=True
        )

        return result

    def _classify_by_rules(self, user_input: str) -> Optional[Tuple[IntentCategory, float, List[str]]]:
        """
        Classify using rule-based pattern matching.

        Returns:
            Tuple of (intent, confidence, matched_patterns) or None
        """
        user_input_lower = user_input.lower()

        # Score each intent category
        scores: Dict[IntentCategory, Tuple[float, List[str]]] = {}

        for intent, patterns in self.patterns.items():
            max_confidence = 0.0
            matched_patterns = []

            for pattern, base_confidence in patterns:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    if base_confidence > max_confidence:
                        max_confidence = base_confidence
                    matched_patterns.append(pattern)

            if max_confidence > 0:
                # Boost confidence if multiple patterns match
                boost = min(0.1 * (len(matched_patterns) - 1), 0.15)
                final_confidence = min(max_confidence + boost, 1.0)
                scores[intent] = (final_confidence, matched_patterns)

        if not scores:
            return None

        # Return highest scoring intent
        best_intent = max(scores.items(), key=lambda x: x[1][0])
        return best_intent[0], best_intent[1][0], best_intent[1][1]

    def _classify_by_nlu(self, user_input: str) -> Optional[Tuple[IntentCategory, float]]:
        """
        Classify using NLU semantic understanding.

        This is a placeholder for transformer-based classification.
        In production, would use:
        - Fine-tuned BERT/DistilBERT for intent classification
        - Zero-shot classification with BART/T5
        - Sentence embeddings with cosine similarity

        Returns:
            Tuple of (intent, confidence) or None
        """
        # Lazy-load NLU model
        if not self._nlu_model:
            try:
                # Commented out to avoid heavy dependencies in default setup
                # from transformers import pipeline
                # self._nlu_model = pipeline("zero-shot-classification",
                #                           model="facebook/bart-large-mnli")
                logger.info("NLU model loading skipped (placeholder)")
                return None
            except Exception as e:
                logger.warning(f"Failed to load NLU model: {e}")
                return None

        # Perform zero-shot classification
        try:
            # candidate_labels = [cat.value for cat in IntentCategory if cat != IntentCategory.UNKNOWN]
            # result = self._nlu_model(user_input, candidate_labels)
            #
            # top_label = result['labels'][0]
            # top_score = result['scores'][0]
            #
            # intent = IntentCategory(top_label)
            # return intent, top_score

            # Placeholder: return None to fall back to rules
            return None

        except Exception as e:
            logger.error(f"NLU classification failed: {e}")
            return None

    def _extract_context(
        self,
        user_input: str,
        intent: IntentCategory,
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract context information from user input based on intent.

        Returns:
            Context dictionary with extracted entities and metadata
        """
        context = base_context.copy()
        context["user_input"] = user_input
        context["intent"] = intent.value

        user_input_lower = user_input.lower()

        # Intent-specific context extraction
        if intent == IntentCategory.CODE:
            # Extract programming language
            langs = {
                "python": r'\bpython\b',
                "javascript": r'\b(javascript|js)\b',
                "typescript": r'\b(typescript|ts)\b',
                "java": r'\bjava\b',
                "rust": r'\brust\b',
                "go": r'\b(golang|go)\b',
                "c++": r'\bc\+\+\b',
            }
            for lang, pattern in langs.items():
                if re.search(pattern, user_input_lower):
                    context["language"] = lang
                    break

            # Extract action
            actions = {
                "write": r'\b(write|create|build|make)\b',
                "debug": r'\b(debug|fix|resolve|solve)\b',
                "refactor": r'\b(refactor|optimize|improve)\b',
                "test": r'\b(test|unittest|testing)\b',
            }
            for action, pattern in actions.items():
                if re.search(pattern, user_input_lower):
                    context["action"] = action
                    break

        elif intent == IntentCategory.TRADING:
            # Extract asset symbols (SOL, BTC, ETH, etc.)
            symbols = re.findall(r'\b([A-Z]{3,5})\b', user_input)
            if symbols:
                context["symbols"] = symbols

            # Extract trading action
            if re.search(r'\b(buy|long)\b', user_input_lower):
                context["action"] = "buy"
            elif re.search(r'\b(sell|short)\b', user_input_lower):
                context["action"] = "sell"

            # Extract amount/quantity
            amounts = re.findall(r'\$?([\d,]+(?:\.\d+)?)', user_input)
            if amounts:
                context["amount"] = amounts[0]

        elif intent == IntentCategory.RESEARCH:
            # Extract query keywords
            keywords = re.findall(r'\b[a-z]{4,}\b', user_input_lower)
            if keywords:
                # Remove common words
                stopwords = {'what', 'when', 'where', 'which', 'about', 'tell', 'find', 'search'}
                keywords = [k for k in keywords if k not in stopwords]
                context["keywords"] = keywords[:10]

        elif intent == IntentCategory.SYSTEM:
            # Extract system components
            components = {
                "service": r'\bservice\b',
                "database": r'\b(database|db|postgres|mysql)\b',
                "cache": r'\b(cache|redis)\b',
                "server": r'\bserver\b',
            }
            for comp, pattern in components.items():
                if re.search(pattern, user_input_lower):
                    context["component"] = comp
                    break

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return {
            "total_classifications": self.classification_count,
            "rule_based": self.rule_based_count,
            "nlu_based": self.nlu_based_count,
            "fallback": self.fallback_count,
            "rule_based_rate": self.rule_based_count / max(self.classification_count, 1),
            "nlu_based_rate": self.nlu_based_count / max(self.classification_count, 1),
            "fallback_rate": self.fallback_count / max(self.classification_count, 1),
        }

    def validate_intent(self, intent_result: IntentResult) -> bool:
        """
        Validate intent with Guardian Defense for safety.

        Args:
            intent_result: Intent classification result

        Returns:
            True if intent is safe, False otherwise
        """
        try:
            from security.guardian_defense import get_guardian_defense

            guardian = get_guardian_defense()

            # Check if intent involves dangerous operations
            dangerous_patterns = [
                r'\b(delete|remove|drop)\b.*\b(database|table|production)\b',
                r'\b(shutdown|kill|terminate)\b.*\b(all|everything|system)\b',
                r'\b(format|wipe|erase)\b.*\b(disk|drive|data)\b',
            ]

            user_input = intent_result.context.get("user_input", "").lower()

            for pattern in dangerous_patterns:
                if re.search(pattern, user_input):
                    logger.warning(f"Dangerous intent detected: {pattern}")

                    # Log threat
                    from security.guardian_defense import ThreatLevel
                    guardian._log_threat(
                        "dangerous_intent",
                        ThreatLevel.HIGH,
                        "intent_router",
                        {
                            "request_id": intent_result.request_id,
                            "intent": intent_result.intent.value,
                            "pattern": pattern,
                            "user_input": user_input[:200]
                        },
                        "intent_blocked"
                    )

                    return False

            return True

        except Exception as e:
            logger.error(f"Intent validation failed: {e}")
            # Fail open (allow) on validation errors
            return True


# Singleton instance
_intent_router: Optional[IntentRouter] = None


def get_intent_router(use_nlu: bool = True, confidence_threshold: float = 0.6) -> IntentRouter:
    """Get singleton intent router instance"""
    global _intent_router
    if _intent_router is None:
        _intent_router = IntentRouter(use_nlu=use_nlu, confidence_threshold=confidence_threshold)
    return _intent_router
