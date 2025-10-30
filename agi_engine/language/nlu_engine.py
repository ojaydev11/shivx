"""
Natural Language Understanding Engine

Provides comprehensive NLU capabilities:
- Intent recognition
- Entity extraction
- Context understanding
- Semantic parsing
- Slot filling
- Sentiment analysis
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from collections import defaultdict


class IntentType(str, Enum):
    """Types of user intents"""
    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONFIRMATION = "confirmation"
    NEGATION = "negation"
    REQUEST = "request"
    INFORM = "inform"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Types of entities"""
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    DATE = "date"
    TIME = "time"
    NUMBER = "number"
    MONEY = "money"
    PERCENTAGE = "percentage"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"


class SentimentPolarity(str, Enum):
    """Sentiment polarities"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class Entity:
    """An extracted entity"""
    entity_type: EntityType
    value: str
    start_idx: int
    end_idx: int
    confidence: float = 1.0
    normalized_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Intent:
    """A recognized intent"""
    intent_type: IntentType
    confidence: float
    domain: Optional[str] = None
    action: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticFrame:
    """Complete semantic representation of an utterance"""
    text: str
    intent: Intent
    entities: List[Entity] = field(default_factory=list)
    sentiment: SentimentPolarity = SentimentPolarity.NEUTRAL
    sentiment_score: float = 0.0  # -1 to 1
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    context_references: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class NLUEngine:
    """
    Natural Language Understanding Engine

    Provides:
    - Intent classification
    - Entity extraction and recognition
    - Semantic parsing
    - Sentiment analysis
    - Context understanding
    - Slot filling
    """

    def __init__(self):
        # Intent patterns (in real system, use ML models)
        self.intent_patterns = self._build_intent_patterns()

        # Entity patterns
        self.entity_patterns = self._build_entity_patterns()

        # Sentiment lexicon
        self.sentiment_lexicon = self._build_sentiment_lexicon()

        # Domain knowledge
        self.domains = {
            "general": ["help", "info", "about"],
            "task": ["create", "build", "make", "do"],
            "query": ["what", "why", "how", "when", "where", "who"],
            "system": ["start", "stop", "reset", "configure"],
            "social": ["hello", "hi", "bye", "thanks", "sorry"],
        }

        # Context tracking
        self.context_history: List[SemanticFrame] = []
        self.max_context_length = 10

    def _build_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Build intent recognition patterns"""
        return {
            IntentType.QUESTION: [
                r"^(what|why|how|when|where|who|which|whose|whom)",
                r"(\?|can you|could you|would you|do you|is it|are there)",
            ],
            IntentType.COMMAND: [
                r"^(create|build|make|generate|develop|implement|write|design)",
                r"^(start|stop|run|execute|launch|terminate|kill)",
                r"^(configure|set|update|change|modify)",
            ],
            IntentType.STATEMENT: [
                r"^(i think|i believe|in my opinion|it seems|apparently)",
                r"(is|are|was|were) (a|an|the)",
            ],
            IntentType.GREETING: [
                r"^(hello|hi|hey|greetings|good (morning|afternoon|evening))",
            ],
            IntentType.FAREWELL: [
                r"^(bye|goodbye|farewell|see you|talk to you later)",
            ],
            IntentType.CONFIRMATION: [
                r"^(yes|yeah|yep|sure|okay|ok|correct|right|exactly|agreed)",
            ],
            IntentType.NEGATION: [
                r"^(no|nope|nah|not really|i don't think so|negative)",
            ],
            IntentType.REQUEST: [
                r"^(please|can you|could you|would you|i need|i want|i'd like)",
            ],
            IntentType.INFORM: [
                r"^(i have|i am|i'm|there is|there are|here is)",
            ],
            IntentType.CLARIFICATION: [
                r"^(what do you mean|i don't understand|can you explain|clarify)",
            ],
        }

    def _build_entity_patterns(self) -> Dict[EntityType, List[str]]:
        """Build entity extraction patterns"""
        return {
            EntityType.NUMBER: [
                r"\b(\d+\.?\d*)\b",
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b",
            ],
            EntityType.DATE: [
                r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b",
                r"\b(today|tomorrow|yesterday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            ],
            EntityType.TIME: [
                r"\b(\d{1,2}:\d{2}(:\d{2})?( ?[ap]m)?)\b",
                r"\b(morning|afternoon|evening|night|midnight|noon)\b",
            ],
            EntityType.MONEY: [
                r"\$\s*\d+\.?\d*",
                r"\b\d+\.?\d* ?(dollars|usd|euros|eur|pounds|gbp)\b",
            ],
            EntityType.PERCENTAGE: [
                r"\b\d+\.?\d*%",
                r"\b\d+\.?\d* percent\b",
            ],
        }

    def _build_sentiment_lexicon(self) -> Dict[str, float]:
        """Build sentiment analysis lexicon"""
        return {
            # Positive words
            "good": 0.5, "great": 0.7, "excellent": 0.9, "amazing": 0.9,
            "wonderful": 0.8, "fantastic": 0.8, "love": 0.7, "like": 0.4,
            "happy": 0.6, "pleased": 0.5, "satisfied": 0.5, "perfect": 0.9,
            "awesome": 0.8, "brilliant": 0.8, "impressive": 0.7,

            # Negative words
            "bad": -0.5, "terrible": -0.8, "awful": -0.8, "horrible": -0.9,
            "hate": -0.8, "dislike": -0.5, "poor": -0.5, "worst": -0.9,
            "disappointing": -0.6, "frustrated": -0.6, "angry": -0.7,
            "sad": -0.6, "unhappy": -0.6, "useless": -0.7, "failed": -0.6,

            # Modifiers
            "very": 1.5, "really": 1.3, "extremely": 1.6, "quite": 1.2,
            "not": -1.0, "never": -1.0, "no": -0.5, "barely": -0.7,
        }

    def understand(self, text: str, context: Optional[Dict[str, Any]] = None) -> SemanticFrame:
        """
        Complete NLU pipeline - analyze text and return semantic frame

        Args:
            text: Input text to understand
            context: Optional context information

        Returns:
            SemanticFrame with complete semantic analysis
        """
        # Normalize text
        normalized_text = self._normalize_text(text)

        # Recognize intent
        intent = self.recognize_intent(normalized_text)

        # Extract entities
        entities = self.extract_entities(normalized_text)

        # Analyze sentiment
        sentiment, sentiment_score = self.analyze_sentiment(normalized_text)

        # Extract topics and keywords
        topics = self._extract_topics(normalized_text)
        keywords = self._extract_keywords(normalized_text)

        # Detect context references
        context_refs = self._detect_context_references(normalized_text)

        # Create semantic frame
        frame = SemanticFrame(
            text=text,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            topics=topics,
            keywords=keywords,
            context_references=context_refs
        )

        # Update context
        self._update_context(frame)

        return frame

    def _normalize_text(self, text: str) -> str:
        """Normalize input text"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def recognize_intent(self, text: str) -> Intent:
        """
        Recognize user intent from text

        Returns Intent with type, confidence, and slots
        """
        best_intent_type = IntentType.UNKNOWN
        best_confidence = 0.0
        matched_patterns = 0

        # Try each intent type
        for intent_type, patterns in self.intent_patterns.items():
            confidence = 0.0
            matches = 0

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matches += 1
                    confidence += 0.5

            # Normalize confidence
            if patterns:
                confidence = min(1.0, confidence / len(patterns) * 2)

            if confidence > best_confidence:
                best_confidence = confidence
                best_intent_type = intent_type
                matched_patterns = matches

        # If no strong match, use heuristics
        if best_confidence < 0.3:
            if "?" in text:
                best_intent_type = IntentType.QUESTION
                best_confidence = 0.6
            elif any(word in text.split() for word in ["please", "can", "could", "would"]):
                best_intent_type = IntentType.REQUEST
                best_confidence = 0.5

        # Determine domain
        domain = self._identify_domain(text)

        # Extract action
        action = self._extract_action(text, best_intent_type)

        # Fill slots
        slots = self._fill_slots(text, best_intent_type)

        return Intent(
            intent_type=best_intent_type,
            confidence=best_confidence,
            domain=domain,
            action=action,
            slots=slots
        )

    def _identify_domain(self, text: str) -> Optional[str]:
        """Identify domain of the utterance"""
        words = text.split()
        domain_scores = defaultdict(int)

        for domain, keywords in self.domains.items():
            for keyword in keywords:
                if keyword in words:
                    domain_scores[domain] += 1

        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return "general"

    def _extract_action(self, text: str, intent_type: IntentType) -> Optional[str]:
        """Extract the main action from text"""
        # Common action verbs
        action_verbs = [
            "create", "build", "make", "generate", "develop", "implement",
            "start", "stop", "run", "execute", "launch", "configure",
            "update", "change", "modify", "delete", "remove", "add",
            "show", "display", "list", "find", "search", "get",
        ]

        words = text.split()
        for word in words:
            if word in action_verbs:
                return word

        return None

    def _fill_slots(self, text: str, intent_type: IntentType) -> Dict[str, Any]:
        """Fill intent slots based on context"""
        slots = {}

        # Extract common slots
        words = text.split()

        # Object slot (what is being acted upon)
        if intent_type in [IntentType.COMMAND, IntentType.REQUEST]:
            # Find nouns after action verbs
            for i, word in enumerate(words[:-1]):
                if word in ["create", "build", "make", "show", "find"]:
                    if i + 1 < len(words):
                        slots["object"] = " ".join(words[i+1:i+3])
                        break

        # Target slot
        if "for" in words:
            idx = words.index("for")
            if idx + 1 < len(words):
                slots["target"] = " ".join(words[idx+1:idx+3])

        # Time slot
        time_words = ["now", "later", "tomorrow", "today", "tonight"]
        for word in time_words:
            if word in words:
                slots["time"] = word
                break

        return slots

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text

        Returns list of Entity objects with type, value, and position
        """
        entities = []

        # Pattern-based extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        entity_type=entity_type,
                        value=match.group(0),
                        start_idx=match.start(),
                        end_idx=match.end(),
                        confidence=0.8
                    )

                    # Normalize value if possible
                    entity.normalized_value = self._normalize_entity_value(
                        entity.value, entity_type
                    )

                    entities.append(entity)

        # Rule-based entity recognition
        entities.extend(self._extract_capitalized_entities(text))

        # Remove duplicates and overlaps
        entities = self._resolve_entity_conflicts(entities)

        return entities

    def _normalize_entity_value(self, value: str, entity_type: EntityType) -> Any:
        """Normalize entity value to canonical form"""
        if entity_type == EntityType.NUMBER:
            # Convert word numbers to digits
            word_to_num = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            return word_to_num.get(value.lower(), value)

        elif entity_type == EntityType.DATE:
            # Normalize dates (simplified)
            day_mapping = {
                "today": "TODAY",
                "tomorrow": "TODAY+1",
                "yesterday": "TODAY-1"
            }
            return day_mapping.get(value.lower(), value)

        return value

    def _extract_capitalized_entities(self, text: str) -> List[Entity]:
        """Extract entities based on capitalization patterns"""
        entities = []

        # Find capitalized words (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

        for match in re.finditer(pattern, text):
            # Determine entity type based on context
            entity_type = EntityType.PERSON  # Default

            # Simple heuristics
            value_lower = match.group(0).lower()
            if any(word in value_lower for word in ["inc", "corp", "ltd", "company"]):
                entity_type = EntityType.ORGANIZATION
            elif any(word in value_lower for word in ["city", "street", "avenue", "country"]):
                entity_type = EntityType.LOCATION

            entities.append(Entity(
                entity_type=entity_type,
                value=match.group(0),
                start_idx=match.start(),
                end_idx=match.end(),
                confidence=0.6
            ))

        return entities

    def _resolve_entity_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities"""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda e: e.start_idx)

        resolved = [entities[0]]

        for entity in entities[1:]:
            last = resolved[-1]

            # Check for overlap
            if entity.start_idx < last.end_idx:
                # Keep entity with higher confidence
                if entity.confidence > last.confidence:
                    resolved[-1] = entity
            else:
                resolved.append(entity)

        return resolved

    def analyze_sentiment(self, text: str) -> Tuple[SentimentPolarity, float]:
        """
        Analyze sentiment of text

        Returns:
            (polarity, score) where score is from -1 (negative) to 1 (positive)
        """
        words = text.lower().split()
        sentiment_score = 0.0
        modifier = 1.0

        for i, word in enumerate(words):
            # Check for modifiers
            if word in self.sentiment_lexicon:
                word_score = self.sentiment_lexicon[word]

                if abs(word_score) > 1:  # It's a modifier
                    modifier = word_score
                else:
                    sentiment_score += word_score * modifier
                    modifier = 1.0  # Reset modifier

        # Normalize score
        if len(words) > 0:
            sentiment_score = max(-1.0, min(1.0, sentiment_score / len(words) * 5))

        # Determine polarity
        if sentiment_score > 0.2:
            polarity = SentimentPolarity.POSITIVE
        elif sentiment_score < -0.2:
            polarity = SentimentPolarity.NEGATIVE
        elif abs(sentiment_score) < 0.1:
            polarity = SentimentPolarity.NEUTRAL
        else:
            polarity = SentimentPolarity.MIXED

        return polarity, sentiment_score

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        # Simple topic extraction based on noun phrases
        topics = []

        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        words = [w for w in text.split() if w not in stop_words and len(w) > 3]

        # Use word frequency as simple topic indicator
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1

        # Get top words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [topic for topic, _ in topics]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"}
        words = [w for w in text.split() if w not in stop_words and len(w) > 2]

        # Return unique words
        return list(set(words))[:10]

    def _detect_context_references(self, text: str) -> List[str]:
        """Detect references to previous context"""
        references = []

        # Pronouns and references
        ref_patterns = [
            r"\b(it|this|that|these|those)\b",
            r"\b(he|she|they|them)\b",
            r"\b(here|there)\b",
            r"\b(earlier|before|previously|above)\b",
        ]

        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)

        return list(set(references))

    def _update_context(self, frame: SemanticFrame):
        """Update context history"""
        self.context_history.append(frame)

        # Maintain max length
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)

    def resolve_coreference(self, text: str) -> str:
        """
        Resolve pronouns and references using context

        Returns text with references resolved
        """
        if not self.context_history:
            return text

        # Simple coreference resolution
        # In real system, use more sophisticated NLP
        resolved = text

        # Get last mentioned entities
        recent_entities = []
        for frame in reversed(self.context_history[-3:]):
            recent_entities.extend(frame.entities)

        # Replace pronouns with most recent matching entity
        pronoun_map = {
            "it": EntityType.OBJECT,
            "he": EntityType.PERSON,
            "she": EntityType.PERSON,
            "they": EntityType.PERSON,
        }

        for pronoun, entity_type in pronoun_map.items():
            if pronoun in text.lower():
                # Find most recent entity of this type
                for entity in reversed(recent_entities):
                    if entity.entity_type == entity_type:
                        resolved = re.sub(
                            rf"\b{pronoun}\b",
                            entity.value,
                            resolved,
                            flags=re.IGNORECASE,
                            count=1
                        )
                        break

        return resolved

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context"""
        return {
            "history_length": len(self.context_history),
            "recent_intents": [f.intent.intent_type for f in self.context_history[-5:]],
            "recent_topics": list(set(
                topic
                for frame in self.context_history[-5:]
                for topic in frame.topics
            ))[:10],
            "recent_entities": [
                {"type": e.entity_type, "value": e.value}
                for frame in self.context_history[-3:]
                for e in frame.entities
            ][:10],
        }
