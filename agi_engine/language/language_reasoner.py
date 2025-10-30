"""
Language Reasoner

Language-based reasoning capabilities:
- Question answering
- Reading comprehension
- Logical inference
- Entailment detection
- Ambiguity resolution
- Common sense reasoning
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict


class QuestionType(str, Enum):
    """Types of questions"""
    FACTUAL = "factual"  # What is X?
    CAUSAL = "causal"  # Why does X happen?
    PROCEDURAL = "procedural"  # How to do X?
    TEMPORAL = "temporal"  # When did X happen?
    LOCATIONAL = "locational"  # Where is X?
    PERSONAL = "personal"  # Who is X?
    BOOLEAN = "boolean"  # Is X true?
    CHOICE = "choice"  # Which X?
    QUANTITY = "quantity"  # How many X?
    COMPARISON = "comparison"  # What's the difference between X and Y?


class InferenceType(str, Enum):
    """Types of inferences"""
    DEDUCTIVE = "deductive"  # Logical deduction
    INDUCTIVE = "inductive"  # Generalization from examples
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # Reasoning by analogy
    COMMON_SENSE = "common_sense"  # World knowledge


class EntailmentRelation(str, Enum):
    """Entailment relations"""
    ENTAILS = "entails"  # A implies B
    CONTRADICTS = "contradicts"  # A contradicts B
    NEUTRAL = "neutral"  # No relation


@dataclass
class Question:
    """Representation of a question"""
    text: str
    question_type: QuestionType
    focus: str  # What the question is about
    constraints: Dict[str, Any] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    expected_answer_type: Optional[str] = None


@dataclass
class Answer:
    """Representation of an answer"""
    text: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inference:
    """Logical inference"""
    inference_type: InferenceType
    premises: List[str]
    conclusion: str
    confidence: float
    rule_applied: Optional[str] = None
    intermediate_steps: List[str] = field(default_factory=list)


@dataclass
class Fact:
    """A fact in the knowledge base"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageReasoner:
    """
    Language-based Reasoning Engine

    Capabilities:
    - Question analysis and classification
    - Answer generation with evidence
    - Logical inference (deductive, inductive, abductive)
    - Reading comprehension
    - Entailment detection
    - Ambiguity resolution
    - Common sense reasoning
    """

    def __init__(self):
        # Knowledge base
        self.facts: List[Fact] = []
        self.fact_index: Dict[str, List[Fact]] = defaultdict(list)

        # Inference rules
        self.inference_rules = self._build_inference_rules()

        # Question patterns
        self.question_patterns = self._build_question_patterns()

        # Common sense knowledge
        self.common_sense = self._build_common_sense_kb()

        # Entailment patterns
        self.entailment_patterns = self._build_entailment_patterns()

    def _build_inference_rules(self) -> List[Dict[str, Any]]:
        """Build logical inference rules"""
        return [
            {
                "name": "modus_ponens",
                "pattern": ["If A then B", "A"],
                "conclusion": "B",
                "type": InferenceType.DEDUCTIVE
            },
            {
                "name": "modus_tollens",
                "pattern": ["If A then B", "not B"],
                "conclusion": "not A",
                "type": InferenceType.DEDUCTIVE
            },
            {
                "name": "syllogism",
                "pattern": ["All A are B", "All B are C"],
                "conclusion": "All A are C",
                "type": InferenceType.DEDUCTIVE
            },
            {
                "name": "generalization",
                "pattern": ["X1 is A", "X2 is A", "X3 is A"],
                "conclusion": "Most X are A",
                "type": InferenceType.INDUCTIVE
            },
            {
                "name": "best_explanation",
                "pattern": ["Observation O", "A explains O better than B"],
                "conclusion": "A is likely true",
                "type": InferenceType.ABDUCTIVE
            },
        ]

    def _build_question_patterns(self) -> Dict[QuestionType, List[str]]:
        """Build question classification patterns"""
        return {
            QuestionType.FACTUAL: [r"^what is", r"^what are", r"^define"],
            QuestionType.CAUSAL: [r"^why", r"^what causes", r"^what makes"],
            QuestionType.PROCEDURAL: [r"^how to", r"^how do", r"^how can"],
            QuestionType.TEMPORAL: [r"^when", r"^what time", r"^at what point"],
            QuestionType.LOCATIONAL: [r"^where", r"^what place", r"^in which location"],
            QuestionType.PERSONAL: [r"^who", r"^which person", r"^whose"],
            QuestionType.BOOLEAN: [r"^is", r"^are", r"^does", r"^did", r"^can", r"^will"],
            QuestionType.CHOICE: [r"^which", r"^what.*or", r"^which one"],
            QuestionType.QUANTITY: [r"^how many", r"^how much", r"^what number"],
            QuestionType.COMPARISON: [r"^what.*difference", r"^how.*compare", r"^.*vs"],
        }

    def _build_common_sense_kb(self) -> Dict[str, List[str]]:
        """Build common sense knowledge base"""
        return {
            # Physical properties
            "heavy_objects": ["elephant", "car", "building", "boulder"],
            "light_objects": ["feather", "paper", "balloon", "leaf"],
            "hot_things": ["fire", "sun", "oven", "lava"],
            "cold_things": ["ice", "snow", "freezer", "winter"],

            # Temporal relations
            "morning_activities": ["wake up", "breakfast", "commute"],
            "evening_activities": ["dinner", "relax", "sleep"],

            # Causal relations
            "causes_growth": ["water", "sunlight", "nutrients"],
            "causes_damage": ["fire", "flood", "earthquake"],

            # Social conventions
            "polite_actions": ["thank", "apologize", "greet"],
            "impolite_actions": ["interrupt", "yell", "ignore"],
        }

    def _build_entailment_patterns(self) -> List[Dict[str, Any]]:
        """Build entailment detection patterns"""
        return [
            {
                "pattern": r"(.*) is a (.*)",
                "entails": r"\1 is a type of \2",
                "relation": EntailmentRelation.ENTAILS
            },
            {
                "pattern": r"All (.*) are (.*)",
                "entails": r"Some \1 are \2",
                "relation": EntailmentRelation.ENTAILS
            },
            {
                "pattern": r"(.*) is (.* tall)",
                "contradicts": r"\1 is short",
                "relation": EntailmentRelation.CONTRADICTS
            },
        ]

    def analyze_question(self, question_text: str) -> Question:
        """
        Analyze question and classify type

        Args:
            question_text: Question string

        Returns:
            Question object with classification
        """
        question_text = question_text.strip().lower()

        # Classify question type
        question_type = self._classify_question_type(question_text)

        # Extract question focus
        focus = self._extract_question_focus(question_text, question_type)

        # Extract keywords
        keywords = self._extract_keywords(question_text)

        # Determine expected answer type
        expected_type = self._determine_answer_type(question_type)

        return Question(
            text=question_text,
            question_type=question_type,
            focus=focus,
            keywords=keywords,
            expected_answer_type=expected_type
        )

    def _classify_question_type(self, question_text: str) -> QuestionType:
        """Classify question into type"""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_text, re.IGNORECASE):
                    return q_type

        # Default to factual
        return QuestionType.FACTUAL

    def _extract_question_focus(self, question_text: str, q_type: QuestionType) -> str:
        """Extract main focus of question"""
        # Remove question words
        question_words = ["what", "why", "how", "when", "where", "who", "which", "is", "are", "does", "did"]

        words = question_text.split()
        focus_words = [w for w in words if w not in question_words and len(w) > 2]

        if focus_words:
            # Take first significant word as focus
            return focus_words[0]

        return "unknown"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at", "for"}
        words = [w for w in text.split() if w not in stop_words and len(w) > 2]

        return words[:5]  # Top 5 keywords

    def _determine_answer_type(self, question_type: QuestionType) -> str:
        """Determine expected answer type"""
        type_map = {
            QuestionType.FACTUAL: "entity",
            QuestionType.CAUSAL: "explanation",
            QuestionType.PROCEDURAL: "procedure",
            QuestionType.TEMPORAL: "time",
            QuestionType.LOCATIONAL: "location",
            QuestionType.PERSONAL: "person",
            QuestionType.BOOLEAN: "yes_no",
            QuestionType.CHOICE: "option",
            QuestionType.QUANTITY: "number",
            QuestionType.COMPARISON: "comparison",
        }

        return type_map.get(question_type, "text")

    def answer_question(
        self,
        question: Question,
        context: Optional[str] = None,
        knowledge: Optional[Dict[str, Any]] = None
    ) -> Answer:
        """
        Answer a question using available knowledge

        Args:
            question: Question to answer
            context: Optional context text
            knowledge: Optional additional knowledge

        Returns:
            Answer with confidence and evidence
        """
        # Search for relevant facts
        relevant_facts = self._find_relevant_facts(question)

        # Extract information from context if provided
        context_info = []
        if context:
            context_info = self._extract_from_context(question, context)

        # Use additional knowledge if provided
        if knowledge:
            for key, value in knowledge.items():
                relevant_facts.append(
                    Fact(subject=key, predicate="is", object=str(value))
                )

        # Generate answer based on question type
        if question.question_type == QuestionType.BOOLEAN:
            answer_text, confidence = self._answer_boolean(question, relevant_facts)
        elif question.question_type == QuestionType.FACTUAL:
            answer_text, confidence = self._answer_factual(question, relevant_facts, context_info)
        elif question.question_type == QuestionType.CAUSAL:
            answer_text, confidence = self._answer_causal(question, relevant_facts)
        elif question.question_type == QuestionType.PROCEDURAL:
            answer_text, confidence = self._answer_procedural(question)
        elif question.question_type == QuestionType.COMPARISON:
            answer_text, confidence = self._answer_comparison(question, relevant_facts)
        else:
            answer_text, confidence = self._answer_generic(question, relevant_facts, context_info)

        # Collect evidence
        evidence = [f"{f.subject} {f.predicate} {f.object}" for f in relevant_facts[:3]]

        return Answer(
            text=answer_text,
            confidence=confidence,
            evidence=evidence,
            source="knowledge_base" if relevant_facts else "inference"
        )

    def _find_relevant_facts(self, question: Question) -> List[Fact]:
        """Find facts relevant to question"""
        relevant = []

        # Search by keywords
        for keyword in question.keywords:
            if keyword in self.fact_index:
                relevant.extend(self.fact_index[keyword])

        # Search by focus
        if question.focus in self.fact_index:
            relevant.extend(self.fact_index[question.focus])

        # Remove duplicates
        seen = set()
        unique_facts = []
        for fact in relevant:
            fact_str = f"{fact.subject}:{fact.predicate}:{fact.object}"
            if fact_str not in seen:
                seen.add(fact_str)
                unique_facts.append(fact)

        return unique_facts

    def _extract_from_context(self, question: Question, context: str) -> List[str]:
        """Extract relevant information from context"""
        # Simple extraction based on sentence matching
        sentences = context.split('.')
        relevant = []

        for sentence in sentences:
            # Check if sentence contains question keywords
            if any(kw in sentence.lower() for kw in question.keywords):
                relevant.append(sentence.strip())

        return relevant

    def _answer_boolean(
        self,
        question: Question,
        facts: List[Fact]
    ) -> Tuple[str, float]:
        """Answer yes/no question"""
        if not facts:
            return "I don't have enough information to answer.", 0.3

        # Check if facts support the question
        # Simplified logic
        return "Yes, that appears to be correct.", 0.7

    def _answer_factual(
        self,
        question: Question,
        facts: List[Fact],
        context_info: List[str]
    ) -> Tuple[str, float]:
        """Answer factual question"""
        # Use context if available
        if context_info:
            return context_info[0], 0.8

        # Use facts
        if facts:
            fact = facts[0]
            return f"{fact.subject} {fact.predicate} {fact.object}", fact.confidence

        return "I don't have that information.", 0.2

    def _answer_causal(
        self,
        question: Question,
        facts: List[Fact]
    ) -> Tuple[str, float]:
        """Answer causal question"""
        # Look for causal relations
        for fact in facts:
            if "causes" in fact.predicate or "because" in fact.predicate:
                return f"Because {fact.object}", fact.confidence

        # Use common sense
        for cause_type, items in self.common_sense.items():
            if any(item in question.text for item in items):
                return f"This is related to {cause_type.replace('_', ' ')}", 0.6

        return "The exact cause is unclear from available information.", 0.4

    def _answer_procedural(self, question: Question) -> Tuple[str, float]:
        """Answer how-to question"""
        # Would need procedure knowledge base
        return "To do this, you would need to follow these steps: [steps would be provided here]", 0.5

    def _answer_comparison(
        self,
        question: Question,
        facts: List[Fact]
    ) -> Tuple[str, float]:
        """Answer comparison question"""
        if len(facts) >= 2:
            return f"The main difference is that {facts[0].subject} {facts[0].predicate} {facts[0].object}, while {facts[1].subject} {facts[1].predicate} {facts[1].object}", 0.7

        return "I would need more information to make a comparison.", 0.3

    def _answer_generic(
        self,
        question: Question,
        facts: List[Fact],
        context_info: List[str]
    ) -> Tuple[str, float]:
        """Generic answer generation"""
        if context_info:
            return context_info[0], 0.7
        elif facts:
            return f"{facts[0].subject} {facts[0].predicate} {facts[0].object}", facts[0].confidence
        else:
            return "I don't have enough information to answer that question.", 0.2

    def infer(
        self,
        premises: List[str],
        inference_type: Optional[InferenceType] = None
    ) -> Optional[Inference]:
        """
        Make logical inference from premises

        Args:
            premises: List of premise statements
            inference_type: Optional specific inference type

        Returns:
            Inference with conclusion
        """
        if inference_type:
            # Use specific inference type
            return self._apply_inference_type(premises, inference_type)

        # Try all inference rules
        for rule in self.inference_rules:
            result = self._try_inference_rule(premises, rule)
            if result:
                return result

        return None

    def _apply_inference_type(
        self,
        premises: List[str],
        inference_type: InferenceType
    ) -> Optional[Inference]:
        """Apply specific inference type"""
        # Filter rules by type
        applicable_rules = [r for r in self.inference_rules if r["type"] == inference_type]

        for rule in applicable_rules:
            result = self._try_inference_rule(premises, rule)
            if result:
                return result

        return None

    def _try_inference_rule(
        self,
        premises: List[str],
        rule: Dict[str, Any]
    ) -> Optional[Inference]:
        """Try to apply inference rule"""
        # Simplified pattern matching
        # In real system, use more sophisticated logic

        if len(premises) < len(rule["pattern"]):
            return None

        # Check if premises match pattern (very simplified)
        confidence = 0.8

        conclusion = rule["conclusion"]

        return Inference(
            inference_type=rule["type"],
            premises=premises,
            conclusion=conclusion,
            confidence=confidence,
            rule_applied=rule["name"]
        )

    def detect_entailment(self, text1: str, text2: str) -> EntailmentRelation:
        """
        Detect entailment relation between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Entailment relation
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Check for explicit contradiction markers
        if self._contains_negation(text1) != self._contains_negation(text2):
            # One is negated, other is not
            if self._same_core_meaning(text1_lower, text2_lower):
                return EntailmentRelation.CONTRADICTS

        # Check entailment patterns
        for pattern_info in self.entailment_patterns:
            if "entails" in pattern_info:
                match1 = re.search(pattern_info["pattern"], text1_lower)
                if match1 and pattern_info["entails"] in text2_lower:
                    return EntailmentRelation.ENTAILS

        # Check for semantic overlap
        overlap = self._semantic_overlap(text1_lower, text2_lower)
        if overlap > 0.8:
            return EntailmentRelation.ENTAILS
        elif overlap > 0.3:
            return EntailmentRelation.NEUTRAL

        return EntailmentRelation.NEUTRAL

    def _contains_negation(self, text: str) -> bool:
        """Check if text contains negation"""
        negation_words = ["not", "no", "never", "neither", "nor", "none", "nobody", "nothing"]
        return any(word in text.lower().split() for word in negation_words)

    def _same_core_meaning(self, text1: str, text2: str) -> bool:
        """Check if texts have same core meaning (ignoring negation)"""
        # Remove negation words
        negation_words = ["not", "no", "never", "neither", "nor", "none"]

        words1 = [w for w in text1.split() if w not in negation_words]
        words2 = [w for w in text2.split() if w not in negation_words]

        # Check overlap
        overlap = len(set(words1) & set(words2))
        total = len(set(words1) | set(words2))

        return overlap / max(total, 1) > 0.6

    def _semantic_overlap(self, text1: str, text2: str) -> float:
        """Calculate semantic overlap between texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return overlap / max(total, 1)

    def resolve_ambiguity(
        self,
        text: str,
        possible_meanings: List[str]
    ) -> Tuple[str, float]:
        """
        Resolve ambiguous text

        Args:
            text: Ambiguous text
            possible_meanings: List of possible interpretations

        Returns:
            (most_likely_meaning, confidence)
        """
        if not possible_meanings:
            return text, 0.5

        if len(possible_meanings) == 1:
            return possible_meanings[0], 1.0

        # Score each meaning based on context and common sense
        scores = []
        for meaning in possible_meanings:
            score = self._score_interpretation(text, meaning)
            scores.append((meaning, score))

        # Return highest scoring
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0]

    def _score_interpretation(self, text: str, interpretation: str) -> float:
        """Score interpretation of ambiguous text"""
        # Simple scoring based on word overlap
        text_words = set(text.lower().split())
        interp_words = set(interpretation.lower().split())

        overlap = len(text_words & interp_words)
        score = overlap / max(len(text_words), 1)

        return score

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0
    ):
        """Add fact to knowledge base"""
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence
        )

        self.facts.append(fact)

        # Index by subject and object
        self.fact_index[subject.lower()].append(fact)
        self.fact_index[object.lower()].append(fact)

    def comprehend(self, text: str) -> Dict[str, Any]:
        """
        Reading comprehension - extract structured information

        Args:
            text: Text to comprehend

        Returns:
            Structured comprehension
        """
        # Extract entities, relations, and facts
        sentences = text.split('.')

        extracted_facts = []
        entities = set()
        relations = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Simple pattern matching for subject-verb-object
            words = sentence.split()
            if len(words) >= 3:
                # Assume subject-verb-object pattern
                subject = words[0]
                verb = words[1] if len(words) > 1 else "is"
                obj = " ".join(words[2:]) if len(words) > 2 else ""

                entities.add(subject)
                if obj:
                    entities.add(obj)
                relations.add(verb)

                extracted_facts.append({
                    "subject": subject,
                    "predicate": verb,
                    "object": obj
                })

        return {
            "entities": list(entities),
            "relations": list(relations),
            "facts": extracted_facts,
            "sentence_count": len([s for s in sentences if s.strip()])
        }

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "total_facts": len(self.facts),
            "indexed_entities": len(self.fact_index),
            "inference_rules": len(self.inference_rules),
            "common_sense_categories": len(self.common_sense),
        }
