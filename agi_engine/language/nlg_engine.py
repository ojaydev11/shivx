"""
Natural Language Generation Engine

Provides comprehensive NLG capabilities:
- Text generation from semantic representations
- Response synthesis
- Explanation generation
- Multi-style generation
- Template-based and neural generation
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import re
from collections import defaultdict


class GenerationStrategy(str, Enum):
    """Text generation strategies"""
    TEMPLATE = "template"
    RULE_BASED = "rule_based"
    RETRIEVAL = "retrieval"
    NEURAL = "neural"
    HYBRID = "hybrid"


class TextStyle(str, Enum):
    """Output text styles"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    DETAILED = "detailed"
    CONCISE = "concise"


class ResponseType(str, Enum):
    """Types of responses"""
    ANSWER = "answer"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    ERROR = "error"
    EXPLANATION = "explanation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    SUGGESTION = "suggestion"


@dataclass
class ResponseTemplate:
    """Template for generating responses"""
    template_id: str
    pattern: str
    response_type: ResponseType
    slots: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    style: TextStyle = TextStyle.CASUAL
    priority: float = 1.0


@dataclass
class GenerationContext:
    """Context for text generation"""
    user_name: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    formality_level: float = 0.5  # 0=casual, 1=formal
    verbosity_level: float = 0.5  # 0=concise, 1=verbose
    personality_traits: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


class NLGEngine:
    """
    Natural Language Generation Engine

    Provides:
    - Template-based generation
    - Rule-based generation
    - Response synthesis
    - Explanation generation
    - Multi-style text generation
    - Context-aware generation
    """

    def __init__(self):
        # Response templates
        self.templates = self._build_templates()

        # Generation rules
        self.generation_rules = self._build_generation_rules()

        # Style variations
        self.style_transformers = self._build_style_transformers()

        # Discourse markers
        self.discourse_markers = self._build_discourse_markers()

        # Current context
        self.context = GenerationContext()

    def _build_templates(self) -> Dict[ResponseType, List[ResponseTemplate]]:
        """Build response templates"""
        templates = defaultdict(list)

        # Greeting templates
        templates[ResponseType.GREETING] = [
            ResponseTemplate("greet_1", "Hello{user}! How can I help you today?", ResponseType.GREETING, ["user"]),
            ResponseTemplate("greet_2", "Hi{user}! What can I do for you?", ResponseType.GREETING, ["user"]),
            ResponseTemplate("greet_3", "Greetings{user}! I'm here to assist you.", ResponseType.GREETING, ["user"]),
        ]

        # Farewell templates
        templates[ResponseType.FAREWELL] = [
            ResponseTemplate("bye_1", "Goodbye{user}! Feel free to come back anytime.", ResponseType.FAREWELL, ["user"]),
            ResponseTemplate("bye_2", "See you later{user}! Have a great day!", ResponseType.FAREWELL, ["user"]),
            ResponseTemplate("bye_3", "Farewell{user}! It was a pleasure assisting you.", ResponseType.FAREWELL, ["user"]),
        ]

        # Confirmation templates
        templates[ResponseType.CONFIRMATION] = [
            ResponseTemplate("confirm_1", "Got it! I'll {action} right away.", ResponseType.CONFIRMATION, ["action"]),
            ResponseTemplate("confirm_2", "Understood. I'm {action} now.", ResponseType.CONFIRMATION, ["action"]),
            ResponseTemplate("confirm_3", "Sure thing! {action} in progress.", ResponseType.CONFIRMATION, ["action"]),
        ]

        # Clarification templates
        templates[ResponseType.CLARIFICATION] = [
            ResponseTemplate("clarify_1", "I'm not sure I understand. Could you clarify {aspect}?", ResponseType.CLARIFICATION, ["aspect"]),
            ResponseTemplate("clarify_2", "To make sure I get this right, you want me to {action}?", ResponseType.CLARIFICATION, ["action"]),
            ResponseTemplate("clarify_3", "Just to confirm: {question}?", ResponseType.CLARIFICATION, ["question"]),
        ]

        # Error templates
        templates[ResponseType.ERROR] = [
            ResponseTemplate("error_1", "I encountered an error: {error}. Let me try a different approach.", ResponseType.ERROR, ["error"]),
            ResponseTemplate("error_2", "Sorry, I couldn't {action} because {reason}.", ResponseType.ERROR, ["action", "reason"]),
            ResponseTemplate("error_3", "There was a problem: {error}. Would you like me to try again?", ResponseType.ERROR, ["error"]),
        ]

        # Explanation templates
        templates[ResponseType.EXPLANATION] = [
            ResponseTemplate("explain_1", "Here's why: {reason}. {details}", ResponseType.EXPLANATION, ["reason", "details"]),
            ResponseTemplate("explain_2", "The reason is {reason}. In other words, {simplified}.", ResponseType.EXPLANATION, ["reason", "simplified"]),
            ResponseTemplate("explain_3", "To explain: {explanation}. Does that make sense?", ResponseType.EXPLANATION, ["explanation"]),
        ]

        # Suggestion templates
        templates[ResponseType.SUGGESTION] = [
            ResponseTemplate("suggest_1", "I suggest {suggestion}. This would {benefit}.", ResponseType.SUGGESTION, ["suggestion", "benefit"]),
            ResponseTemplate("suggest_2", "You might want to {action}. It could {result}.", ResponseType.SUGGESTION, ["action", "result"]),
            ResponseTemplate("suggest_3", "Have you considered {option}? It would {advantage}.", ResponseType.SUGGESTION, ["option", "advantage"]),
        ]

        return templates

    def _build_generation_rules(self) -> Dict[str, Callable]:
        """Build rule-based generation functions"""
        return {
            "enumeration": self._generate_enumeration,
            "comparison": self._generate_comparison,
            "definition": self._generate_definition,
            "procedure": self._generate_procedure,
            "description": self._generate_description,
        }

    def _build_style_transformers(self) -> Dict[TextStyle, Callable]:
        """Build style transformation functions"""
        return {
            TextStyle.FORMAL: self._make_formal,
            TextStyle.CASUAL: self._make_casual,
            TextStyle.TECHNICAL: self._make_technical,
            TextStyle.SIMPLE: self._make_simple,
            TextStyle.DETAILED: self._make_detailed,
            TextStyle.CONCISE: self._make_concise,
        }

    def _build_discourse_markers(self) -> Dict[str, List[str]]:
        """Build discourse markers for coherent text"""
        return {
            "sequence": ["first", "second", "third", "then", "next", "finally"],
            "contrast": ["however", "but", "on the other hand", "conversely"],
            "cause": ["because", "since", "due to", "as a result"],
            "example": ["for example", "for instance", "such as", "like"],
            "emphasis": ["indeed", "in fact", "actually", "notably"],
            "conclusion": ["therefore", "thus", "in conclusion", "to sum up"],
        }

    def generate(
        self,
        content: Dict[str, Any],
        response_type: ResponseType = ResponseType.ANSWER,
        style: TextStyle = TextStyle.CASUAL,
        strategy: GenerationStrategy = GenerationStrategy.HYBRID
    ) -> str:
        """
        Generate natural language text from content

        Args:
            content: Semantic content to express
            response_type: Type of response to generate
            style: Desired text style
            strategy: Generation strategy to use

        Returns:
            Generated natural language text
        """
        # Select generation strategy
        if strategy == GenerationStrategy.TEMPLATE:
            text = self._generate_from_template(content, response_type)
        elif strategy == GenerationStrategy.RULE_BASED:
            text = self._generate_rule_based(content)
        elif strategy == GenerationStrategy.RETRIEVAL:
            text = self._generate_retrieval_based(content)
        elif strategy == GenerationStrategy.HYBRID:
            # Use templates for structured responses, rules for complex content
            if response_type in self.templates and len(content) <= 3:
                text = self._generate_from_template(content, response_type)
            else:
                text = self._generate_rule_based(content)
        else:
            text = self._generate_rule_based(content)

        # Apply style transformation
        text = self._apply_style(text, style)

        # Post-process
        text = self._post_process(text)

        return text

    def _generate_from_template(
        self,
        content: Dict[str, Any],
        response_type: ResponseType
    ) -> str:
        """Generate text using templates"""
        templates = self.templates.get(response_type, [])

        if not templates:
            # Fallback to generic template
            return self._generate_generic_response(content)

        # Select template based on available slots
        best_template = None
        best_coverage = 0

        for template in templates:
            # Check how many slots we can fill
            coverage = sum(1 for slot in template.slots if slot in content)
            if coverage > best_coverage:
                best_coverage = coverage
                best_template = template

        if not best_template:
            best_template = random.choice(templates)

        # Fill template
        text = best_template.pattern

        for slot in best_template.slots:
            placeholder = "{" + slot + "}"
            value = content.get(slot, "")

            if slot == "user" and value:
                value = f" {value}"
            elif not value:
                # Remove placeholder if no value
                text = text.replace(placeholder, "")

            text = text.replace(placeholder, str(value))

        return text

    def _generate_generic_response(self, content: Dict[str, Any]) -> str:
        """Generate generic response from content"""
        parts = []

        for key, value in content.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                parts.append(f"{key}: {value}")

        return ". ".join(parts) + "."

    def _generate_rule_based(self, content: Dict[str, Any]) -> str:
        """Generate text using rules"""
        # Detect content type and use appropriate rule
        if "items" in content or "list" in content:
            return self._generate_enumeration(content)
        elif "comparison" in content or any("vs" in str(k).lower() for k in content.keys()):
            return self._generate_comparison(content)
        elif "definition" in content or "concept" in content:
            return self._generate_definition(content)
        elif "steps" in content or "procedure" in content:
            return self._generate_procedure(content)
        else:
            return self._generate_description(content)

    def _generate_enumeration(self, content: Dict[str, Any]) -> str:
        """Generate enumeration/list"""
        items = content.get("items", content.get("list", []))

        if not items:
            return "No items to display."

        intro = content.get("intro", "Here are the items:")
        parts = [intro]

        for i, item in enumerate(items, 1):
            marker = self.discourse_markers["sequence"][min(i-1, 5)]
            parts.append(f"{marker}, {item}")

        return " ".join(parts) + "."

    def _generate_comparison(self, content: Dict[str, Any]) -> str:
        """Generate comparison"""
        item1 = content.get("item1", content.get("a", "first item"))
        item2 = content.get("item2", content.get("b", "second item"))
        aspects = content.get("aspects", [])

        parts = [f"Comparing {item1} and {item2}:"]

        if aspects:
            for aspect in aspects:
                val1 = content.get(f"{aspect}_1", "unknown")
                val2 = content.get(f"{aspect}_2", "unknown")
                parts.append(f"{item1} has {aspect} of {val1}, while {item2} has {val2}.")
        else:
            parts.append(f"{item1} differs from {item2} in several ways.")

        return " ".join(parts)

    def _generate_definition(self, content: Dict[str, Any]) -> str:
        """Generate definition"""
        concept = content.get("concept", content.get("term", "the concept"))
        definition = content.get("definition", "")
        examples = content.get("examples", [])

        parts = [f"{concept} is {definition}."]

        if examples:
            parts.append(f"For example, {examples[0]}.")

        return " ".join(parts)

    def _generate_procedure(self, content: Dict[str, Any]) -> str:
        """Generate procedure/steps"""
        steps = content.get("steps", content.get("procedure", []))
        goal = content.get("goal", "complete the task")

        parts = [f"To {goal}:"]

        for i, step in enumerate(steps):
            marker = self.discourse_markers["sequence"][min(i, 5)]
            parts.append(f"{marker}, {step}")

        return " ".join(parts) + "."

    def _generate_description(self, content: Dict[str, Any]) -> str:
        """Generate descriptive text"""
        parts = []

        for key, value in content.items():
            if isinstance(value, (list, tuple)):
                value_str = ", ".join(str(v) for v in value)
                parts.append(f"The {key} include {value_str}")
            elif isinstance(value, dict):
                # Nested content
                parts.append(f"Regarding {key}: {self._generate_description(value)}")
            else:
                parts.append(f"The {key} is {value}")

        return ". ".join(parts) + "."

    def _generate_retrieval_based(self, content: Dict[str, Any]) -> str:
        """Generate using retrieval (placeholder for future implementation)"""
        # In real system, retrieve similar responses from corpus
        return self._generate_rule_based(content)

    def _apply_style(self, text: str, style: TextStyle) -> str:
        """Apply style transformation to text"""
        transformer = self.style_transformers.get(style)

        if transformer:
            text = transformer(text)

        return text

    def _make_formal(self, text: str) -> str:
        """Transform to formal style"""
        # Replace contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "I'm": "I am",
            "you're": "you are",
            "it's": "it is",
        }

        for contraction, expansion in contractions.items():
            text = re.sub(rf"\b{contraction}\b", expansion, text, flags=re.IGNORECASE)

        # Remove casual phrases
        text = text.replace("got it", "understood")
        text = text.replace("sure thing", "certainly")
        text = text.replace("okay", "very well")

        return text

    def _make_casual(self, text: str) -> str:
        """Transform to casual style"""
        # Add contractions
        expansions = {
            "cannot": "can't",
            "will not": "won't",
            "do not": "don't",
            "does not": "doesn't",
            "is not": "isn't",
            "are not": "aren't",
        }

        for expansion, contraction in expansions.items():
            text = re.sub(rf"\b{expansion}\b", contraction, text, flags=re.IGNORECASE)

        return text

    def _make_technical(self, text: str) -> str:
        """Transform to technical style"""
        # Add technical precision
        text = re.sub(r"\bthings\b", "components", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdo\b", "execute", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmake\b", "generate", text, flags=re.IGNORECASE)

        return text

    def _make_simple(self, text: str) -> str:
        """Transform to simple style"""
        # Simplify complex words
        simplifications = {
            "utilize": "use",
            "terminate": "end",
            "commence": "start",
            "acquire": "get",
            "demonstrate": "show",
        }

        for complex_word, simple_word in simplifications.items():
            text = re.sub(rf"\b{complex_word}\b", simple_word, text, flags=re.IGNORECASE)

        return text

    def _make_detailed(self, text: str) -> str:
        """Transform to detailed style"""
        # Add elaboration (simplified implementation)
        sentences = text.split(". ")
        detailed_sentences = []

        for sentence in sentences:
            detailed_sentences.append(sentence)
            # Add clarification phrases occasionally
            if random.random() < 0.3:
                detailed_sentences.append("In other words, this means that the process involves careful consideration")

        return ". ".join(detailed_sentences)

    def _make_concise(self, text: str) -> str:
        """Transform to concise style"""
        # Remove filler words
        fillers = ["very", "really", "quite", "just", "basically", "actually"]

        for filler in fillers:
            text = re.sub(rf"\b{filler}\b\s*", "", text, flags=re.IGNORECASE)

        # Remove redundant phrases
        text = text.replace("in order to", "to")
        text = text.replace("due to the fact that", "because")

        return text

    def _post_process(self, text: str) -> str:
        """Post-process generated text"""
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)

        # Fix punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Ensure ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'

        return text.strip()

    def explain(
        self,
        concept: str,
        explanation: str,
        examples: Optional[List[str]] = None,
        style: TextStyle = TextStyle.CASUAL
    ) -> str:
        """
        Generate explanation

        Args:
            concept: What to explain
            explanation: Core explanation
            examples: Optional examples
            style: Explanation style

        Returns:
            Well-formed explanation
        """
        parts = []

        # Introduction
        parts.append(f"Let me explain {concept}.")

        # Core explanation
        parts.append(explanation)

        # Examples
        if examples:
            parts.append("For example:")
            for i, example in enumerate(examples[:3], 1):
                parts.append(f"{i}. {example}")

        # Conclusion
        parts.append("Does this help clarify things?")

        text = " ".join(parts)
        return self._apply_style(text, style)

    def synthesize_response(
        self,
        intent: str,
        content: Dict[str, Any],
        context: Optional[GenerationContext] = None
    ) -> str:
        """
        Synthesize appropriate response based on intent and content

        Args:
            intent: User intent
            content: Response content
            context: Generation context

        Returns:
            Synthesized response
        """
        if context:
            self.context = context

        # Map intent to response type
        intent_to_type = {
            "question": ResponseType.ANSWER,
            "command": ResponseType.CONFIRMATION,
            "greeting": ResponseType.GREETING,
            "farewell": ResponseType.FAREWELL,
            "clarification": ResponseType.CLARIFICATION,
        }

        response_type = intent_to_type.get(intent.lower(), ResponseType.ANSWER)

        # Determine style based on context
        style = TextStyle.CASUAL
        if self.context.formality_level > 0.7:
            style = TextStyle.FORMAL
        elif self.context.domain and "technical" in self.context.domain:
            style = TextStyle.TECHNICAL

        # Generate response
        return self.generate(content, response_type, style)

    def generate_multi_sentence(
        self,
        sentences: List[Dict[str, Any]],
        coherence_markers: bool = True
    ) -> str:
        """
        Generate multiple coherent sentences

        Args:
            sentences: List of sentence contents
            coherence_markers: Whether to add discourse markers

        Returns:
            Coherent multi-sentence text
        """
        parts = []

        for i, sentence_content in enumerate(sentences):
            # Generate individual sentence
            sentence = self._generate_rule_based(sentence_content)

            # Add discourse marker if needed
            if coherence_markers and i > 0:
                # Select appropriate marker
                marker_type = sentence_content.get("relation", "sequence")
                if marker_type in self.discourse_markers:
                    marker = random.choice(self.discourse_markers[marker_type])
                    sentence = f"{marker}, {sentence.lower()}"

            parts.append(sentence)

        return " ".join(parts)

    def paraphrase(self, text: str, style: Optional[TextStyle] = None) -> str:
        """
        Generate paraphrase of text

        Args:
            text: Text to paraphrase
            style: Optional target style

        Returns:
            Paraphrased text
        """
        # Simple paraphrasing using synonyms and restructuring
        paraphrase_map = {
            "create": "generate",
            "build": "construct",
            "make": "produce",
            "show": "display",
            "find": "locate",
            "get": "obtain",
            "use": "utilize",
            "help": "assist",
        }

        words = text.split()
        new_words = []

        for word in words:
            lower_word = word.lower()
            if lower_word in paraphrase_map and random.random() < 0.5:
                new_words.append(paraphrase_map[lower_word])
            else:
                new_words.append(word)

        paraphrased = " ".join(new_words)

        if style:
            paraphrased = self._apply_style(paraphrased, style)

        return paraphrased
