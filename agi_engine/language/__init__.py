"""
Language Intelligence - Pillar 6 of ShivX AGI System

Natural language understanding, generation, and reasoning capabilities.

Components:
- NLU Engine: Intent recognition, entity extraction, semantic parsing
- NLG Engine: Text generation, response synthesis, explanation generation
- Dialogue Manager: Multi-turn conversation, context tracking
- Language Reasoner: Question answering, inference, comprehension
"""

from agi_engine.language.nlu_engine import (
    NLUEngine,
    Intent,
    Entity,
    SemanticFrame,
    IntentType,
    EntityType
)
from agi_engine.language.nlg_engine import (
    NLGEngine,
    ResponseTemplate,
    GenerationStrategy,
    TextStyle,
    ResponseType
)
from agi_engine.language.dialogue_manager import (
    DialogueManager,
    DialogueState,
    Turn,
    DialogueAct
)
from agi_engine.language.language_reasoner import (
    LanguageReasoner,
    Question,
    Answer,
    Inference,
    QuestionType
)

__all__ = [
    # NLU
    "NLUEngine",
    "Intent",
    "Entity",
    "SemanticFrame",
    "IntentType",
    "EntityType",
    # NLG
    "NLGEngine",
    "ResponseTemplate",
    "GenerationStrategy",
    "TextStyle",
    "ResponseType",
    # Dialogue
    "DialogueManager",
    "DialogueState",
    "Turn",
    "DialogueAct",
    # Reasoning
    "LanguageReasoner",
    "Question",
    "Answer",
    "Inference",
    "QuestionType",
]
