"""
Semantic Long-Term Memory Graph (SLMG) for ShivX AGI

This module provides human-like memory capabilities:
- Episodic: time-stamped events and experiences
- Semantic: facts, concepts, and knowledge
- Procedural: skills, procedures, and how-to knowledge

Memory is persistent, consolidated, and retrieved efficiently.
"""

from .graph_store.store import MemoryGraphStore
from .retrieval.retriever import HybridRetriever
from .episodic.episodic import EpisodicMemory
from .semantic.semantic import SemanticMemory
from .procedural.procedural import ProceduralMemory

__all__ = [
    "MemoryGraphStore",
    "HybridRetriever",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
]

__version__ = "1.0.0"
