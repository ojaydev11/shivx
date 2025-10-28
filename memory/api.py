"""
Unified Memory API for ShivX AGI.

Provides a simple interface to all memory capabilities.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from memory.consolidation.consolidator import MemoryConsolidator
from memory.encoders.multimodal_encoder import MultimodalEncoder
from memory.encoders.text_encoder import TextEncoder
from memory.episodic.episodic import EpisodicMemory
from memory.graph_store.store import MemoryGraphStore
from memory.procedural.procedural import ProceduralMemory
from memory.retrieval.retriever import HybridRetriever
from memory.schemas import (
    Event,
    Fact,
    MemoryMode,
    RetrievalResult,
    Skill,
)
from memory.semantic.semantic import SemanticMemory


class MemoryAPI:
    """
    Unified API for semantic long-term memory.

    Provides high-level interface to:
    - Store events, facts, skills
    - Retrieve memories via hybrid search
    - Manage memory lifecycle
    """

    def __init__(
        self,
        db_path: str = "./data/memory/graph.db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize memory system.

        Args:
            db_path: Path to graph database
            embedding_model: Text embedding model
            device: Compute device (cpu, cuda, mps)
        """
        # Core components
        self.graph_store = MemoryGraphStore(db_path=db_path)
        self.text_encoder = TextEncoder(model_name=embedding_model, device=device)
        self.multimodal_encoder = MultimodalEncoder(
            text_model=embedding_model, device=device
        )

        # Memory types
        self.episodic = EpisodicMemory(
            graph_store=self.graph_store,
            text_encoder=self.text_encoder,
        )
        self.semantic = SemanticMemory(
            graph_store=self.graph_store,
            text_encoder=self.text_encoder,
        )
        self.procedural = ProceduralMemory(
            graph_store=self.graph_store,
            text_encoder=self.text_encoder,
        )

        # Retrieval
        self.retriever = HybridRetriever(
            graph_store=self.graph_store,
            text_encoder=self.text_encoder,
        )

        # Consolidation
        self.consolidator = MemoryConsolidator(
            graph_store=self.graph_store,
            text_encoder=self.text_encoder,
        )

        logger.info("Memory API initialized")

    # ========================================================================
    # High-level storage methods
    # ========================================================================

    def store_event(self, description: str, **kwargs) -> str:
        """
        Store an episodic event.

        Args:
            description: Event description
            **kwargs: participants, location, outcome, etc.

        Returns:
            Node ID
        """
        event = Event(
            timestamp=kwargs.get("timestamp", datetime.utcnow()),
            description=description,
            participants=kwargs.get("participants", []),
            location=kwargs.get("location"),
            outcome=kwargs.get("outcome"),
            emotional_valence=kwargs.get("emotional_valence"),
        )
        return self.episodic.store_event(
            event,
            importance=kwargs.get("importance", 0.5),
            tags=kwargs.get("tags"),
        )

    def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        **kwargs,
    ) -> str:
        """
        Store a semantic fact.

        Args:
            subject: Subject of fact
            predicate: Relationship/property
            object: Object/value
            **kwargs: confidence, source, etc.

        Returns:
            Node ID
        """
        fact = Fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=kwargs.get("confidence", 0.8),
            source=kwargs.get("source"),
            verified=kwargs.get("verified", False),
        )
        return self.semantic.store_fact(
            fact,
            importance=kwargs.get("importance", 0.6),
            tags=kwargs.get("tags"),
        )

    def store_skill(
        self,
        name: str,
        description: str,
        steps: List[str],
        **kwargs,
    ) -> str:
        """
        Store a procedural skill.

        Args:
            name: Skill name
            description: Description
            steps: Step-by-step procedure
            **kwargs: prerequisites, etc.

        Returns:
            Node ID
        """
        skill = Skill(
            name=name,
            description=description,
            steps=steps,
            prerequisites=kwargs.get("prerequisites", []),
            success_rate=kwargs.get("success_rate", 0.0),
            execution_count=kwargs.get("execution_count", 0),
        )
        return self.procedural.store_skill(
            skill,
            importance=kwargs.get("importance", 0.7),
            tags=kwargs.get("tags"),
        )

    def store_code(
        self,
        code: str,
        description: str,
        language: str = "python",
        **kwargs,
    ) -> str:
        """Store a code snippet."""
        return self.procedural.store_code(
            code=code,
            description=description,
            language=language,
            tags=kwargs.get("tags"),
        )

    # ========================================================================
    # Retrieval methods
    # ========================================================================

    def recall(
        self,
        query: str,
        k: int = 10,
        mode: MemoryMode = MemoryMode.HYBRID,
        **kwargs,
    ) -> RetrievalResult:
        """
        Recall memories matching query.

        Args:
            query: Query string
            k: Number of results
            mode: Retrieval mode (episodic, semantic, procedural, hybrid)
            **kwargs: Additional filters

        Returns:
            Retrieval results
        """
        return self.retriever.recall(query=query, k=k, mode=mode, **kwargs)

    def recall_recent_events(self, days: int = 7, limit: int = 10):
        """Recall recent episodic events."""
        return self.episodic.recall_recent(limit=limit, days=days)

    def recall_facts_about(self, entity: str, limit: int = 10):
        """Recall facts about an entity."""
        return self.semantic.recall_facts_about(entity, limit=limit)

    def recall_skill(self, skill_name: str):
        """Recall a specific skill."""
        return self.procedural.recall_skill(skill_name)

    # ========================================================================
    # Management methods
    # ========================================================================

    def consolidate(self):
        """Run memory consolidation."""
        return self.consolidator.consolidate()

    def forget(self, node_id: str) -> bool:
        """Forget (delete) a memory node."""
        return self.graph_store.forget(node_id)

    def export_graph(self, filepath: str) -> None:
        """Export memory graph to JSON."""
        self.graph_store.export_graph(filepath)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_nodes": self.graph_store.count_nodes(),
            "total_edges": self.graph_store.count_edges(),
            "node_types": {
                "events": self.graph_store.count_nodes(node_type="event"),
                "facts": self.graph_store.count_nodes(node_type="fact"),
                "skills": self.graph_store.count_nodes(node_type="skill"),
                "entities": self.graph_store.count_nodes(node_type="entity"),
                "code": self.graph_store.count_nodes(node_type="code"),
            },
        }

    def close(self) -> None:
        """Close database connections."""
        self.graph_store.close()
        logger.info("Memory API closed")
