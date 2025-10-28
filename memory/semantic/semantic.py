"""
Semantic memory for storing facts, concepts, and general knowledge.

Like human semantic memory, this stores:
- Facts (subject-predicate-object triples)
- Concepts and their relationships
- General knowledge independent of specific experiences
"""

from datetime import datetime
from typing import List, Optional

from loguru import logger

from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import Fact, MemoryNode, NodeType, RelationType


class SemanticMemory:
    """
    Semantic memory system for AGI.

    Stores facts and conceptual knowledge with provenance tracking.
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize semantic memory.

        Args:
            graph_store: Memory graph database
            text_encoder: Text embedding encoder
            confidence_threshold: Minimum confidence for facts
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder
        self.confidence_threshold = confidence_threshold
        logger.info(
            f"Semantic memory initialized (confidence >= {confidence_threshold})"
        )

    def store_fact(
        self,
        fact: Fact,
        importance: float = 0.6,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a semantic fact.

        Args:
            fact: Fact to store
            importance: Importance score [0, 1]
            tags: Optional tags

        Returns:
            Node ID
        """
        # Format as natural language
        fact_text = f"{fact.subject} {fact.predicate} {fact.object}"

        # Generate embedding
        embedding = self.text_encoder.encode(fact_text)

        # Create memory node
        node = MemoryNode(
            node_type=NodeType.FACT,
            content=fact_text,
            embedding=embedding,
            importance=importance,
            confidence=fact.confidence,
            metadata={
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "verified": fact.verified,
            },
            tags=tags or [],
            source=fact.source or "semantic",
            provenance=[fact.source] if fact.source else [],
        )

        node_id = self.graph_store.add_node(node)

        # Create entity nodes if they don't exist
        subject_id = self._get_or_create_entity(fact.subject)
        object_id = self._get_or_create_entity(fact.object)

        # Link fact to entities
        self.graph_store.link(
            node_id, subject_id, RelationType.RELATED_TO, weight=1.0
        )
        self.graph_store.link(
            node_id, object_id, RelationType.RELATED_TO, weight=1.0
        )

        logger.debug(f"Stored fact: {fact_text}")
        return node_id

    def _get_or_create_entity(self, entity_name: str) -> str:
        """Get or create an entity node."""
        # Search for existing entity
        existing = self.graph_store.search_text(entity_name, limit=1)
        for node in existing:
            if (
                node.node_type == NodeType.ENTITY
                and node.content.lower() == entity_name.lower()
            ):
                return node.id

        # Create new entity
        embedding = self.text_encoder.encode(entity_name)
        entity_node = MemoryNode(
            node_type=NodeType.ENTITY,
            content=entity_name,
            embedding=embedding,
            importance=0.5,
            source="semantic",
        )

        return self.graph_store.add_node(entity_node)

    def recall_facts_about(
        self, entity: str, limit: int = 10
    ) -> List[MemoryNode]:
        """
        Recall facts about a specific entity.

        Args:
            entity: Entity to query
            limit: Maximum results

        Returns:
            List of fact nodes
        """
        # Find entity node
        entity_id = self._get_or_create_entity(entity)

        # Get neighbors that are facts
        cursor = self.graph_store.conn.cursor()
        cursor.execute(
            """
            SELECT n.*
            FROM nodes n
            JOIN edges e ON n.id = e.source_id
            WHERE e.target_id = ? AND n.node_type = 'fact'
            ORDER BY n.importance DESC
            LIMIT ?
        """,
            (entity_id, limit),
        )

        import json
        import numpy as np

        facts = []
        for row in cursor.fetchall():
            embedding = (
                np.frombuffer(row["embedding"], dtype=np.float32).tolist()
                if row["embedding"]
                else None
            )
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            provenance = json.loads(row["provenance"]) if row["provenance"] else []
            tags = json.loads(row["tags"]) if row["tags"] else []

            node = MemoryNode(
                id=row["id"],
                node_type=NodeType(row["node_type"]),
                content=row["content"],
                embedding=embedding,
                metadata=metadata,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                last_accessed=(
                    datetime.fromisoformat(row["last_accessed"])
                    if row["last_accessed"]
                    else None
                ),
                importance=row["importance"],
                confidence=row["confidence"],
                access_count=row["access_count"],
                source=row["source"],
                provenance=provenance,
                tags=tags,
            )
            facts.append(node)

        return facts

    def verify_fact(self, fact_id: str) -> None:
        """Mark a fact as verified."""
        node = self.graph_store.get_node(fact_id)
        if node:
            node.metadata["verified"] = True
            node.confidence = min(1.0, node.confidence + 0.1)
            self.graph_store.add_node(node)
            logger.debug(f"Verified fact: {fact_id}")

    def contradict(self, fact_id_1: str, fact_id_2: str) -> None:
        """Mark two facts as contradictory."""
        self.graph_store.link(
            fact_id_1, fact_id_2, RelationType.CONTRADICTS, weight=1.0
        )
        logger.debug(f"Marked contradiction: {fact_id_1} <-> {fact_id_2}")

    def support(self, fact_id_1: str, fact_id_2: str) -> None:
        """Mark one fact as supporting another."""
        self.graph_store.link(
            fact_id_1, fact_id_2, RelationType.SUPPORTS, weight=1.0
        )
