"""
Episodic memory for storing time-stamped events and experiences.

Like human autobiographical memory, episodic memory stores:
- What happened
- When it happened
- Who was involved
- Where it happened
- How it felt (emotional valence)
"""

from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import Event, MemoryNode, NodeType, RelationType


class EpisodicMemory:
    """
    Episodic memory system for AGI.

    Stores experiences with temporal context.
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
        retention_days: int = 365,
    ):
        """
        Initialize episodic memory.

        Args:
            graph_store: Memory graph database
            text_encoder: Text embedding encoder
            retention_days: Days to retain memories
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder
        self.retention_days = retention_days
        logger.info(f"Episodic memory initialized (retention: {retention_days} days)")

    def store_event(
        self,
        event: Event,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store an episodic event.

        Args:
            event: Event to store
            importance: Importance score [0, 1]
            tags: Optional tags for indexing

        Returns:
            Node ID
        """
        # Generate embedding
        embedding = self.text_encoder.encode(event.description)

        # Create memory node
        node = MemoryNode(
            node_type=NodeType.EVENT,
            content=event.description,
            embedding=embedding,
            importance=importance,
            metadata={
                "timestamp": event.timestamp.isoformat(),
                "participants": event.participants,
                "location": event.location,
                "outcome": event.outcome,
                "emotional_valence": event.emotional_valence,
            },
            tags=tags or [],
            source="episodic",
        )

        node_id = self.graph_store.add_node(node)

        # Link to previous event (temporal chaining)
        self._link_to_previous_event(node_id, event.timestamp)

        logger.debug(f"Stored episodic event: {event.description[:50]}...")
        return node_id

    def _link_to_previous_event(
        self, node_id: str, timestamp: datetime
    ) -> None:
        """Link event to chronologically previous event."""
        # Find recent events (last 24 hours)
        cursor = self.graph_store.conn.cursor()
        cursor.execute(
            """
            SELECT id, created_at FROM nodes
            WHERE node_type = 'event'
            AND created_at < ?
            ORDER BY created_at DESC
            LIMIT 1
        """,
            (timestamp.isoformat(),),
        )

        row = cursor.fetchone()
        if row:
            prev_id = row["id"]
            self.graph_store.link(
                prev_id, node_id, RelationType.PRECEDES, weight=1.0
            )

    def recall_recent(
        self, limit: int = 10, days: int = 7
    ) -> List[MemoryNode]:
        """
        Recall recent events.

        Args:
            limit: Maximum number of events
            days: Look back this many days

        Returns:
            List of event nodes
        """
        cursor = self.graph_store.conn.cursor()
        cutoff = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff_date = (cutoff - datetime.timedelta(days=days)).isoformat()

        cursor.execute(
            """
            SELECT * FROM nodes
            WHERE node_type = 'event'
            AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (cutoff_date, limit),
        )

        import json
        import numpy as np

        nodes = []
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
            nodes.append(node)

        logger.debug(f"Recalled {len(nodes)} recent events")
        return nodes

    def recall_by_participant(
        self, participant: str, limit: int = 10
    ) -> List[MemoryNode]:
        """
        Recall events involving a specific participant.

        Args:
            participant: Participant name
            limit: Maximum results

        Returns:
            List of events
        """
        # This is a simplified implementation
        # In production, would use JSON queries or indexed metadata
        all_events = self.recall_recent(limit=100, days=365)

        filtered = []
        for event in all_events:
            participants = event.metadata.get("participants", [])
            if participant in participants:
                filtered.append(event)
                if len(filtered) >= limit:
                    break

        return filtered

    def timeline(
        self, start_date: datetime, end_date: datetime
    ) -> List[MemoryNode]:
        """
        Get chronological timeline of events.

        Args:
            start_date: Start of timeline
            end_date: End of timeline

        Returns:
            Chronologically ordered events
        """
        cursor = self.graph_store.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM nodes
            WHERE node_type = 'event'
            AND created_at >= ? AND created_at <= ?
            ORDER BY created_at ASC
        """,
            (start_date.isoformat(), end_date.isoformat()),
        )

        import json
        import numpy as np

        nodes = []
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
            nodes.append(node)

        return nodes
