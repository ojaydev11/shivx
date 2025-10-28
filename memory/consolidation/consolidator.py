"""
Memory consolidation system.

Performs background maintenance:
- Merges similar nodes
- Strengthens important connections
- Decays unimportant memories
- Prunes low-value nodes
"""

from datetime import datetime, timedelta
from typing import List, Tuple

from loguru import logger

from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import ConsolidationReport, MemoryNode


class MemoryConsolidator:
    """
    Consolidates and maintains the memory graph.

    Like sleep consolidation in humans, this process:
    - Strengthens important memories
    - Weakens unimportant connections
    - Merges duplicate/similar memories
    - Prunes noise
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
        merge_threshold: float = 0.85,
        decay_rate: float = 0.95,
        min_importance: float = 0.1,
    ):
        """
        Initialize consolidator.

        Args:
            graph_store: Memory graph database
            text_encoder: Text encoder for similarity
            merge_threshold: Similarity threshold for merging
            decay_rate: Importance decay rate per cycle
            min_importance: Prune nodes below this importance
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder
        self.merge_threshold = merge_threshold
        self.decay_rate = decay_rate
        self.min_importance = min_importance
        logger.info(
            f"Memory consolidator initialized "
            f"(merge>={merge_threshold}, decay={decay_rate})"
        )

    def consolidate(self) -> ConsolidationReport:
        """
        Run full consolidation cycle.

        Returns:
            Consolidation report with statistics
        """
        start_time = datetime.utcnow()
        logger.info("Starting memory consolidation...")

        report = ConsolidationReport()

        # Step 1: Decay importance
        decayed = self._decay_importance()
        logger.debug(f"Decayed importance for {decayed} nodes")

        # Step 2: Merge similar nodes
        merged = self._merge_similar_nodes()
        report.nodes_merged = merged
        logger.debug(f"Merged {merged} similar nodes")

        # Step 3: Strengthen/weaken edges
        strengthened, weakened = self._adjust_edge_weights()
        report.edges_strengthened = strengthened
        report.edges_weakened = weakened
        logger.debug(f"Adjusted edges: +{strengthened}, -{weakened}")

        # Step 4: Prune low-importance nodes
        pruned = self._prune_nodes()
        report.nodes_pruned = pruned
        logger.debug(f"Pruned {pruned} low-importance nodes")

        # Calculate duration
        end_time = datetime.utcnow()
        report.duration_seconds = (end_time - start_time).total_seconds()

        logger.info(
            f"Consolidation complete: "
            f"merged={report.nodes_merged}, "
            f"pruned={report.nodes_pruned}, "
            f"duration={report.duration_seconds:.1f}s"
        )

        return report

    def _decay_importance(self) -> int:
        """Apply importance decay to all nodes."""
        cursor = self.graph_store.conn.cursor()
        cursor.execute(
            """
            UPDATE nodes
            SET importance = importance * ?
            WHERE importance > ?
        """,
            (self.decay_rate, self.min_importance),
        )
        self.graph_store.conn.commit()
        return cursor.rowcount

    def _merge_similar_nodes(self) -> int:
        """Find and merge highly similar nodes."""
        merged_count = 0

        # Get all nodes with embeddings
        cursor = self.graph_store.conn.cursor()
        cursor.execute(
            """
            SELECT id, node_type, embedding, importance
            FROM nodes
            WHERE embedding IS NOT NULL
            ORDER BY importance DESC
            LIMIT 1000
        """
        )

        import numpy as np

        nodes_data = []
        for row in cursor.fetchall():
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            nodes_data.append(
                {
                    "id": row["id"],
                    "type": row["node_type"],
                    "embedding": embedding,
                    "importance": row["importance"],
                }
            )

        # Find similar pairs
        for i in range(len(nodes_data)):
            for j in range(i + 1, len(nodes_data)):
                node_i = nodes_data[i]
                node_j = nodes_data[j]

                # Only merge same type
                if node_i["type"] != node_j["type"]:
                    continue

                # Compute similarity
                emb_i = node_i["embedding"]
                emb_j = node_j["embedding"]
                similarity = np.dot(emb_i, emb_j) / (
                    np.linalg.norm(emb_i) * np.linalg.norm(emb_j)
                )

                if similarity >= self.merge_threshold:
                    # Merge j into i (keep more important one)
                    if node_i["importance"] >= node_j["importance"]:
                        self._merge_nodes(node_i["id"], node_j["id"])
                    else:
                        self._merge_nodes(node_j["id"], node_i["id"])
                    merged_count += 1

        return merged_count

    def _merge_nodes(self, keep_id: str, delete_id: str) -> None:
        """Merge two nodes, keeping one and deleting the other."""
        # Transfer edges from deleted node to kept node
        cursor = self.graph_store.conn.cursor()

        # Transfer outgoing edges
        cursor.execute(
            """
            UPDATE edges
            SET source_id = ?
            WHERE source_id = ?
        """,
            (keep_id, delete_id),
        )

        # Transfer incoming edges
        cursor.execute(
            """
            UPDATE edges
            SET target_id = ?
            WHERE target_id = ?
        """,
            (keep_id, delete_id),
        )

        # Boost importance of kept node
        cursor.execute(
            """
            UPDATE nodes
            SET importance = MIN(1.0, importance * 1.1)
            WHERE id = ?
        """,
            (keep_id,),
        )

        # Delete the merged node
        self.graph_store.forget(delete_id)

        self.graph_store.conn.commit()

    def _adjust_edge_weights(self) -> Tuple[int, int]:
        """Strengthen frequently used edges, weaken others."""
        cursor = self.graph_store.conn.cursor()

        # Get nodes with recent access
        recent_cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()

        cursor.execute(
            """
            SELECT id, access_count FROM nodes
            WHERE last_accessed >= ?
        """,
            (recent_cutoff,),
        )

        active_nodes = {row["id"]: row["access_count"] for row in cursor.fetchall()}

        strengthened = 0
        weakened = 0

        # Strengthen edges between active nodes
        for node_id in active_nodes:
            cursor.execute(
                """
                UPDATE edges
                SET weight = MIN(2.0, weight * 1.1)
                WHERE source_id = ? OR target_id = ?
            """,
                (node_id, node_id),
            )
            strengthened += cursor.rowcount

        # Weaken old edges
        old_cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        cursor.execute(
            """
            UPDATE edges
            SET weight = MAX(0.1, weight * 0.9)
            WHERE created_at < ?
        """,
            (old_cutoff,),
        )
        weakened += cursor.rowcount

        self.graph_store.conn.commit()
        return strengthened, weakened

    def _prune_nodes(self) -> int:
        """Remove nodes with very low importance."""
        cursor = self.graph_store.conn.cursor()

        # Find candidates
        cursor.execute(
            """
            SELECT id FROM nodes
            WHERE importance < ?
            AND access_count < 2
            AND created_at < datetime('now', '-30 days')
        """,
            (self.min_importance,),
        )

        candidates = [row["id"] for row in cursor.fetchall()]

        # Delete candidates
        for node_id in candidates:
            self.graph_store.forget(node_id)

        return len(candidates)

    def optimize_database(self) -> None:
        """Optimize database storage."""
        logger.info("Optimizing database...")
        cursor = self.graph_store.conn.cursor()
        cursor.execute("VACUUM")
        cursor.execute("ANALYZE")
        self.graph_store.conn.commit()
        logger.info("Database optimized")
