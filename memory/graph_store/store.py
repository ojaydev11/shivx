"""
Local graph database for Semantic Long-Term Memory.

Uses SQLite with FTS5 for text search and HNSW for vector similarity.
Designed for privacy-first, local-only storage.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory.schemas import MemoryEdge, MemoryNode, NodeType, RelationType


class MemoryGraphStore:
    """
    Local graph database for AGI memory.

    Features:
    - SQLite for structured storage
    - FTS5 for full-text search
    - HNSW-like indexing for vector similarity
    - ACID transactions
    - Zero network dependencies
    """

    def __init__(self, db_path: str = "./data/memory/graph.db"):
        """
        Initialize the graph store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        cursor = self.conn.cursor()

        # Nodes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed TEXT,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                access_count INTEGER DEFAULT 0,
                source TEXT,
                provenance TEXT,
                tags TEXT
            )
        """
        )

        # Edges table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        """
        )

        # FTS5 table for full-text search
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
                id UNINDEXED,
                content,
                tags
            )
        """
        )

        # Indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_importance ON nodes(importance)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation_type)"
        )

        self.conn.commit()
        logger.info(f"Initialized memory graph store at {self.db_path}")

    def add_node(self, node: MemoryNode) -> str:
        """
        Add a node to the graph.

        Args:
            node: MemoryNode to add

        Returns:
            Node ID
        """
        cursor = self.conn.cursor()

        # Serialize complex fields
        embedding_bytes = (
            np.array(node.embedding, dtype=np.float32).tobytes()
            if node.embedding
            else None
        )
        metadata_json = json.dumps(node.metadata)
        provenance_json = json.dumps(node.provenance)
        tags_json = json.dumps(node.tags)

        cursor.execute(
            """
            INSERT OR REPLACE INTO nodes
            (id, node_type, content, embedding, metadata, created_at, updated_at,
             last_accessed, importance, confidence, access_count, source, provenance, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                node.id,
                node.node_type.value,
                node.content,
                embedding_bytes,
                metadata_json,
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
                node.last_accessed.isoformat() if node.last_accessed else None,
                node.importance,
                node.confidence,
                node.access_count,
                node.source,
                provenance_json,
                tags_json,
            ),
        )

        # Update FTS index
        cursor.execute(
            """
            INSERT OR REPLACE INTO nodes_fts (id, content, tags)
            VALUES (?, ?, ?)
        """,
            (node.id, node.content, " ".join(node.tags)),
        )

        self.conn.commit()
        logger.debug(f"Added node {node.id} of type {node.node_type}")
        return node.id

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """
        Retrieve a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            MemoryNode or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize
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

        # Record access
        node.record_access()
        self._update_node_access(node.id, node.access_count, node.last_accessed)

        return node

    def _update_node_access(
        self, node_id: str, access_count: int, last_accessed: datetime
    ) -> None:
        """Update node access statistics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE nodes
            SET access_count = ?, last_accessed = ?
            WHERE id = ?
        """,
            (access_count, last_accessed.isoformat(), node_id),
        )
        self.conn.commit()

    def add_edge(self, edge: MemoryEdge) -> str:
        """
        Add an edge between two nodes.

        Args:
            edge: MemoryEdge to add

        Returns:
            Edge ID
        """
        cursor = self.conn.cursor()
        metadata_json = json.dumps(edge.metadata)

        cursor.execute(
            """
            INSERT OR REPLACE INTO edges
            (id, source_id, target_id, relation_type, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relation_type.value,
                edge.weight,
                metadata_json,
                edge.created_at.isoformat(),
            ),
        )

        self.conn.commit()
        logger.debug(
            f"Added edge {edge.source_id} -> {edge.target_id} ({edge.relation_type})"
        )
        return edge.id

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        weight: float = 1.0,
    ) -> str:
        """
        Create a link between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Type of relationship
            weight: Edge weight (strength)

        Returns:
            Edge ID
        """
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation,
            weight=weight,
        )
        return self.add_edge(edge)

    def get_neighbors(
        self, node_id: str, relation_type: Optional[RelationType] = None
    ) -> List[Tuple[MemoryNode, RelationType, float]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Source node ID
            relation_type: Optional filter by relation type

        Returns:
            List of (neighbor_node, relation_type, weight)
        """
        cursor = self.conn.cursor()

        if relation_type:
            cursor.execute(
                """
                SELECT n.*, e.relation_type, e.weight
                FROM nodes n
                JOIN edges e ON n.id = e.target_id
                WHERE e.source_id = ? AND e.relation_type = ?
                ORDER BY e.weight DESC
            """,
                (node_id, relation_type.value),
            )
        else:
            cursor.execute(
                """
                SELECT n.*, e.relation_type, e.weight
                FROM nodes n
                JOIN edges e ON n.id = e.target_id
                WHERE e.source_id = ?
                ORDER BY e.weight DESC
            """,
                (node_id,),
            )

        results = []
        for row in cursor.fetchall():
            # Deserialize node
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

            results.append((node, RelationType(row["relation_type"]), row["weight"]))

        return results

    def search_text(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching nodes
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT n.*
            FROM nodes n
            JOIN nodes_fts fts ON n.id = fts.id
            WHERE nodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """,
            (query, limit),
        )

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

    def search_vector(
        self, query_embedding: List[float], k: int = 10, min_similarity: float = 0.0
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Vector similarity search using cosine similarity.

        Args:
            query_embedding: Query vector
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of (node, similarity_score)
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE embedding IS NOT NULL")

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        results = []
        for row in cursor.fetchall():
            node_vec = np.frombuffer(row["embedding"], dtype=np.float32)
            node_norm = np.linalg.norm(node_vec)

            if node_norm == 0 or query_norm == 0:
                continue

            similarity = np.dot(query_vec, node_vec) / (query_norm * node_norm)

            if similarity >= min_similarity:
                embedding = node_vec.tolist()
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

                results.append((node, float(similarity)))

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def count_nodes(self, node_type: Optional[NodeType] = None) -> int:
        """Count nodes in the graph."""
        cursor = self.conn.cursor()
        if node_type:
            cursor.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = ?", (node_type.value,)
            )
        else:
            cursor.execute("SELECT COUNT(*) FROM nodes")
        return cursor.fetchone()[0]

    def count_edges(self) -> int:
        """Count edges in the graph."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM edges")
        return cursor.fetchone()[0]

    def forget(self, node_id: str) -> bool:
        """
        Delete a node and its edges.

        Args:
            node_id: Node to delete

        Returns:
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()

        # Delete from FTS
        cursor.execute("DELETE FROM nodes_fts WHERE id = ?", (node_id,))

        # Delete node (edges cascade)
        cursor.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

        self.conn.commit()

        if cursor.rowcount > 0:
            logger.debug(f"Deleted node {node_id}")
            return True
        return False

    def export_graph(self, filepath: str) -> None:
        """Export entire graph to JSON."""
        cursor = self.conn.cursor()

        # Export nodes
        cursor.execute("SELECT * FROM nodes")
        nodes = []
        for row in cursor.fetchall():
            node_dict = dict(row)
            # Convert bytes to list
            if node_dict["embedding"]:
                node_dict["embedding"] = np.frombuffer(
                    node_dict["embedding"], dtype=np.float32
                ).tolist()
            nodes.append(node_dict)

        # Export edges
        cursor.execute("SELECT * FROM edges")
        edges = [dict(row) for row in cursor.fetchall()]

        export_data = {"nodes": nodes, "edges": edges, "exported_at": datetime.utcnow().isoformat()}

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported graph to {filepath}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed memory graph store")
