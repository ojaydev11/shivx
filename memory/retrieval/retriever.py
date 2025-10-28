"""
Hybrid retriever combining multiple search strategies.

Implements:
- Dense retrieval (vector similarity)
- Sparse retrieval (FTS5 full-text search)
- Graph-based retrieval (traversal)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import (
    MemoryMode,
    MemoryNode,
    NodeType,
    RetrievalQuery,
    RetrievalResult,
)


class HybridRetriever:
    """
    Hybrid retrieval system for memory recall.

    Combines:
    1. Dense (vector similarity) - semantic matching
    2. Sparse (FTS5) - keyword matching
    3. Graph (traversal) - associative recall
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            graph_store: Memory graph database
            text_encoder: Text embedding encoder
            weights: Retrieval weights {dense, sparse, graph}
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder

        # Default weights
        self.weights = weights or {"dense": 0.5, "sparse": 0.3, "graph": 0.2}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Hybrid retriever initialized with weights: {self.weights}")

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Retrieve memories matching the query.

        Args:
            query: Retrieval query

        Returns:
            Retrieval results with scored nodes
        """
        start_time = datetime.utcnow()

        # Encode query
        query_embedding = self.text_encoder.encode(query.query)

        # Get results from each method
        dense_results = self._dense_retrieval(
            query_embedding, k=query.k * 2, filters=query.filters
        )
        sparse_results = self._sparse_retrieval(query.query, k=query.k * 2)

        # Combine scores
        node_scores = self._combine_scores(dense_results, sparse_results)

        # Apply filters
        filtered_nodes = self._apply_filters(
            node_scores,
            query.filters,
            query.min_importance,
            query.time_range,
        )

        # Sort by score and take top-k
        sorted_nodes = sorted(
            filtered_nodes, key=lambda x: x[1], reverse=True
        )[:query.k]

        nodes = [node for node, score in sorted_nodes]
        scores = [score for node, score in sorted_nodes]

        # Optionally add graph-based expansion
        if self.weights.get("graph", 0) > 0 and nodes:
            expanded_nodes = self._graph_expansion(nodes, max_hops=2)
            # Add expanded nodes with lower score
            for exp_node in expanded_nodes[:query.k // 2]:
                if exp_node not in nodes:
                    nodes.append(exp_node)
                    scores.append(0.3)  # Low score for expanded

        duration = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = RetrievalResult(
            nodes=nodes,
            scores=scores,
            metadata={
                "query": query.query,
                "mode": query.mode.value,
                "latency_ms": duration,
                "dense_count": len(dense_results),
                "sparse_count": len(sparse_results),
            },
        )

        logger.debug(
            f"Retrieved {len(nodes)} nodes in {duration:.1f}ms "
            f"(dense: {len(dense_results)}, sparse: {len(sparse_results)})"
        )

        return result

    def _dense_retrieval(
        self,
        query_embedding: List[float],
        k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[MemoryNode, float]]:
        """Dense vector similarity retrieval."""
        results = self.graph_store.search_vector(
            query_embedding, k=k, min_similarity=0.1
        )
        return results

    def _sparse_retrieval(
        self, query: str, k: int = 20
    ) -> List[Tuple[MemoryNode, float]]:
        """Sparse FTS5 full-text retrieval."""
        nodes = self.graph_store.search_text(query, limit=k)
        # Assign scores based on rank (simple approach)
        results = [(node, 1.0 / (i + 1)) for i, node in enumerate(nodes)]
        return results

    def _combine_scores(
        self,
        dense_results: List[Tuple[MemoryNode, float]],
        sparse_results: List[Tuple[MemoryNode, float]],
    ) -> List[Tuple[MemoryNode, float]]:
        """Combine scores from different retrieval methods."""
        # Build score dictionary
        scores_dict: Dict[str, Tuple[MemoryNode, List[float]]] = {}

        # Add dense scores
        for node, score in dense_results:
            if node.id not in scores_dict:
                scores_dict[node.id] = (node, [0.0, 0.0])
            scores_dict[node.id][1][0] = score

        # Add sparse scores
        for node, score in sparse_results:
            if node.id not in scores_dict:
                scores_dict[node.id] = (node, [0.0, 0.0])
            scores_dict[node.id][1][1] = score

        # Compute weighted combination
        combined = []
        for node_id, (node, subscores) in scores_dict.items():
            dense_score, sparse_score = subscores
            final_score = (
                self.weights["dense"] * dense_score +
                self.weights["sparse"] * sparse_score
            )
            # Boost by importance
            final_score *= (0.5 + 0.5 * node.importance)
            combined.append((node, final_score))

        return combined

    def _apply_filters(
        self,
        node_scores: List[Tuple[MemoryNode, float]],
        filters: Optional[Dict],
        min_importance: float,
        time_range: Optional[Tuple[datetime, datetime]],
    ) -> List[Tuple[MemoryNode, float]]:
        """Apply filters to results."""
        filtered = []

        for node, score in node_scores:
            # Importance filter
            if node.importance < min_importance:
                continue

            # Time range filter
            if time_range:
                start, end = time_range
                if node.created_at < start or node.created_at > end:
                    continue

            # Custom filters
            if filters:
                # Node type filter
                if "node_type" in filters:
                    if node.node_type != NodeType(filters["node_type"]):
                        continue

                # Tags filter
                if "tags" in filters:
                    required_tags = set(filters["tags"])
                    node_tags = set(node.tags)
                    if not required_tags.issubset(node_tags):
                        continue

            filtered.append((node, score))

        return filtered

    def _graph_expansion(
        self, seed_nodes: List[MemoryNode], max_hops: int = 2
    ) -> List[MemoryNode]:
        """Expand retrieval via graph traversal."""
        expanded = set()
        frontier = [(node, 0) for node in seed_nodes]

        while frontier:
            current_node, depth = frontier.pop(0)

            if depth >= max_hops:
                continue

            # Get neighbors
            neighbors = self.graph_store.get_neighbors(current_node.id)

            for neighbor, relation, weight in neighbors:
                if neighbor.id not in expanded and neighbor not in seed_nodes:
                    expanded.add(neighbor.id)
                    if depth + 1 < max_hops:
                        frontier.append((neighbor, depth + 1))

        # Retrieve full nodes
        expanded_nodes = []
        for node_id in expanded:
            node = self.graph_store.get_node(node_id)
            if node:
                expanded_nodes.append(node)

        return expanded_nodes

    def recall(
        self,
        query: str,
        k: int = 10,
        mode: MemoryMode = MemoryMode.HYBRID,
        **kwargs,
    ) -> RetrievalResult:
        """
        Convenience method for recall.

        Args:
            query: Query string
            k: Number of results
            mode: Retrieval mode
            **kwargs: Additional query parameters

        Returns:
            Retrieval results
        """
        retrieval_query = RetrievalQuery(
            query=query,
            mode=mode,
            k=k,
            filters=kwargs.get("filters"),
            min_importance=kwargs.get("min_importance", 0.0),
        )
        return self.retrieve(retrieval_query)
