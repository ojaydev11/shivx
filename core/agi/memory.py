"""
AGI Memory Module
Episodic & Semantic Memory System

Provides:
- Episodic Memory: Personal experiences and events (Vector DB)
- Semantic Memory: Facts and concepts (Knowledge Graph)
- Working Memory: Short-term context
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Episodic memory entry"""
    episode_id: str
    timestamp: datetime
    event_type: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Concept:
    """Semantic memory concept"""
    concept_id: str
    name: str
    description: str
    properties: Dict[str, Any]
    relations: List[tuple]  # (relation_type, target_concept_id)


class VectorMemory:
    """
    Episodic Memory using Vector Database
    Stores experiences with semantic embeddings for retrieval
    """

    def __init__(self, dimension: int = 384):
        """
        Initialize vector memory

        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.episodes: List[Episode] = []

        # Try to import vector DB libraries
        self.faiss_available = False
        self.chromadb_available = False

        try:
            import faiss
            self.faiss = faiss
            self.index = faiss.IndexFlatL2(dimension)
            self.faiss_available = True
            logger.info("FAISS integration available")
        except ImportError:
            logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
            self.index = None

        try:
            import chromadb
            self.chromadb = chromadb
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.create_collection("episodes")
            self.chromadb_available = True
            logger.info("ChromaDB integration available")
        except ImportError:
            logger.warning("ChromaDB not installed. Install with: pip install chromadb")

        logger.info(f"Vector memory initialized: dimension={dimension}")

    async def store_episode(
        self,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Episode:
        """
        Store an episode in memory

        Args:
            event_type: Type of event (trade, analysis, conversation, etc.)
            content: Event content
            metadata: Additional metadata

        Returns:
            Stored episode
        """
        episode_id = f"ep_{datetime.utcnow().timestamp()}"

        # Generate embedding (simplified - would use sentence transformer)
        embedding = self._generate_embedding(content)

        episode = Episode(
            episode_id=episode_id,
            timestamp=datetime.utcnow(),
            event_type=event_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Store in vector index
        if self.faiss_available and self.index is not None:
            self.index.add(np.array([embedding], dtype=np.float32))

        # Store in ChromaDB if available
        if self.chromadb_available:
            self.collection.add(
                ids=[episode_id],
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[{"event_type": event_type, **(metadata or {})}]
            )

        self.episodes.append(episode)

        logger.debug(f"Episode stored: {episode_id}")
        return episode

    async def recall(
        self,
        query: str,
        k: int = 5,
        event_type: Optional[str] = None
    ) -> List[Episode]:
        """
        Recall similar episodes

        Args:
            query: Query text
            k: Number of results to return
            event_type: Filter by event type

        Returns:
            List of similar episodes
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        if self.chromadb_available:
            # Use ChromaDB for retrieval
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where={"event_type": event_type} if event_type else None
            )

            # Return matching episodes
            episode_ids = results["ids"][0]
            return [ep for ep in self.episodes if ep.episode_id in episode_ids][:k]

        elif self.faiss_available and self.index is not None:
            # Use FAISS for retrieval
            query_vector = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_vector, k)

            # Filter by event type if specified
            results = [self.episodes[idx] for idx in indices[0] if idx < len(self.episodes)]
            if event_type:
                results = [ep for ep in results if ep.event_type == event_type]

            return results[:k]

        else:
            # Fallback: simple text matching
            return self._simple_recall(query, k, event_type)

    def _simple_recall(self, query: str, k: int, event_type: Optional[str]) -> List[Episode]:
        """Simple text-based recall (fallback)"""
        query_lower = query.lower()

        # Score episodes by keyword overlap
        scored_episodes = []
        for ep in self.episodes:
            if event_type and ep.event_type != event_type:
                continue

            score = sum(1 for word in query_lower.split() if word in ep.content.lower())
            scored_episodes.append((score, ep))

        # Sort by score and return top k
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for score, ep in scored_episodes[:k]]

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding (simplified)"""
        # In production, would use sentence-transformers
        # For now, use simple hash-based embedding
        words = text.lower().split()
        embedding = np.zeros(self.dimension)

        for i, word in enumerate(words[:self.dimension]):
            embedding[i] = (hash(word) % 1000) / 1000.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class KnowledgeGraph:
    """
    Semantic Memory using Knowledge Graph
    Stores facts, concepts, and their relationships
    """

    def __init__(self):
        """Initialize knowledge graph"""
        self.concepts: Dict[str, Concept] = {}
        self.data_dir = Path("data/agi/knowledge")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Try to import graph database
        self.neo4j_available = False
        try:
            from neo4j import GraphDatabase
            self.neo4j = GraphDatabase
            self.neo4j_available = True
            logger.info("Neo4j integration available")
        except ImportError:
            logger.warning("Neo4j not installed. Install with: pip install neo4j")

        logger.info("Knowledge graph initialized")

    async def add_concept(
        self,
        name: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Concept:
        """
        Add a concept to knowledge graph

        Args:
            name: Concept name
            description: Description
            properties: Additional properties

        Returns:
            Created concept
        """
        concept_id = f"concept_{len(self.concepts)}"

        concept = Concept(
            concept_id=concept_id,
            name=name,
            description=description,
            properties=properties or {},
            relations=[]
        )

        self.concepts[concept_id] = concept

        logger.debug(f"Concept added: {name}")
        return concept

    async def add_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str
    ):
        """
        Add relation between concepts

        Args:
            source_id: Source concept ID
            relation_type: Relation type (is_a, has_property, causes, etc.)
            target_id: Target concept ID
        """
        if source_id in self.concepts:
            self.concepts[source_id].relations.append((relation_type, target_id))
            logger.debug(f"Relation added: {source_id} -{relation_type}-> {target_id}")

    async def query(
        self,
        concept_name: str,
        relation_type: Optional[str] = None
    ) -> List[Concept]:
        """
        Query knowledge graph

        Args:
            concept_name: Concept to query
            relation_type: Filter by relation type

        Returns:
            Related concepts
        """
        # Find concept
        source_concept = None
        for concept in self.concepts.values():
            if concept.name.lower() == concept_name.lower():
                source_concept = concept
                break

        if not source_concept:
            return []

        # Get related concepts
        related = []
        for rel_type, target_id in source_concept.relations:
            if relation_type and rel_type != relation_type:
                continue

            if target_id in self.concepts:
                related.append(self.concepts[target_id])

        return related

    async def save(self):
        """Save knowledge graph to disk"""
        save_path = self.data_dir / "knowledge_graph.json"

        data = {
            "concepts": {
                cid: {
                    "name": c.name,
                    "description": c.description,
                    "properties": c.properties,
                    "relations": c.relations
                }
                for cid, c in self.concepts.items()
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Knowledge graph saved: {save_path}")


class MemoryModule:
    """
    Complete memory system combining episodic and semantic memory
    """

    def __init__(self):
        """Initialize memory module"""
        self.episodic = VectorMemory()
        self.semantic = KnowledgeGraph()
        self.working_memory: List[Dict[str, Any]] = []  # Short-term context
        self.max_working_memory = 10

        logger.info("Memory module initialized")

    async def remember(
        self,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Episode:
        """
        Store experience in episodic memory

        Args:
            event_type: Event type
            content: Event content
            metadata: Additional metadata

        Returns:
            Stored episode
        """
        episode = await self.episodic.store_episode(event_type, content, metadata)

        # Add to working memory
        self.working_memory.append({
            "type": "episode",
            "episode_id": episode.episode_id,
            "content": content,
            "timestamp": episode.timestamp
        })

        # Trim working memory
        if len(self.working_memory) > self.max_working_memory:
            self.working_memory = self.working_memory[-self.max_working_memory:]

        return episode

    async def recall_similar(self, query: str, k: int = 5) -> List[Episode]:
        """
        Recall similar experiences

        Args:
            query: Query text
            k: Number of results

        Returns:
            Similar episodes
        """
        return await self.episodic.recall(query, k)

    async def learn_concept(
        self,
        name: str,
        description: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Concept:
        """
        Learn new concept (semantic memory)

        Args:
            name: Concept name
            description: Description
            properties: Properties

        Returns:
            Learned concept
        """
        return await self.semantic.add_concept(name, description, properties)

    async def get_context(self) -> List[Dict[str, Any]]:
        """
        Get current working memory context

        Returns:
            Working memory contents
        """
        return self.working_memory.copy()

    def get_capabilities(self) -> Dict[str, bool]:
        """Get memory capabilities status"""
        return {
            "episodic_memory": True,
            "semantic_memory": True,
            "working_memory": True,
            "vector_search": self.episodic.faiss_available or self.episodic.chromadb_available,
            "knowledge_graph": True
        }
