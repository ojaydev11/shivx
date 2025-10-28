"""
Data schemas for Semantic Long-Term Memory Graph.

Defines node types, edges, and memory structures using Pydantic.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class NodeType(str, Enum):
    """Types of nodes in the memory graph."""

    EVENT = "event"  # Episodic events
    FACT = "fact"  # Semantic facts
    CONCEPT = "concept"  # Abstract concepts
    ENTITY = "entity"  # Named entities
    TASK = "task"  # Task/goal nodes
    SKILL = "skill"  # Procedural skills
    CODE = "code"  # Code snippets
    CONVERSATION = "conversation"  # Dialogue turns


class RelationType(str, Enum):
    """Types of edges between nodes."""

    CAUSES = "causes"
    PRECEDES = "precedes"
    SIMILAR_TO = "similar_to"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    RELATED_TO = "related_to"
    LEARNED_FROM = "learned_from"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"


class MemoryMode(str, Enum):
    """Memory retrieval modes."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    HYBRID = "hybrid"


class MemoryNode(BaseModel):
    """A node in the memory graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: NodeType
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Temporal information
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None

    # Memory management
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)

    # Provenance
    source: Optional[str] = None
    provenance: List[str] = Field(default_factory=list)

    # Tags for fast filtering
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    def boost_importance(self, factor: float = 1.1) -> None:
        """Increase importance, capped at 1.0."""
        self.importance = min(1.0, self.importance * factor)

    def decay_importance(self, rate: float = 0.95) -> None:
        """Decay importance over time."""
        self.importance *= rate

    def record_access(self) -> None:
        """Record that this node was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.boost_importance(1.05)


class MemoryEdge(BaseModel):
    """An edge connecting two nodes in the memory graph."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = Field(default=1.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Event(BaseModel):
    """An episodic event."""

    timestamp: datetime
    description: str
    participants: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    outcome: Optional[str] = None
    emotional_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)


class Fact(BaseModel):
    """A semantic fact."""

    subject: str
    predicate: str
    object: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source: Optional[str] = None
    verified: bool = False


class Skill(BaseModel):
    """A procedural skill or recipe."""

    name: str
    description: str
    steps: List[str]
    prerequisites: List[str] = Field(default_factory=list)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    execution_count: int = 0


class RetrievalQuery(BaseModel):
    """Query for memory retrieval."""

    query: str
    mode: MemoryMode = MemoryMode.HYBRID
    k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    time_range: Optional[tuple[datetime, datetime]] = None


class RetrievalResult(BaseModel):
    """Result from memory retrieval."""

    nodes: List[MemoryNode]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ConsolidationReport(BaseModel):
    """Report from memory consolidation."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    nodes_merged: int = 0
    nodes_pruned: int = 0
    edges_strengthened: int = 0
    edges_weakened: int = 0
    duration_seconds: float = 0.0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
