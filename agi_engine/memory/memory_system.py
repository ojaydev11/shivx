"""
Comprehensive Memory System for AGI

Supports multiple memory types and sophisticated retrieval mechanisms.
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import sqlite3
from pathlib import Path
import hashlib


class MemoryType(str, Enum):
    """Types of memory"""
    WORKING = "working"  # Short-term, currently active
    EPISODIC = "episodic"  # Experiences, events
    SEMANTIC = "semantic"  # Facts, knowledge
    PROCEDURAL = "procedural"  # Skills, how-to
    DECLARATIVE = "declarative"  # Long-term facts


@dataclass
class Memory:
    """A single memory item"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0  # 0-1
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    associations: List[str] = field(default_factory=list)  # Links to other memories
    context: Dict[str, Any] = field(default_factory=dict)
    decay_rate: float = 0.01  # How fast memory fades

    def access(self):
        """Record memory access"""
        self.access_count += 1
        self.last_accessed = time.time()
        # Accessing strengthens the memory
        self.importance = min(1.0, self.importance * 1.1)

    def strength(self) -> float:
        """
        Calculate current memory strength

        Combines:
        - Base importance
        - Access frequency
        - Recency
        - Decay over time
        """
        # Time since creation
        age = time.time() - self.timestamp
        decay = max(0.0, 1.0 - (age * self.decay_rate))

        # Time since last access
        recency_bonus = 1.0 / (1.0 + (time.time() - self.last_accessed) / 3600.0)

        # Access frequency bonus
        frequency_bonus = min(1.0, self.access_count / 10.0)

        strength = self.importance * decay * (1.0 + recency_bonus + frequency_bonus) / 3.0
        return min(1.0, strength)


class MemorySystem:
    """
    Unified memory system for AGI

    Features:
    - Multiple memory types (working, episodic, semantic, procedural)
    - Memory consolidation (short-term → long-term)
    - Context-aware retrieval
    - Associative memory (links between memories)
    - Importance weighting
    - Memory decay
    - Persistent storage
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "data/agi_memory.db"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # In-memory indices for fast access
        self.working_memories: Dict[str, Memory] = {}
        self.memory_by_type: Dict[MemoryType, Set[str]] = {
            t: set() for t in MemoryType
        }
        self.memory_by_tag: Dict[str, Set[str]] = {}

        # Memory capacity limits
        self.working_memory_capacity = 7  # Miller's law: 7±2 items
        self.consolidation_threshold = 0.7  # When to move to long-term

        # Load recent working memories
        self._load_recent_working_memories()

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL NOT NULL,
                access_count INTEGER NOT NULL,
                last_accessed REAL NOT NULL,
                tags TEXT,
                associations TEXT,
                context TEXT,
                decay_rate REAL NOT NULL
            )
        """)

        # Index for fast retrieval
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memories(timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_importance
            ON memories(importance DESC)
        """)

        conn.commit()
        conn.close()

    def _load_recent_working_memories(self):
        """Load recent working memories from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT * FROM memories
            WHERE memory_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (MemoryType.WORKING.value, self.working_memory_capacity))

        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            self.working_memories[memory.memory_id] = memory
            self.memory_by_type[memory.memory_type].add(memory.memory_id)

        conn.close()

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object"""
        return Memory(
            memory_id=row[0],
            memory_type=MemoryType(row[1]),
            content=json.loads(row[2]),
            timestamp=row[3],
            importance=row[4],
            access_count=row[5],
            last_accessed=row[6],
            tags=set(json.loads(row[7])) if row[7] else set(),
            associations=json.loads(row[8]) if row[8] else [],
            context=json.loads(row[9]) if row[9] else {},
            decay_rate=row[10]
        )

    def _memory_to_row(self, memory: Memory) -> tuple:
        """Convert Memory object to database row"""
        return (
            memory.memory_id,
            memory.memory_type.value,
            json.dumps(memory.content),
            memory.timestamp,
            memory.importance,
            memory.access_count,
            memory.last_accessed,
            json.dumps(list(memory.tags)),
            json.dumps(memory.associations),
            json.dumps(memory.context),
            memory.decay_rate
        )

    def store(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.WORKING,
        importance: float = 1.0,
        tags: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Store a new memory

        Returns the created Memory object
        """
        # Generate memory ID
        content_str = json.dumps(content, sort_keys=True)
        memory_id = hashlib.md5(
            f"{content_str}:{time.time()}".encode()
        ).hexdigest()[:16]

        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            tags=tags or set(),
            context=context or {}
        )

        # Store in appropriate location
        if memory_type == MemoryType.WORKING:
            # Add to working memory
            if len(self.working_memories) >= self.working_memory_capacity:
                # Consolidate or forget
                self._manage_working_memory()

            self.working_memories[memory_id] = memory
        else:
            # Store directly in long-term (database)
            self._persist_memory(memory)

        # Update indices
        self.memory_by_type[memory_type].add(memory_id)
        for tag in memory.tags:
            if tag not in self.memory_by_tag:
                self.memory_by_tag[tag] = set()
            self.memory_by_tag[tag].add(memory_id)

        return memory

    def _persist_memory(self, memory: Memory):
        """Persist memory to database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._memory_to_row(memory))
        conn.commit()
        conn.close()

    def _manage_working_memory(self):
        """
        Manage working memory capacity

        Consolidates important memories to long-term or forgets weak ones
        """
        if not self.working_memories:
            return

        # Sort by strength
        memories = list(self.working_memories.values())
        memories.sort(key=lambda m: m.strength(), reverse=True)

        # Keep strongest, consolidate or forget rest
        to_keep = memories[:self.working_memory_capacity]
        to_process = memories[self.working_memory_capacity:]

        for memory in to_process:
            if memory.importance >= self.consolidation_threshold:
                # Important enough for long-term memory
                self._consolidate_to_long_term(memory)
            # else: forget (don't persist)

            # Remove from working memory
            del self.working_memories[memory.memory_id]

    def _consolidate_to_long_term(self, memory: Memory):
        """Move memory from working to long-term storage"""
        # Change type to appropriate long-term type
        if "event" in memory.tags or "experience" in memory.tags:
            memory.memory_type = MemoryType.EPISODIC
        elif "skill" in memory.tags or "procedure" in memory.tags:
            memory.memory_type = MemoryType.PROCEDURAL
        elif "fact" in memory.tags or "knowledge" in memory.tags:
            memory.memory_type = MemoryType.SEMANTIC
        else:
            memory.memory_type = MemoryType.DECLARATIVE

        # Persist to database
        self._persist_memory(memory)

        # Update indices
        self.memory_by_type[memory.memory_type].add(memory.memory_id)

    def retrieve(
        self,
        query: Optional[Dict[str, Any]] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 10,
        min_strength: float = 0.1
    ) -> List[Memory]:
        """
        Retrieve memories based on query

        Supports:
        - Type-based retrieval
        - Tag-based retrieval
        - Content similarity
        - Strength thresholding
        """
        candidates = set()

        # Filter by type
        if memory_type:
            candidates = self.memory_by_type.get(memory_type, set()).copy()
        else:
            for mem_set in self.memory_by_type.values():
                candidates.update(mem_set)

        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self.memory_by_tag:
                    tag_matches.update(self.memory_by_tag[tag])
            candidates = candidates.intersection(tag_matches) if tag_matches else candidates

        # Load memories
        memories = []

        # From working memory
        for mem_id in candidates:
            if mem_id in self.working_memories:
                mem = self.working_memories[mem_id]
                if mem.strength() >= min_strength:
                    memories.append(mem)

        # From database (if needed)
        if len(memories) < limit:
            memories.extend(self._retrieve_from_db(
                memory_type=memory_type,
                tags=tags,
                limit=limit - len(memories),
                min_strength=min_strength,
                exclude_ids={m.memory_id for m in memories}
            ))

        # Sort by relevance (strength)
        memories.sort(key=lambda m: m.strength(), reverse=True)

        # Mark as accessed
        for mem in memories[:limit]:
            mem.access()
            # Update in database if long-term
            if mem.memory_type != MemoryType.WORKING:
                self._persist_memory(mem)

        return memories[:limit]

    def _retrieve_from_db(
        self,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[Set[str]] = None,
        limit: int = 10,
        min_strength: float = 0.1,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[Memory]:
        """Retrieve memories from database"""
        conn = sqlite3.connect(self.db_path)

        query = "SELECT * FROM memories WHERE 1=1"
        params = []

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        if exclude_ids:
            placeholders = ",".join("?" * len(exclude_ids))
            query += f" AND memory_id NOT IN ({placeholders})"
            params.extend(exclude_ids)

        query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit * 2)  # Fetch extra for filtering

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        memories = []
        for row in rows:
            memory = self._row_to_memory(row)

            # Filter by tags if specified
            if tags and not memory.tags.intersection(tags):
                continue

            # Filter by strength
            if memory.strength() >= min_strength:
                memories.append(memory)

            if len(memories) >= limit:
                break

        return memories

    def associate(self, memory_id1: str, memory_id2: str):
        """Create association between two memories"""
        # Get memories
        mem1 = self.working_memories.get(memory_id1)
        mem2 = self.working_memories.get(memory_id2)

        if mem1 and memory_id2 not in mem1.associations:
            mem1.associations.append(memory_id2)
            if mem1.memory_type != MemoryType.WORKING:
                self._persist_memory(mem1)

        if mem2 and memory_id1 not in mem2.associations:
            mem2.associations.append(memory_id1)
            if mem2.memory_type != MemoryType.WORKING:
                self._persist_memory(mem2)

    def get_associated(self, memory_id: str, max_depth: int = 2) -> List[Memory]:
        """Get memories associated with given memory (up to max_depth links)"""
        visited = set()
        to_visit = [(memory_id, 0)]
        associated = []

        while to_visit:
            current_id, depth = to_visit.pop(0)

            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Get memory
            memory = self.working_memories.get(current_id)
            if not memory:
                # Try to load from database
                memories = self._retrieve_from_db(limit=1, exclude_ids=visited)
                memory = memories[0] if memories else None

            if memory and depth > 0:  # Don't include the starting memory
                associated.append(memory)

            # Add associations to visit
            if memory:
                for assoc_id in memory.associations:
                    if assoc_id not in visited:
                        to_visit.append((assoc_id, depth + 1))

        return associated

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        conn = sqlite3.connect(self.db_path)

        cursor = conn.execute("SELECT COUNT(*), memory_type FROM memories GROUP BY memory_type")
        type_counts = {row[1]: row[0] for row in cursor.fetchall()}

        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        conn.close()

        return {
            "working_memory_count": len(self.working_memories),
            "working_memory_capacity": self.working_memory_capacity,
            "total_memories": total,
            "by_type": type_counts,
            "unique_tags": len(self.memory_by_tag),
        }
