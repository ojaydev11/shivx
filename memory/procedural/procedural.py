"""
Procedural memory for storing skills, procedures, and how-to knowledge.

Like human procedural memory, this stores:
- Step-by-step procedures
- Skills and techniques
- Code snippets and recipes
- Prerequisites and dependencies
"""

from datetime import datetime
from typing import List, Optional

from loguru import logger

from memory.encoders.text_encoder import TextEncoder
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import MemoryNode, NodeType, RelationType, Skill


class ProceduralMemory:
    """
    Procedural memory system for AGI.

    Stores skills and procedures with execution tracking.
    """

    def __init__(
        self,
        graph_store: MemoryGraphStore,
        text_encoder: TextEncoder,
    ):
        """
        Initialize procedural memory.

        Args:
            graph_store: Memory graph database
            text_encoder: Text embedding encoder
        """
        self.graph_store = graph_store
        self.text_encoder = text_encoder
        logger.info("Procedural memory initialized")

    def store_skill(
        self,
        skill: Skill,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a procedural skill.

        Args:
            skill: Skill to store
            importance: Importance score [0, 1]
            tags: Optional tags

        Returns:
            Node ID
        """
        # Format skill as text
        skill_text = f"{skill.name}: {skill.description}\n"
        skill_text += "Steps:\n" + "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(skill.steps)
        )

        # Generate embedding
        embedding = self.text_encoder.encode(skill_text)

        # Create memory node
        node = MemoryNode(
            node_type=NodeType.SKILL,
            content=skill_text,
            embedding=embedding,
            importance=importance,
            metadata={
                "name": skill.name,
                "description": skill.description,
                "steps": skill.steps,
                "prerequisites": skill.prerequisites,
                "success_rate": skill.success_rate,
                "execution_count": skill.execution_count,
            },
            tags=tags or [],
            source="procedural",
        )

        node_id = self.graph_store.add_node(node)

        # Link prerequisites
        for prereq in skill.prerequisites:
            prereq_id = self._find_skill_by_name(prereq)
            if prereq_id:
                self.graph_store.link(
                    node_id, prereq_id, RelationType.DEPENDS_ON, weight=1.0
                )

        logger.debug(f"Stored skill: {skill.name}")
        return node_id

    def _find_skill_by_name(self, name: str) -> Optional[str]:
        """Find skill node by name."""
        results = self.graph_store.search_text(name, limit=5)
        for node in results:
            if node.node_type == NodeType.SKILL and node.metadata.get("name") == name:
                return node.id
        return None

    def recall_skill(self, skill_name: str) -> Optional[MemoryNode]:
        """
        Recall a skill by name.

        Args:
            skill_name: Name of skill

        Returns:
            Skill node or None
        """
        skill_id = self._find_skill_by_name(skill_name)
        if skill_id:
            return self.graph_store.get_node(skill_id)
        return None

    def record_execution(
        self, skill_id: str, success: bool
    ) -> None:
        """
        Record skill execution outcome.

        Args:
            skill_id: Skill node ID
            success: Whether execution succeeded
        """
        node = self.graph_store.get_node(skill_id)
        if not node:
            return

        # Update execution count
        exec_count = node.metadata.get("execution_count", 0) + 1
        node.metadata["execution_count"] = exec_count

        # Update success rate (exponential moving average)
        old_rate = node.metadata.get("success_rate", 0.5)
        new_rate = 0.9 * old_rate + 0.1 * (1.0 if success else 0.0)
        node.metadata["success_rate"] = new_rate

        # Update importance based on usage
        node.boost_importance(1.05)

        # Save updates
        self.graph_store.add_node(node)

        logger.debug(
            f"Recorded execution for {skill_id}: "
            f"success={success}, rate={new_rate:.2f}"
        )

    def get_prerequisites(self, skill_id: str) -> List[MemoryNode]:
        """
        Get prerequisite skills.

        Args:
            skill_id: Skill node ID

        Returns:
            List of prerequisite skills
        """
        neighbors = self.graph_store.get_neighbors(
            skill_id, relation_type=RelationType.DEPENDS_ON
        )
        return [node for node, rel, weight in neighbors]

    def store_code(
        self,
        code: str,
        description: str,
        language: str = "python",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Store a code snippet.

        Args:
            code: Code snippet
            description: Description of what the code does
            language: Programming language
            tags: Optional tags

        Returns:
            Node ID
        """
        content = f"{description}\n\n```{language}\n{code}\n```"

        # Generate embedding from description
        embedding = self.text_encoder.encode(description)

        node = MemoryNode(
            node_type=NodeType.CODE,
            content=content,
            embedding=embedding,
            importance=0.6,
            metadata={
                "language": language,
                "description": description,
                "code": code,
            },
            tags=tags or [language],
            source="procedural",
        )

        node_id = self.graph_store.add_node(node)
        logger.debug(f"Stored code snippet: {description[:50]}...")
        return node_id

    def recall_code(
        self, query: str, language: Optional[str] = None, limit: int = 5
    ) -> List[MemoryNode]:
        """
        Recall code snippets matching query.

        Args:
            query: Search query
            language: Optional language filter
            limit: Maximum results

        Returns:
            List of code nodes
        """
        # Search by text
        results = self.graph_store.search_text(query, limit=limit * 2)

        # Filter by type and language
        filtered = []
        for node in results:
            if node.node_type != NodeType.CODE:
                continue
            if language and node.metadata.get("language") != language:
                continue
            filtered.append(node)
            if len(filtered) >= limit:
                break

        return filtered
