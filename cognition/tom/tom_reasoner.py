"""
Theory-of-Mind reasoner for modeling beliefs and intents of agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class BeliefState:
    """Belief state of an agent."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.beliefs: Dict[str, Any] = {}
        self.goals: List[str] = []
        self.knowledge: Dict[str, bool] = {}
        self.updated_at = datetime.utcnow()

    def update_belief(self, key: str, value: Any) -> None:
        """Update a belief."""
        self.beliefs[key] = value
        self.updated_at = datetime.utcnow()

    def knows(self, fact: str) -> bool:
        """Check if agent knows a fact."""
        return self.knowledge.get(fact, False)

    def learn(self, fact: str) -> None:
        """Agent learns a fact."""
        self.knowledge[fact] = True
        self.updated_at = datetime.utcnow()


class ToMReasoner:
    """
    Theory-of-Mind reasoner.

    Models beliefs, goals, and knowledge of multiple agents.
    """

    def __init__(self, max_agents: int = 10, belief_depth: int = 2):
        """
        Initialize ToM reasoner.

        Args:
            max_agents: Maximum number of agents to model
            belief_depth: Depth of belief nesting (e.g., "I think you think...")
        """
        self.max_agents = max_agents
        self.belief_depth = belief_depth
        self.agents: Dict[str, BeliefState] = {}

        logger.info(
            f"ToM reasoner initialized: max_agents={max_agents}, depth={belief_depth}"
        )

    def add_agent(self, agent_id: str) -> BeliefState:
        """Add agent to model."""
        if agent_id in self.agents:
            return self.agents[agent_id]

        if len(self.agents) >= self.max_agents:
            logger.warning(f"Max agents ({self.max_agents}) reached")
            return None

        belief_state = BeliefState(agent_id)
        self.agents[agent_id] = belief_state

        logger.debug(f"Added agent: {agent_id}")
        return belief_state

    def get_agent(self, agent_id: str) -> Optional[BeliefState]:
        """Get agent's belief state."""
        return self.agents.get(agent_id)

    def update_belief(self, agent_id: str, key: str, value: Any) -> None:
        """Update agent's belief."""
        agent = self.get_agent(agent_id)
        if not agent:
            agent = self.add_agent(agent_id)

        if agent:
            agent.update_belief(key, value)
            logger.debug(f"Updated belief for {agent_id}: {key}={value}")

    def agent_knows(self, agent_id: str, fact: str) -> bool:
        """Check if agent knows a fact."""
        agent = self.get_agent(agent_id)
        if not agent:
            return False
        return agent.knows(fact)

    def teach(self, agent_id: str, fact: str) -> None:
        """Teach agent a fact."""
        agent = self.get_agent(agent_id)
        if not agent:
            agent = self.add_agent(agent_id)

        if agent:
            agent.learn(fact)
            logger.debug(f"Taught {agent_id}: {fact}")

    def predict_action(
        self, agent_id: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Predict what action an agent might take.

        Args:
            agent_id: Agent to predict
            context: Context information

        Returns:
            Predicted action or None
        """
        agent = self.get_agent(agent_id)
        if not agent:
            return None

        # Simplified prediction based on goals
        if agent.goals:
            return f"work_towards_{agent.goals[0]}"

        return None

    def resolve_conflict(
        self, agent1_id: str, agent2_id: str, issue: str
    ) -> Dict[str, Any]:
        """
        Suggest resolution for belief conflict.

        Args:
            agent1_id: First agent
            agent2_id: Second agent
            issue: Conflict issue

        Returns:
            Resolution suggestion
        """
        agent1 = self.get_agent(agent1_id)
        agent2 = self.get_agent(agent2_id)

        if not agent1 or not agent2:
            return {"resolution": "No conflict data"}

        return {
            "issue": issue,
            "agent1_belief": agent1.beliefs.get(issue),
            "agent2_belief": agent2.beliefs.get(issue),
            "resolution": "Gather more evidence",
        }

    def get_common_knowledge(self, agent_ids: List[str]) -> List[str]:
        """Get facts known by all specified agents."""
        if not agent_ids:
            return []

        common = None
        for agent_id in agent_ids:
            agent = self.get_agent(agent_id)
            if not agent:
                continue

            agent_knowledge = set(agent.knowledge.keys())
            if common is None:
                common = agent_knowledge
            else:
                common = common.intersection(agent_knowledge)

        return list(common) if common else []

    def get_stats(self) -> Dict[str, Any]:
        """Get ToM statistics."""
        return {
            "total_agents": len(self.agents),
            "max_agents": self.max_agents,
            "belief_depth": self.belief_depth,
        }
