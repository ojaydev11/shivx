"""
Social Intelligence & Theory of Mind - Pillar 9

This module provides comprehensive social intelligence capabilities for AGI,
enabling it to understand, predict, and interact with other agents.

Key components:
- Theory of Mind: Model other agents' beliefs, intentions, and mental states
- Social Reasoning: Understand social norms, predict behavior, recognize intent
- Collaboration: Cooperative planning, communication, conflict resolution

Integration points:
- Uses planning.GoalPlanner for cooperative goal decomposition
- Uses memory.MemorySystem for tracking social interactions
- Supports multi-agent coordination and teamwork
"""

from .theory_of_mind import (
    TheoryOfMind,
    AgentModel,
    MentalState,
    Belief,
    BeliefType,
    Intention,
    IntentionType,
)

from .social_reasoner import (
    SocialReasoner,
    SocialNorm,
    NormType,
    SocialContext,
    SocialRole,
    Intent,
    Behavior,
    BehaviorType,
)

from .collaboration_engine import (
    CollaborationEngine,
    CollaborativeTask,
    TaskStatus,
    CommunicationStrategy,
    CommunicationType,
    ConflictResolution,
    ConflictType,
)

__all__ = [
    # Theory of Mind
    "TheoryOfMind",
    "AgentModel",
    "MentalState",
    "Belief",
    "BeliefType",
    "Intention",
    "IntentionType",

    # Social Reasoning
    "SocialReasoner",
    "SocialNorm",
    "NormType",
    "SocialContext",
    "SocialRole",
    "Intent",
    "Behavior",
    "BehaviorType",

    # Collaboration
    "CollaborationEngine",
    "CollaborativeTask",
    "TaskStatus",
    "CommunicationStrategy",
    "CommunicationType",
    "ConflictResolution",
    "ConflictType",
]
