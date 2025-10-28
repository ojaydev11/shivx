"""
ShivX Multi-Agent Framework
============================

Specialized autonomous agents for different capabilities:
- Planner Agent: Breaks down goals into actionable tasks
- Researcher Agent: Gathers and analyzes information
- Coder Agent: Writes and modifies code
- Operator Agent: Executes system commands and operations
- Finance Agent: Handles trading and financial operations
- Safety Agent: Validates safety constraints

Features:
- Base agent abstraction with lifecycle management
- Agent communication protocol (message passing)
- Capability-based task routing
- Resource-aware execution
- Guardian defense integration
"""

from core.agents.base_agent import (
    BaseAgent,
    AgentStatus,
    AgentMessage,
    AgentCapability,
    TaskResult,
)
from core.agents.planner import PlannerAgent
from core.agents.researcher import ResearcherAgent
from core.agents.coder import CoderAgent
from core.agents.operator import OperatorAgent
from core.agents.finance import FinanceAgent
from core.agents.safety import SafetyAgent

__all__ = [
    "BaseAgent",
    "AgentStatus",
    "AgentMessage",
    "AgentCapability",
    "TaskResult",
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "OperatorAgent",
    "FinanceAgent",
    "SafetyAgent",
]
