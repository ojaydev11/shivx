"""
ShivX Multi-Agent Orchestration Framework
==========================================

Core autonomous operation capabilities enabling ShivX to function as an AGI OS.

Components:
- Intent Router: Classifies user requests and routes to appropriate agents
- Task Graph Executor: Manages DAG-based task execution with parallelism
- Handoff Manager: Handles state transfer between agents
- Resource Governor: Enforces resource quotas and limits

Created: 2025-10-28
Priority: P1 - HIGH
"""

from core.orchestration.intent_router import IntentRouter, IntentCategory, IntentResult
from core.orchestration.task_graph import (
    TaskGraph,
    TaskNode,
    TaskType,
    TaskStatus,
    ExecutionResult,
)
from core.orchestration.handoff import HandoffManager, HandoffContext, HandoffResult
from core.orchestration.resource_governor import ResourceGovernor, ResourceQuota, ResourceUsage

__all__ = [
    "IntentRouter",
    "IntentCategory",
    "IntentResult",
    "TaskGraph",
    "TaskNode",
    "TaskType",
    "TaskStatus",
    "ExecutionResult",
    "HandoffManager",
    "HandoffContext",
    "HandoffResult",
    "ResourceGovernor",
    "ResourceQuota",
    "ResourceUsage",
]
