"""
Pillar 5: Planning & Goal-Directed Behavior

Enables AGI to set goals, decompose them, create plans, and pursue objectives autonomously.
"""

from .goal_planner import GoalPlanner, Goal, Plan, PlanStep

__all__ = [
    "GoalPlanner",
    "Goal",
    "Plan",
    "PlanStep",
]
