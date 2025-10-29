"""
AGI Approach Modules
Different research directions for AGI
"""
from .base import BaseAGIApproach
from .world_model import WorldModelLearner
from .meta_learner import MetaLearner
from .causal_reasoner import CausalReasoner

__all__ = [
    "BaseAGIApproach",
    "WorldModelLearner",
    "MetaLearner",
    "CausalReasoner",
]
