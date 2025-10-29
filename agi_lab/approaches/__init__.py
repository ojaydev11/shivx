"""
AGI Approach Modules
Different research directions for AGI
"""
from .base import BaseAGIApproach
from .world_model import WorldModelLearner
from .meta_learner import MetaLearner
from .causal_reasoner import CausalReasoner
from .neurosymbolic import NeurosymbolicAI
from .active_inference import ActiveInferenceAgent
from .compositional import CompositionalReasoner
from .analogical import AnalogicalReasoner
from .hybrid import HybridAGI

__all__ = [
    "BaseAGIApproach",
    "WorldModelLearner",
    "MetaLearner",
    "CausalReasoner",
    "NeurosymbolicAI",
    "ActiveInferenceAgent",
    "CompositionalReasoner",
    "AnalogicalReasoner",
    "HybridAGI",
]
