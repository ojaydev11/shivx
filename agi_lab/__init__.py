"""
AGI Lab - Parallel AGI Exploration Framework
Brain-inspired parallel exploration of AGI approaches
"""
from .schemas import (
    AGIApproachType,
    NeuralPattern,
    ExperimentResult,
    AGIFitnessMetrics,
    ExplorationSession,
)
from .pattern_recorder import PatternRecorder
from .parallel_explorer import ParallelExplorer
from .task_generator import TaskGenerator
from .approaches import (
    BaseAGIApproach,
    WorldModelLearner,
    MetaLearner,
    CausalReasoner,
)

__all__ = [
    "AGIApproachType",
    "NeuralPattern",
    "ExperimentResult",
    "AGIFitnessMetrics",
    "ExplorationSession",
    "PatternRecorder",
    "ParallelExplorer",
    "TaskGenerator",
    "BaseAGIApproach",
    "WorldModelLearner",
    "MetaLearner",
    "CausalReasoner",
]
