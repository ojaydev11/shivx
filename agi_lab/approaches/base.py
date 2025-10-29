"""
Base AGI Approach
Abstract interface for all AGI research approaches
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time

from ..schemas import (
    AGIApproachType,
    ExperimentResult,
    NeuralPattern,
    AGIFitnessMetrics
)
from ..pattern_recorder import PatternRecorder


class BaseAGIApproach(ABC):
    """Base class for AGI approaches"""

    def __init__(
        self,
        approach_type: AGIApproachType,
        config: Optional[Dict[str, Any]] = None,
        pattern_recorder: Optional[PatternRecorder] = None
    ):
        self.approach_type = approach_type
        self.config = config or {}
        self.recorder = pattern_recorder or PatternRecorder()
        self.patterns: List[NeuralPattern] = []

    @abstractmethod
    def train(self, tasks: List[Dict[str, Any]]) -> None:
        """Train on a set of tasks"""
        pass

    @abstractmethod
    def evaluate(self, test_tasks: List[Dict[str, Any]]) -> AGIFitnessMetrics:
        """Evaluate AGI-ness on test tasks"""
        pass

    @abstractmethod
    def transfer(self, new_domain: str, tasks: List[Dict[str, Any]]) -> float:
        """Test transfer learning to new domain"""
        pass

    def record_pattern(
        self,
        pattern_type: str,
        context: str,
        data: Dict[str, Any],
        success_score: float = 0.0,
        generalization_score: float = 0.0,
        novelty_score: float = 0.0
    ) -> NeuralPattern:
        """Record a neural pattern during execution"""
        pattern = NeuralPattern(
            approach_type=self.approach_type,
            pattern_type=pattern_type,
            context=context,
            data=data,
            success_score=success_score,
            generalization_score=generalization_score,
            novelty_score=novelty_score,
        )
        self.patterns.append(pattern)
        self.recorder.record_pattern(pattern)
        return pattern

    def run_experiment(
        self,
        train_tasks: List[Dict[str, Any]],
        test_tasks: List[Dict[str, Any]],
        transfer_tasks: Optional[List[Dict[str, Any]]] = None
    ) -> ExperimentResult:
        """Run full experiment: train, evaluate, measure transfer"""
        start_time = time.time()

        # Train
        self.train(train_tasks)

        # Evaluate
        fitness = self.evaluate(test_tasks)

        # Transfer learning
        transfer_score = 0.0
        if transfer_tasks:
            transfer_score = self.transfer("new_domain", transfer_tasks)

        training_time = time.time() - start_time

        # Create result
        result = ExperimentResult(
            approach_type=self.approach_type,
            config=self.config,
            patterns=self.patterns,
            task_success_rate=fitness.general_reasoning,
            generalization_score=fitness.transfer_learning,
            transfer_learning_score=transfer_score,
            reasoning_depth=len(self.patterns),
            novelty_score=sum(p.novelty_score for p in self.patterns) / max(len(self.patterns), 1),
            efficiency=1.0 / (training_time + 1.0),
            training_time_sec=training_time,
            memory_usage_mb=self._estimate_memory(),
        )

        return result

    def _estimate_memory(self) -> float:
        """Estimate memory usage in MB"""
        # Simple heuristic based on patterns
        return len(self.patterns) * 0.01  # ~10KB per pattern

    def get_state(self) -> Dict[str, Any]:
        """Get current state for cross-pollination"""
        return {
            "approach_type": self.approach_type.value,
            "config": self.config,
            "num_patterns": len(self.patterns),
            "best_patterns": [p.dict() for p in sorted(
                self.patterns,
                key=lambda x: x.success_score,
                reverse=True
            )[:10]]
        }
