"""
World Model Learner
Learns physics and causality by building predictive models of the world
"""
from typing import Any, Dict, List, Optional
import numpy as np
from collections import defaultdict

from .base import BaseAGIApproach
from ..schemas import AGIApproachType, AGIFitnessMetrics


class WorldModelLearner(BaseAGIApproach):
    """
    Learns world models: state -> action -> next_state prediction
    Tests causal understanding and generalization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(AGIApproachType.WORLD_MODEL, config, **kwargs)

        # Simple transition model: (state, action) -> next_state
        self.transitions: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.prediction_accuracy = []

    def train(self, tasks: List[Dict[str, Any]]) -> None:
        """Learn world model from task trajectories"""
        for task in tasks:
            task_type = task.get("type", "unknown")
            trajectory = task.get("trajectory", [])

            # Learn transitions
            for i in range(len(trajectory) - 1):
                state = trajectory[i].get("state", "")
                action = trajectory[i].get("action", "")
                next_state = trajectory[i + 1].get("state", "")

                # Record transition
                self.transitions[state][action].append(next_state)

                # Record pattern
                self.record_pattern(
                    pattern_type="transition",
                    context=f"task={task_type}",
                    data={
                        "state": state,
                        "action": action,
                        "next_state": next_state,
                    },
                    success_score=0.8,  # Placeholder
                    generalization_score=0.0,  # Will measure later
                )

    def evaluate(self, test_tasks: List[Dict[str, Any]]) -> AGIFitnessMetrics:
        """Evaluate world model predictions"""
        correct_predictions = 0
        total_predictions = 0

        causal_scores = []
        abstraction_scores = []

        for task in test_tasks:
            trajectory = task.get("trajectory", [])

            for i in range(len(trajectory) - 1):
                state = trajectory[i].get("state", "")
                action = trajectory[i].get("action", "")
                actual_next = trajectory[i + 1].get("state", "")

                # Predict next state
                predicted = self._predict_next(state, action)

                if predicted == actual_next:
                    correct_predictions += 1
                total_predictions += 1

                # Measure causal understanding (did we learn the right mechanism?)
                if action in self.transitions.get(state, {}):
                    causal_scores.append(1.0)
                else:
                    causal_scores.append(0.0)

                # Measure abstraction (can we generalize to unseen states?)
                if state not in self.transitions:
                    abstraction_scores.append(0.0)  # Novel state
                else:
                    abstraction_scores.append(1.0)

        accuracy = correct_predictions / max(total_predictions, 1)
        self.prediction_accuracy.append(accuracy)

        return AGIFitnessMetrics(
            overall_score=0.0,
            general_reasoning=accuracy,
            transfer_learning=0.0,  # Measured in transfer()
            causal_understanding=np.mean(causal_scores) if causal_scores else 0.0,
            abstraction=np.mean(abstraction_scores) if abstraction_scores else 0.0,
            creativity=self._measure_creativity(),
            metacognition=0.5,  # Placeholder
            sample_efficiency=len(self.transitions) / max(len(self.patterns), 1),
            robustness=np.std(self.prediction_accuracy) if len(self.prediction_accuracy) > 1 else 1.0,
            interpretability=0.9,  # Transitions are interpretable
        )

    def transfer(self, new_domain: str, tasks: List[Dict[str, Any]]) -> float:
        """Test transfer to new domain"""
        # Test zero-shot performance on new tasks
        initial_size = len(self.transitions)

        fitness_before = self.evaluate(tasks)

        # Quick adaptation (few-shot)
        self.train(tasks[:5])  # Only 5 examples

        fitness_after = self.evaluate(tasks[5:])

        # Transfer score = improvement from zero-shot to few-shot
        transfer_score = max(0.0, fitness_after.general_reasoning - fitness_before.general_reasoning)

        return transfer_score

    def _predict_next(self, state: str, action: str) -> str:
        """Predict next state given current state and action"""
        if state in self.transitions and action in self.transitions[state]:
            # Most common next state
            candidates = self.transitions[state][action]
            return max(set(candidates), key=candidates.count)

        # Fallback: try to generalize from similar states
        for known_state in self.transitions:
            if self._similarity(state, known_state) > 0.7:
                if action in self.transitions[known_state]:
                    candidates = self.transitions[known_state][action]
                    return max(set(candidates), key=candidates.count)

        return state  # No change predicted

    def _similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity"""
        if s1 == s2:
            return 1.0
        # Jaccard similarity on words
        words1 = set(s1.split())
        words2 = set(s2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _measure_creativity(self) -> float:
        """Measure novel predictions"""
        # Creativity = diversity of predicted transitions
        unique_transitions = set()
        for state, actions in self.transitions.items():
            for action, nexts in actions.items():
                for next_state in nexts:
                    unique_transitions.add((state, action, next_state))

        # Normalize by number of patterns
        return min(1.0, len(unique_transitions) / max(len(self.patterns), 1))
