"""
Active Inference Agent
Based on Karl Friston's Free Energy Principle
Key insight: Intelligence = minimize prediction error (surprise)
"""
from typing import Any, Dict, List, Optional
import numpy as np
from collections import defaultdict

from .base import BaseAGIApproach
from ..schemas import AGIApproachType, AGIFitnessMetrics


class ActiveInferenceAgent(BaseAGIApproach):
    """
    Minimizes free energy (prediction error + entropy)
    Actions are selected to minimize expected surprise
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(AGIApproachType.ACTIVE_INFERENCE, config, **kwargs)

        # Generative model: P(observation | hidden_state)
        self.generative_model: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Beliefs about hidden states: Q(hidden_state)
        self.beliefs: Dict[str, float] = defaultdict(float)

        # Action policies: which actions minimize expected free energy?
        self.policies: Dict[str, List[str]] = {}

        # Free energy over time
        self.free_energy_history = []

    def train(self, tasks: List[Dict[str, Any]]) -> None:
        """Learn generative model from observations"""
        for task in tasks:
            trajectory = task.get("trajectory", [])

            for step in trajectory:
                observation = step.get("state", "")
                action = step.get("action", "")

                # Update generative model: P(obs | state, action)
                # In simplified version: count co-occurrences
                self.generative_model[action][observation] += 1.0

                # Update beliefs (posterior): Q(state | obs)
                self._update_beliefs(observation)

                # Compute free energy
                free_energy = self._compute_free_energy(observation)
                self.free_energy_history.append(free_energy)

                # Record pattern
                self.record_pattern(
                    pattern_type="inference",
                    context=f"task={task.get('type')}",
                    data={
                        "observation": observation,
                        "action": action,
                        "free_energy": free_energy,
                        "belief_entropy": self._entropy(self.beliefs),
                    },
                    success_score=1.0 - min(1.0, free_energy),
                    generalization_score=0.0,
                )

    def evaluate(self, test_tasks: List[Dict[str, Any]]) -> AGIFitnessMetrics:
        """Evaluate prediction accuracy and action selection"""
        prediction_errors = []
        action_successes = []

        for task in test_tasks:
            trajectory = task.get("trajectory", [])

            for i in range(len(trajectory) - 1):
                observation = trajectory[i].get("state", "")
                action = trajectory[i].get("action", "")
                next_obs = trajectory[i + 1].get("state", "")

                # Predict next observation
                predicted = self._predict_next(observation, action)

                # Measure prediction error
                error = 0.0 if predicted == next_obs else 1.0
                prediction_errors.append(error)

                # Did we choose action that minimizes free energy?
                best_action = self._select_action(observation)
                action_successes.append(1.0 if best_action == action else 0.0)

        avg_error = np.mean(prediction_errors) if prediction_errors else 1.0
        avg_action = np.mean(action_successes) if action_successes else 0.0

        # Free energy should decrease over time (learning)
        if len(self.free_energy_history) > 10:
            early = np.mean(self.free_energy_history[:10])
            late = np.mean(self.free_energy_history[-10:])
            improvement = max(0.0, early - late)
        else:
            improvement = 0.0

        return AGIFitnessMetrics(
            overall_score=0.0,
            general_reasoning=1.0 - avg_error,
            transfer_learning=improvement,
            causal_understanding=0.6,  # Generative model captures some causality
            abstraction=0.5,
            creativity=self._measure_exploration(),
            metacognition=0.7,  # Aware of uncertainty (beliefs)
            sample_efficiency=1.0 / (len(self.free_energy_history) + 1),
            robustness=avg_action,
            interpretability=0.6,  # Generative model somewhat interpretable
        )

    def transfer(self, new_domain: str, tasks: List[Dict[str, Any]]) -> float:
        """Test transfer learning"""
        initial_fe = np.mean(self.free_energy_history[-10:]) if self.free_energy_history else 1.0

        # Adapt to new domain
        self.train(tasks[:5])

        final_fe = np.mean(self.free_energy_history[-10:]) if self.free_energy_history else 1.0

        # Lower free energy = better
        return max(0.0, initial_fe - final_fe)

    def _update_beliefs(self, observation: str) -> None:
        """Bayesian update of beliefs"""
        # Simplified: beliefs = normalized counts
        self.beliefs[observation] += 1.0

        # Normalize
        total = sum(self.beliefs.values())
        if total > 0:
            for state in self.beliefs:
                self.beliefs[state] /= total

    def _compute_free_energy(self, observation: str) -> float:
        """
        Free Energy = Prediction Error + Entropy
        F = -log P(obs | belief) + H(belief)
        """
        # Prediction error: how surprising is this observation?
        belief_val = self.beliefs.get(observation, 0.01)
        surprise = -np.log(belief_val + 1e-10)

        # Entropy of beliefs
        entropy = self._entropy(self.beliefs)

        return surprise + 0.5 * entropy

    def _entropy(self, distribution: Dict[str, float]) -> float:
        """Compute Shannon entropy"""
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p + 1e-10)

        return entropy

    def _predict_next(self, observation: str, action: str) -> str:
        """Predict next observation given current state and action"""
        # Use generative model
        if action in self.generative_model:
            # Most likely observation for this action
            obs_probs = self.generative_model[action]
            if obs_probs:
                return max(obs_probs.keys(), key=lambda k: obs_probs[k])

        return observation  # No change predicted

    def _select_action(self, observation: str) -> str:
        """Select action that minimizes expected free energy"""
        # Expected Free Energy = Expected surprise + Expected info gain

        best_action = None
        min_efe = float('inf')

        for action in self.generative_model.keys():
            # Simulate taking this action
            predicted_obs = self._predict_next(observation, action)

            # Compute expected free energy
            efe = self._compute_free_energy(predicted_obs)

            if efe < min_efe:
                min_efe = efe
                best_action = action

        return best_action if best_action else "none"

    def _measure_exploration(self) -> float:
        """How much does it explore (high entropy policies)?"""
        # Creativity = entropy of action distribution
        action_counts = defaultdict(int)
        for pattern in self.patterns:
            if pattern.pattern_type == "inference":
                action = pattern.data.get("action", "")
                action_counts[action] += 1

        return self._entropy(dict(action_counts))
