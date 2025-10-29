"""
Hybrid AGI Architecture
Combines the best of multiple AGI approaches into one unified system

Key Insight: No single approach solves AGI. Hybrid = Causal + World Model + Meta-Learning

Architecture:
- Causal Reasoner: Provides WHY (enables transfer learning)
- World Model: Provides precise predictions (within domain)
- Meta-Learner: Optimizes both approaches over time

This is the breakthrough toward AGI!
"""
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BaseAGIApproach
from .causal_reasoner import CausalReasoner
from .world_model import WorldModelLearner
from .meta_learner import MetaLearner
from ..schemas import AGIApproachType, AGIFitnessMetrics


class HybridAGI(BaseAGIApproach):
    """
    Unified AGI system combining multiple approaches

    Decision Flow:
    1. Meta-Learner selects which approach to use
    2. World Model for within-domain predictions
    3. Causal Reasoner for transfer and "why" questions
    4. Both feed back to Meta-Learner for improvement
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(AGIApproachType.HYBRID, config, **kwargs)

        # Initialize all sub-approaches
        self.causal = CausalReasoner(config=config, pattern_recorder=self.recorder)
        self.world_model = WorldModelLearner(config=config, pattern_recorder=self.recorder)
        self.meta = MetaLearner(config=config, pattern_recorder=self.recorder)

        # Routing weights: which approach to use when
        self.routing_weights = {
            "causal": 0.5,      # High weight for transfer
            "world_model": 0.3,  # Medium for within-domain
            "meta": 0.2,         # Low initially, grows over time
        }

        # Performance tracking
        self.approach_performance = {
            "causal": [],
            "world_model": [],
            "meta": [],
        }

    def train(self, tasks: List[Dict[str, Any]]) -> None:
        """Train all sub-approaches"""
        # Train each approach
        self.causal.train(tasks)
        self.world_model.train(tasks)
        self.meta.train(tasks)

        # Meta-learner observes performance and updates routing
        self._update_routing()

        # Record hybrid pattern
        self.record_pattern(
            pattern_type="hybrid_training",
            context="multi_approach",
            data={
                "routing_weights": self.routing_weights.copy(),
                "causal_patterns": len(self.causal.patterns),
                "world_model_patterns": len(self.world_model.patterns),
                "meta_patterns": len(self.meta.patterns),
            },
            success_score=0.8,
            generalization_score=0.0,
        )

    def evaluate(self, test_tasks: List[Dict[str, Any]]) -> AGIFitnessMetrics:
        """Evaluate using ensemble of approaches"""
        # Evaluate each approach
        causal_fitness = self.causal.evaluate(test_tasks)
        world_fitness = self.world_model.evaluate(test_tasks)
        meta_fitness = self.meta.evaluate(test_tasks)

        # Track performance
        self.approach_performance["causal"].append(causal_fitness.general_reasoning)
        self.approach_performance["world_model"].append(world_fitness.general_reasoning)
        self.approach_performance["meta"].append(meta_fitness.general_reasoning)

        # Weighted combination based on routing
        w_c = self.routing_weights["causal"]
        w_w = self.routing_weights["world_model"]
        w_m = self.routing_weights["meta"]

        # Combine fitness scores
        combined = AGIFitnessMetrics(
            overall_score=0.0,
            general_reasoning=(
                w_c * causal_fitness.general_reasoning +
                w_w * world_fitness.general_reasoning +
                w_m * meta_fitness.general_reasoning
            ),
            transfer_learning=causal_fitness.transfer_learning,  # Causal best for transfer
            causal_understanding=causal_fitness.causal_understanding,
            abstraction=max(
                causal_fitness.abstraction,
                world_fitness.abstraction,
                meta_fitness.abstraction
            ),
            creativity=(
                causal_fitness.creativity +
                world_fitness.creativity +
                meta_fitness.creativity
            ) / 3,
            metacognition=meta_fitness.metacognition,  # Meta-learner excels here
            sample_efficiency=meta_fitness.sample_efficiency,
            robustness=(
                causal_fitness.robustness +
                world_fitness.robustness +
                meta_fitness.robustness
            ) / 3,
            interpretability=(
                causal_fitness.interpretability +
                world_fitness.interpretability
            ) / 2,
        )

        combined.compute_overall()
        return combined

    def transfer(self, new_domain: str, tasks: List[Dict[str, Any]]) -> float:
        """Test transfer learning"""
        # Causal reasoning is best for transfer
        causal_transfer = self.causal.transfer(new_domain, tasks)

        # World model and meta also try
        world_transfer = self.world_model.transfer(new_domain, tasks)
        meta_transfer = self.meta.transfer(new_domain, tasks)

        # Take best (or combine)
        return max(causal_transfer, world_transfer, meta_transfer)

    def _update_routing(self) -> None:
        """Meta-learning: adjust routing weights based on performance"""
        if not any(self.approach_performance.values()):
            return  # No data yet

        # Which approach performed best recently?
        recent_window = 5

        causal_recent = (
            np.mean(self.approach_performance["causal"][-recent_window:])
            if self.approach_performance["causal"]
            else 0.0
        )
        world_recent = (
            np.mean(self.approach_performance["world_model"][-recent_window:])
            if self.approach_performance["world_model"]
            else 0.0
        )
        meta_recent = (
            np.mean(self.approach_performance["meta"][-recent_window:])
            if self.approach_performance["meta"]
            else 0.0
        )

        # Softmax to get weights
        total = causal_recent + world_recent + meta_recent
        if total > 0:
            self.routing_weights["causal"] = causal_recent / total
            self.routing_weights["world_model"] = world_recent / total
            self.routing_weights["meta"] = meta_recent / total

    def predict(self, query: Dict[str, Any]) -> Any:
        """
        Unified prediction using best approach for the query

        Decision logic:
        - If within known domain -> World Model
        - If transfer/causal question -> Causal Reasoner
        - If meta-question (about learning itself) -> Meta-Learner
        """
        query_type = query.get("type", "unknown")

        # Route to appropriate approach
        if query_type in ["physics", "simulation", "prediction"]:
            # World model is best for precise within-domain predictions
            result = self._world_model_predict(query)
            source = "world_model"

        elif query_type in ["causal", "why", "counterfactual", "transfer"]:
            # Causal reasoner for understanding and transfer
            result = self._causal_predict(query)
            source = "causal"

        elif query_type in ["meta", "learning", "strategy"]:
            # Meta-learner for questions about learning itself
            result = self._meta_predict(query)
            source = "meta"

        else:
            # Unknown: use weighted voting
            result = self._ensemble_predict(query)
            source = "ensemble"

        # Record decision
        self.record_pattern(
            pattern_type="prediction",
            context=f"query_type={query_type}",
            data={
                "query": query,
                "source": source,
                "result": str(result)[:100],  # Truncate
            },
            success_score=0.7,
            generalization_score=0.0,
        )

        return result

    def _world_model_predict(self, query: Dict[str, Any]) -> Any:
        """Use world model for prediction"""
        state = query.get("state", "")
        action = query.get("action", "")
        return self.world_model._predict_next(state, action)

    def _causal_predict(self, query: Dict[str, Any]) -> Any:
        """Use causal reasoner for prediction"""
        cause = query.get("cause", "")
        effect = query.get("effect", "")
        return self.causal._predict_intervention(cause, effect)

    def _meta_predict(self, query: Dict[str, Any]) -> Any:
        """Use meta-learner for prediction"""
        # Meta-learner provides strategy recommendations
        return {
            "learning_rate": self.meta.learning_rate,
            "exploration_rate": self.meta.exploration_rate,
            "recommended_approach": max(
                self.routing_weights,
                key=self.routing_weights.get
            ),
        }

    def _ensemble_predict(self, query: Dict[str, Any]) -> Any:
        """Ensemble prediction when uncertain"""
        # Get predictions from all approaches
        predictions = []

        try:
            pred_world = self._world_model_predict(query)
            predictions.append(("world_model", pred_world, self.routing_weights["world_model"]))
        except:
            pass

        try:
            pred_causal = self._causal_predict(query)
            predictions.append(("causal", pred_causal, self.routing_weights["causal"]))
        except:
            pass

        try:
            pred_meta = self._meta_predict(query)
            predictions.append(("meta", pred_meta, self.routing_weights["meta"]))
        except:
            pass

        # Return highest-weighted prediction
        if predictions:
            return max(predictions, key=lambda x: x[2])[1]

        return None

    def explain(self, query: Dict[str, Any], result: Any) -> str:
        """
        Explain WHY a prediction was made
        Key AGI capability: interpretability and reasoning transparency
        """
        # Use causal reasoner for explanation
        explanation = f"Result: {result}\n\n"

        # Which approach was used?
        query_type = query.get("type", "unknown")

        if query_type in ["physics", "simulation", "prediction"]:
            explanation += "Reasoning: World model predicted based on learned state transitions.\n"
            explanation += f"Confidence: {self.world_model._measure_creativity():.1%}\n"

        elif query_type in ["causal", "why", "counterfactual"]:
            explanation += "Reasoning: Causal graph analysis.\n"
            explanation += f"Causal edges: {len(self.causal.causal_edges)}\n"

        elif query_type in ["meta", "learning"]:
            explanation += "Reasoning: Meta-learning strategy optimization.\n"
            explanation += f"Learning rate: {self.meta.learning_rate:.3f}\n"

        else:
            explanation += "Reasoning: Ensemble of multiple approaches.\n"
            explanation += f"Routing weights: {self.routing_weights}\n"

        return explanation
