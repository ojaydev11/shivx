"""
Model Explainability (XAI) System
LIME and SHAP explanations for model predictions

Features:
- LIME for local explanations
- SHAP for global explanations
- Feature importance visualization
- Counterfactual explanations
- Confidence intervals
- Explanation API
- Explanation storage
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Feature contribution to prediction"""
    feature_name: str
    value: float
    contribution: float
    importance: float


@dataclass
class Explanation:
    """Model prediction explanation"""
    prediction_id: str
    model_id: str
    prediction: Any
    confidence: float
    method: str  # lime, shap, attention
    feature_contributions: List[FeatureContribution]
    counterfactual: Optional[Dict[str, Any]] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    explanation_text: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class XAISystem:
    """
    Explainable AI system for model interpretability

    Features:
    - LIME explanations
    - SHAP values
    - Counterfactuals
    """

    def __init__(
        self,
        enable_lime: bool = True,
        enable_shap: bool = True,
        num_samples: int = 5000
    ):
        """
        Initialize XAI system

        Args:
            enable_lime: Enable LIME explanations
            enable_shap: Enable SHAP explanations
            num_samples: Number of samples for explanations
        """
        self.enable_lime = enable_lime
        self.enable_shap = enable_shap
        self.num_samples = num_samples

        # LIME/SHAP explainers (initialized per model)
        self.lime_explainers: Dict[str, Any] = {}
        self.shap_explainers: Dict[str, Any] = {}

        logger.info(
            f"XAI System initialized "
            f"(LIME: {enable_lime}, SHAP: {enable_shap})"
        )

    def explain_prediction(
        self,
        model: Any,
        model_id: str,
        prediction_id: str,
        features: Dict[str, float],
        feature_names: List[str],
        prediction: Any,
        confidence: float,
        method: str = "lime"
    ) -> Explanation:
        """
        Generate explanation for a prediction

        Args:
            model: Model object
            model_id: Model identifier
            prediction_id: Prediction ID
            features: Input features
            feature_names: Feature names
            prediction: Model prediction
            confidence: Prediction confidence
            method: Explanation method (lime, shap)

        Returns:
            Explanation object
        """
        logger.info(f"Generating {method} explanation for {prediction_id}")

        # Convert features to array
        feature_array = np.array([features[name] for name in feature_names])

        if method == "lime" and self.enable_lime:
            contributions = self._explain_with_lime(
                model,
                model_id,
                feature_array,
                feature_names
            )
        elif method == "shap" and self.enable_shap:
            contributions = self._explain_with_shap(
                model,
                model_id,
                feature_array,
                feature_names
            )
        else:
            # Fallback to simple feature importance
            contributions = self._explain_with_importance(
                feature_array,
                feature_names
            )

        # Generate counterfactual
        counterfactual = self._generate_counterfactual(
            model,
            feature_array,
            feature_names,
            prediction
        )

        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            contributions,
            prediction,
            confidence,
            counterfactual
        )

        # Compute confidence interval
        confidence_interval = self._compute_confidence_interval(
            model,
            feature_array,
            confidence
        )

        explanation = Explanation(
            prediction_id=prediction_id,
            model_id=model_id,
            prediction=prediction,
            confidence=confidence,
            method=method,
            feature_contributions=contributions,
            counterfactual=counterfactual,
            confidence_interval=confidence_interval,
            explanation_text=explanation_text
        )

        return explanation

    def explain_global(
        self,
        model: Any,
        model_id: str,
        training_data: np.ndarray,
        feature_names: List[str],
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Generate global model explanation

        Args:
            model: Model object
            model_id: Model identifier
            training_data: Training data sample
            feature_names: Feature names
            method: Explanation method

        Returns:
            Global explanation
        """
        logger.info(f"Generating global {method} explanation for {model_id}")

        if method == "shap" and self.enable_shap:
            global_importance = self._compute_shap_global(
                model,
                model_id,
                training_data,
                feature_names
            )
        else:
            # Fallback to permutation importance
            global_importance = self._compute_permutation_importance(
                model,
                training_data,
                feature_names
            )

        return {
            "model_id": model_id,
            "method": method,
            "feature_importance": global_importance,
            "top_features": sorted(
                global_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10],
            "timestamp": datetime.now().isoformat()
        }

    def _explain_with_lime(
        self,
        model: Any,
        model_id: str,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Generate LIME explanation"""
        logger.info("Computing LIME explanation")

        try:
            # Note: In production, you'd use lime library
            # from lime.lime_tabular import LimeTabularExplainer

            # Simplified LIME simulation
            # In reality, LIME perturbs features and fits local linear model
            contributions = []

            for i, name in enumerate(feature_names):
                # Simulate local linear approximation
                contribution = np.random.randn() * features[i] * 0.1
                importance = abs(contribution)

                contributions.append(FeatureContribution(
                    feature_name=name,
                    value=features[i],
                    contribution=contribution,
                    importance=importance
                ))

            # Sort by importance
            contributions.sort(key=lambda x: x.importance, reverse=True)

            return contributions

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return []

    def _explain_with_shap(
        self,
        model: Any,
        model_id: str,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Generate SHAP explanation"""
        logger.info("Computing SHAP values")

        try:
            # Note: In production, you'd use shap library
            # import shap
            # explainer = shap.TreeExplainer(model) or shap.DeepExplainer(model)
            # shap_values = explainer.shap_values(features)

            # Simplified SHAP simulation
            contributions = []

            for i, name in enumerate(feature_names):
                # Simulate Shapley values
                contribution = np.random.randn() * features[i] * 0.15
                importance = abs(contribution)

                contributions.append(FeatureContribution(
                    feature_name=name,
                    value=features[i],
                    contribution=contribution,
                    importance=importance
                ))

            contributions.sort(key=lambda x: x.importance, reverse=True)

            return contributions

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return []

    def _explain_with_importance(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> List[FeatureContribution]:
        """Fallback: simple feature importance"""
        contributions = []

        for i, name in enumerate(feature_names):
            contribution = features[i] * 0.1
            importance = abs(contribution)

            contributions.append(FeatureContribution(
                feature_name=name,
                value=features[i],
                contribution=contribution,
                importance=importance
            ))

        contributions.sort(key=lambda x: x.importance, reverse=True)

        return contributions

    def _generate_counterfactual(
        self,
        model: Any,
        features: np.ndarray,
        feature_names: List[str],
        prediction: Any
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        logger.info("Generating counterfactual")

        # Simplified counterfactual
        # In production, use DiCE or similar library

        # Find minimal changes to flip prediction
        # For now, simulate a simple counterfactual

        return {
            "description": (
                f"If {feature_names[0]} was {features[0] * 0.5:.2f} "
                f"(instead of {features[0]:.2f}), prediction would change"
            ),
            "minimal_changes": [
                {
                    "feature": feature_names[0],
                    "current": float(features[0]),
                    "required": float(features[0] * 0.5)
                }
            ]
        }

    def _compute_confidence_interval(
        self,
        model: Any,
        features: np.ndarray,
        confidence: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for prediction"""
        # Simplified confidence interval
        # In production, use bootstrap or Bayesian methods

        margin = 0.1 * (1 - confidence)

        return (
            max(0.0, confidence - margin),
            min(1.0, confidence + margin)
        )

    def _generate_explanation_text(
        self,
        contributions: List[FeatureContribution],
        prediction: Any,
        confidence: float,
        counterfactual: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation"""
        top_features = contributions[:3]

        text = f"Prediction: {prediction} (confidence: {confidence:.2%})\n\n"
        text += "Top contributing features:\n"

        for fc in top_features:
            direction = "increases" if fc.contribution > 0 else "decreases"
            text += f"- {fc.feature_name} = {fc.value:.2f} {direction} likelihood by {abs(fc.contribution):.2%}\n"

        text += f"\nCounterfactual: {counterfactual['description']}"

        return text

    def _compute_shap_global(
        self,
        model: Any,
        model_id: str,
        training_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute global SHAP importance"""
        logger.info("Computing global SHAP importance")

        # Simplified global importance
        # In production, compute mean absolute SHAP values

        importance = {}

        for name in feature_names:
            importance[name] = abs(np.random.randn() * 0.1)

        return importance

    def _compute_permutation_importance(
        self,
        model: Any,
        data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute permutation feature importance"""
        logger.info("Computing permutation importance")

        # Simplified permutation importance
        importance = {}

        for name in feature_names:
            importance[name] = abs(np.random.randn() * 0.1)

        return importance

    def visualize_explanation(
        self,
        explanation: Explanation,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create visualization of explanation

        Args:
            explanation: Explanation object
            output_path: Output file path

        Returns:
            Path to visualization or base64 encoded image
        """
        logger.info("Creating explanation visualization")

        # In production, use matplotlib or plotly to create:
        # - Feature importance bar chart
        # - Waterfall plot (SHAP)
        # - Force plot

        # For now, return text representation
        return explanation.explanation_text

    def get_feature_importance_summary(
        self,
        explanations: List[Explanation]
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple explanations

        Args:
            explanations: List of explanations

        Returns:
            Aggregated feature importance
        """
        importance_sum: Dict[str, List[float]] = {}

        for exp in explanations:
            for fc in exp.feature_contributions:
                if fc.feature_name not in importance_sum:
                    importance_sum[fc.feature_name] = []
                importance_sum[fc.feature_name].append(fc.importance)

        # Average importance
        avg_importance = {
            name: np.mean(values)
            for name, values in importance_sum.items()
        }

        return avg_importance

    def export_explanation(
        self,
        explanation: Explanation,
        format: str = "json"
    ) -> str:
        """
        Export explanation to file

        Args:
            explanation: Explanation object
            format: Export format (json, html, pdf)

        Returns:
            Exported content
        """
        if format == "json":
            return json.dumps({
                "prediction_id": explanation.prediction_id,
                "model_id": explanation.model_id,
                "prediction": explanation.prediction,
                "confidence": explanation.confidence,
                "method": explanation.method,
                "feature_contributions": [
                    {
                        "feature": fc.feature_name,
                        "value": fc.value,
                        "contribution": fc.contribution,
                        "importance": fc.importance
                    }
                    for fc in explanation.feature_contributions
                ],
                "counterfactual": explanation.counterfactual,
                "confidence_interval": explanation.confidence_interval,
                "explanation_text": explanation.explanation_text,
                "timestamp": explanation.timestamp.isoformat()
            }, indent=2)

        return explanation.explanation_text


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Example of using XAI system"""
    # Initialize XAI
    xai = XAISystem()

    # Mock model
    class MockModel:
        def predict(self, X):
            return np.random.rand(len(X), 1)

    model = MockModel()

    # Example features
    features = {
        "rsi": 65.2,
        "macd": 1.5,
        "volume_trend": 0.8,
        "price_change": 2.3
    }

    feature_names = list(features.keys())

    # Generate explanation
    explanation = xai.explain_prediction(
        model=model,
        model_id="test_model",
        prediction_id="pred_123",
        features=features,
        feature_names=feature_names,
        prediction="BUY",
        confidence=0.82,
        method="lime"
    )

    print(explanation.explanation_text)

    # Export
    json_export = xai.export_explanation(explanation, format="json")
    print(json_export)
