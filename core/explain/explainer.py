"""
Explainability Engine - Explain AI decisions transparently.

Trust requires understanding. This module makes AI reasoning transparent:
- Feature importance (what influenced the decision?)
- Attention visualization (what did the model focus on?)
- Counterfactual explanations (what would change the outcome?)
- Natural language explanations (human-readable reasoning)

This is CRITICAL for AGI - humans must understand and trust AI decisions.

Part of ShivX 10/10 AGI transformation (Phase 5).
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Represents an explanation for a decision"""
    decision: Any
    method: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    attention_weights: Optional[np.ndarray] = None
    counterfactuals: List[Dict[str, Any]] = field(default_factory=list)
    natural_language: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FeatureAttributor:
    """
    Feature Attribution - Which features influenced the decision?

    Uses gradient-based attribution methods.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize Feature Attributor.

        Args:
            model: Neural network model to explain
        """
        self.model = model

    def attribute(
        self,
        input_data: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute feature attributions.

        Args:
            input_data: Input to explain
            target_class: Target class (for classification)

        Returns:
            Dictionary of feature importances
        """
        self.model.eval()

        input_data.requires_grad_(True)

        # Forward pass
        output = self.model(input_data)

        if target_class is None:
            # Use predicted class
            target_class = torch.argmax(output, dim=-1).item()

        # Get score for target class
        score = output[0, target_class]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Get gradients
        gradients = input_data.grad.data.cpu().numpy()[0]

        # Compute attributions (gradient * input)
        input_np = input_data.detach().cpu().numpy()[0]
        attributions = gradients * input_np

        # Create feature importance dict
        feature_importance = {
            f"feature_{i}": float(attr)
            for i, attr in enumerate(attributions)
        }

        return feature_importance

    def integrated_gradients(
        self,
        input_data: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50,
    ) -> Dict[str, float]:
        """
        Integrated Gradients - More robust attribution method.

        Args:
            input_data: Input to explain
            baseline: Baseline input (default: zeros)
            num_steps: Number of integration steps

        Returns:
            Feature attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(input_data)

        # Interpolate between baseline and input
        alphas = torch.linspace(0, 1, num_steps).to(input_data.device)

        gradients = []

        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            output = self.model(interpolated)
            target_class = torch.argmax(output, dim=-1).item()
            score = output[0, target_class]

            # Backward pass
            self.model.zero_grad()
            score.backward()

            # Store gradient
            gradients.append(interpolated.grad.data.cpu().numpy()[0])

        # Average gradients
        avg_gradients = np.mean(gradients, axis=0)

        # Multiply by input difference
        input_diff = (input_data - baseline).detach().cpu().numpy()[0]
        attributions = avg_gradients * input_diff

        feature_importance = {
            f"feature_{i}": float(attr)
            for i, attr in enumerate(attributions)
        }

        return feature_importance


class AttentionVisualizer:
    """
    Attention Visualization - Show what model attends to.

    For transformer-based models with attention mechanisms.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize Attention Visualizer.

        Args:
            model: Model with attention mechanism
        """
        self.model = model

    def extract_attention(
        self,
        input_data: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """
        Extract attention weights from model.

        Args:
            input_data: Input data

        Returns:
            Attention weights (or None if not available)
        """
        self.model.eval()

        # Check if model has attention
        if not hasattr(self.model, 'get_attention'):
            logger.warning("Model does not have attention mechanism")
            return None

        with torch.no_grad():
            output, attention = self.model.get_attention(input_data)

        return attention.cpu().numpy()

    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        tokens: Optional[List[str]] = None,
    ) -> str:
        """
        Create text visualization of attention.

        Args:
            attention_weights: Attention matrix
            tokens: Token labels (optional)

        Returns:
            Text visualization
        """
        if tokens is None:
            tokens = [f"token_{i}" for i in range(attention_weights.shape[0])]

        # Find highest attention weights
        top_indices = np.argsort(attention_weights.flatten())[-5:]

        visualization = "Top attention weights:\n"
        for idx in reversed(top_indices):
            row = idx // attention_weights.shape[1]
            col = idx % attention_weights.shape[1]
            weight = attention_weights[row, col]

            visualization += f"  {tokens[row]} -> {tokens[col]}: {weight:.3f}\n"

        return visualization


class CounterfactualGenerator:
    """
    Counterfactual Explanations - What changes would alter the decision?

    "If X were different, the prediction would change to Y"
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize Counterfactual Generator.

        Args:
            model: Model to explain
        """
        self.model = model

    def generate_counterfactuals(
        self,
        input_data: torch.Tensor,
        original_prediction: int,
        target_class: Optional[int] = None,
        num_samples: int = 10,
        perturbation_std: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual examples.

        Args:
            input_data: Original input
            original_prediction: Original predicted class
            target_class: Desired target class (None = any different class)
            num_samples: Number of counterfactuals to generate
            perturbation_std: Standard deviation of perturbations

        Returns:
            List of counterfactual examples
        """
        self.model.eval()

        counterfactuals = []

        for _ in range(num_samples):
            # Add random perturbation
            perturbation = torch.randn_like(input_data) * perturbation_std
            perturbed = input_data + perturbation

            # Predict on perturbed input
            with torch.no_grad():
                output = self.model(perturbed)
                pred_class = torch.argmax(output, dim=-1).item()

            # Check if prediction changed
            if target_class is not None:
                if pred_class == target_class:
                    # Found counterfactual!
                    counterfactual = {
                        "input": perturbed.cpu().numpy()[0],
                        "prediction": pred_class,
                        "confidence": torch.softmax(output, dim=-1)[0, pred_class].item(),
                        "perturbation": perturbation.cpu().numpy()[0],
                    }
                    counterfactuals.append(counterfactual)
            else:
                if pred_class != original_prediction:
                    # Any different prediction counts
                    counterfactual = {
                        "input": perturbed.cpu().numpy()[0],
                        "prediction": pred_class,
                        "confidence": torch.softmax(output, dim=-1)[0, pred_class].item(),
                        "perturbation": perturbation.cpu().numpy()[0],
                    }
                    counterfactuals.append(counterfactual)

        logger.info(f"Generated {len(counterfactuals)} counterfactuals")

        return counterfactuals

    def find_minimal_change(
        self,
        input_data: torch.Tensor,
        original_prediction: int,
        target_class: int,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
    ) -> Optional[torch.Tensor]:
        """
        Find minimal change to input that changes prediction.

        Args:
            input_data: Original input
            original_prediction: Original prediction
            target_class: Target class
            learning_rate: Optimization learning rate
            max_iterations: Maximum optimization iterations

        Returns:
            Minimal counterfactual (or None if not found)
        """
        self.model.eval()

        # Start with original input
        counterfactual = input_data.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([counterfactual], lr=learning_rate)

        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Forward pass
            output = self.model(counterfactual)

            # Loss: maximize target class probability + minimize distance
            target_prob = output[0, target_class]
            distance = torch.norm(counterfactual - input_data)

            loss = -target_prob + 0.1 * distance

            # Backward pass
            loss.backward()
            optimizer.step()

            # Check if we've changed the prediction
            with torch.no_grad():
                pred_class = torch.argmax(output, dim=-1).item()
                if pred_class == target_class:
                    logger.info(f"Found counterfactual at iteration {iteration}")
                    return counterfactual.detach()

        logger.warning("Failed to find counterfactual")
        return None


class NaturalLanguageExplainer:
    """
    Natural Language Explanations - Human-readable reasoning.

    Converts technical explanations into natural language.
    """

    def __init__(self):
        """Initialize Natural Language Explainer"""
        pass

    def explain_classification(
        self,
        prediction: int,
        class_names: List[str],
        feature_importance: Dict[str, float],
        confidence: float,
    ) -> str:
        """
        Generate natural language explanation for classification.

        Args:
            prediction: Predicted class
            class_names: Class labels
            feature_importance: Feature attributions
            confidence: Prediction confidence

        Returns:
            Natural language explanation
        """
        pred_label = class_names[prediction] if prediction < len(class_names) else f"class_{prediction}"

        explanation = f"I predicted '{pred_label}' with {confidence:.1%} confidence.\n\n"

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation += "Key factors influencing this decision:\n"

        for i, (feature, importance) in enumerate(sorted_features[:5]):
            direction = "increased" if importance > 0 else "decreased"
            explanation += f"  {i+1}. {feature} {direction} the likelihood (importance: {abs(importance):.3f})\n"

        return explanation

    def explain_counterfactual(
        self,
        original_prediction: int,
        counterfactual_prediction: int,
        class_names: List[str],
        changes: Dict[str, float],
    ) -> str:
        """
        Explain what changes led to different prediction.

        Args:
            original_prediction: Original prediction
            counterfactual_prediction: Counterfactual prediction
            class_names: Class labels
            changes: Feature changes

        Returns:
            Natural language explanation
        """
        orig_label = class_names[original_prediction] if original_prediction < len(class_names) else f"class_{original_prediction}"
        cf_label = class_names[counterfactual_prediction] if counterfactual_prediction < len(class_names) else f"class_{counterfactual_prediction}"

        explanation = f"If the following features were different, the prediction would change from '{orig_label}' to '{cf_label}':\n\n"

        # Sort by change magnitude
        sorted_changes = sorted(
            changes.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for i, (feature, change) in enumerate(sorted_changes[:5]):
            direction = "increased" if change > 0 else "decreased"
            explanation += f"  {i+1}. {feature} {direction} by {abs(change):.3f}\n"

        return explanation


class ExplainabilityEngine:
    """
    Main explainability system.

    Provides comprehensive explanations for model decisions.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initialize Explainability Engine.

        Args:
            model: Model to explain
        """
        self.model = model

        self.attributor = FeatureAttributor(model)
        self.attention_viz = AttentionVisualizer(model)
        self.counterfactual_gen = CounterfactualGenerator(model)
        self.nl_explainer = NaturalLanguageExplainer()

        logger.info("Explainability Engine initialized")

    def explain(
        self,
        input_data: torch.Tensor,
        class_names: Optional[List[str]] = None,
        explain_counterfactuals: bool = True,
    ) -> Explanation:
        """
        Generate comprehensive explanation for input.

        Args:
            input_data: Input to explain
            class_names: Class labels
            explain_counterfactuals: Whether to generate counterfactuals

        Returns:
            Comprehensive explanation
        """
        self.model.eval()

        # Get prediction
        with torch.no_grad():
            output = self.model(input_data)
            prediction = torch.argmax(output, dim=-1).item()
            confidence = torch.softmax(output, dim=-1)[0, prediction].item()

        # Feature attribution
        feature_importance = self.attributor.attribute(input_data, prediction)

        # Attention visualization
        attention_weights = self.attention_viz.extract_attention(input_data)

        # Counterfactuals
        counterfactuals = []
        if explain_counterfactuals:
            counterfactuals = self.counterfactual_gen.generate_counterfactuals(
                input_data,
                prediction,
                num_samples=5,
            )

        # Natural language explanation
        nl_explanation = self.nl_explainer.explain_classification(
            prediction,
            class_names or [],
            feature_importance,
            confidence,
        )

        # Create explanation object
        explanation = Explanation(
            decision=prediction,
            method="comprehensive",
            feature_importance=feature_importance,
            attention_weights=attention_weights,
            counterfactuals=counterfactuals,
            natural_language=nl_explanation,
            confidence=confidence,
        )

        logger.info(f"Generated explanation for prediction: {prediction}")

        return explanation

    def explain_simple(
        self,
        input_data: torch.Tensor,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate simple text explanation.

        Args:
            input_data: Input to explain
            class_names: Class labels

        Returns:
            Simple natural language explanation
        """
        explanation = self.explain(
            input_data,
            class_names,
            explain_counterfactuals=False,
        )

        return explanation.natural_language

    def get_stats(self) -> Dict[str, Any]:
        """Get explainability statistics"""
        return {
            "model_type": type(self.model).__name__,
            "has_attention": hasattr(self.model, 'get_attention'),
        }


# Convenience function
def quick_explain(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Quick explanation generation.

    Args:
        model: Model to explain
        input_data: Input to explain
        class_names: Class labels

    Returns:
        Natural language explanation
    """
    explainer = ExplainabilityEngine(model)
    explanation = explainer.explain_simple(input_data, class_names)

    return explanation
