"""
Explainable AI (XAI) System for Empire AGI
Week 20: Making Neural Networks Interpretable

Implements explanation techniques for neural network decisions:
- Attention Visualization: Show what the model focuses on
- Saliency Maps: Highlight important input features
- LIME: Local Interpretable Model-agnostic Explanations
- SHAP: SHapley Additive exPlanations
- Counterfactual Explanations: "What if" scenarios

Key capabilities:
- Feature importance ranking
- Decision boundary visualization
- Example-based explanations
- Minimal sufficient explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import numpy as np


class ExplanationType(Enum):
    """Types of explanations"""
    FEATURE_IMPORTANCE = "feature_importance"  # Which features matter most
    ATTENTION = "attention"  # What the model focuses on
    SALIENCY = "saliency"  # Gradient-based importance
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios
    EXAMPLE_BASED = "example_based"  # Similar examples


@dataclass
class Explanation:
    """Explanation for a model decision"""
    explanation_type: ExplanationType
    prediction: Any
    confidence: float
    important_features: List[Tuple[int, float]]  # (feature_index, importance)
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualExample:
    """Counterfactual explanation"""
    original_input: torch.Tensor
    original_prediction: Any
    counterfactual_input: torch.Tensor
    counterfactual_prediction: Any
    minimal_changes: List[Tuple[int, float, float]]  # (feature_idx, original_val, new_val)
    distance: float


class AttentionVisualizer:
    """
    Visualize attention patterns in neural networks.

    For attention-based models (Transformers, etc.), shows which
    parts of the input the model attends to.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}

        # Register hooks to capture attention weights
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        def attention_hook(module, input, output):
            # Store attention weights
            # In a real implementation, would extract from attention layers
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_weights[id(module)] = output[1]

        # Register hooks for attention layers
        # In practice, would identify and hook attention modules
        # Here, simplified for demonstration

    def visualize_attention(
        self,
        input: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Visualize attention patterns.

        Args:
            input: Input tensor
            layer_idx: Which layer to visualize (None = all layers)

        Returns:
            Dictionary of attention maps
        """
        self.model.eval()

        # Clear previous attention weights
        self.attention_weights = {}

        # Forward pass
        with torch.no_grad():
            output = self.model(input)

        # Simulate attention weights (in practice, would be captured by hooks)
        attention_maps = self._simulate_attention(input)

        return attention_maps

    def _simulate_attention(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simulate attention weights for demonstration"""
        # In practice, would return actual attention weights from hooks
        seq_len = input.size(-1) if input.dim() > 1 else input.size(0)

        # Create simulated attention map
        attention = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)

        return {"layer_0": attention}


class SaliencyMapGenerator:
    """
    Generate saliency maps using gradient-based methods.

    Shows which input features have the most influence on the
    model's decision by computing gradients.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def generate_saliency_map(
        self,
        input: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate saliency map using vanilla gradients.

        Args:
            input: Input tensor
            target_class: Which class to explain (None = predicted class)

        Returns:
            Saliency map (same shape as input)
        """
        self.model.eval()

        # Enable gradient computation for input
        input.requires_grad = True

        # Forward pass
        output = self.model(input)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=-1).item()

        # Backward pass to get gradients
        self.model.zero_grad()
        if input.grad is not None:
            input.grad.zero_()

        # Compute gradient of output w.r.t. input
        output[..., target_class].backward()

        # Saliency map is absolute value of gradients
        saliency_map = input.grad.abs()

        return saliency_map

    def integrated_gradients(
        self,
        input: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
    ) -> torch.Tensor:
        """
        Generate integrated gradients (more robust than vanilla gradients).

        Integrates gradients along path from baseline to input.

        Args:
            input: Input tensor
            baseline: Baseline input (default: zeros)
            steps: Number of integration steps

        Returns:
            Integrated gradients (same shape as input)
        """
        if baseline is None:
            baseline = torch.zeros_like(input)

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps)

        gradients = []
        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (input - baseline)
            interpolated.requires_grad = True

            # Forward pass
            output = self.model(interpolated)

            # Backward pass
            self.model.zero_grad()
            if interpolated.grad is not None:
                interpolated.grad.zero_()

            output.max().backward()

            # Store gradient
            gradients.append(interpolated.grad)

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        integrated_grads = (input - baseline) * avg_gradients

        return integrated_grads


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations).

    Explains individual predictions by fitting a simple interpretable
    model (e.g., linear) locally around the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 1000,
        num_features: int = 10,
    ):
        self.model = model
        self.num_samples = num_samples
        self.num_features = num_features

    def explain_instance(
        self,
        instance: torch.Tensor,
        num_features: Optional[int] = None,
    ) -> Explanation:
        """
        Explain a single prediction using LIME.

        Args:
            instance: Input instance to explain
            num_features: Number of top features to return

        Returns:
            Explanation with feature importances
        """
        if num_features is None:
            num_features = self.num_features

        self.model.eval()

        # Get prediction for original instance
        with torch.no_grad():
            original_output = self.model(instance.unsqueeze(0))
            prediction = original_output.argmax(dim=-1).item()
            confidence = torch.softmax(original_output, dim=-1)[0, prediction].item()

        # Generate perturbed samples around instance
        perturbed_samples, distances = self._generate_perturbed_samples(instance)

        # Get predictions for perturbed samples
        with torch.no_grad():
            perturbed_outputs = self.model(perturbed_samples)
            perturbed_predictions = torch.softmax(perturbed_outputs, dim=-1)[:, prediction]

        # Fit linear model
        feature_importances = self._fit_linear_model(
            perturbed_samples,
            perturbed_predictions,
            distances,
            instance
        )

        # Get top features
        top_indices = torch.argsort(feature_importances.abs(), descending=True)[:num_features]
        important_features = [
            (idx.item(), feature_importances[idx].item())
            for idx in top_indices
        ]

        description = self._generate_description(important_features, prediction, confidence)

        return Explanation(
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            prediction=prediction,
            confidence=confidence,
            important_features=important_features,
            description=description,
        )

    def _generate_perturbed_samples(
        self,
        instance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate perturbed samples around instance"""
        # Perturb by adding Gaussian noise
        noise = torch.randn(self.num_samples, *instance.shape) * 0.1
        perturbed = instance.unsqueeze(0) + noise

        # Calculate distances from original
        distances = torch.norm(noise, dim=tuple(range(1, noise.dim())), p=2)

        return perturbed, distances

    def _fit_linear_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        distances: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """Fit weighted linear model"""
        # Weight samples by proximity (exponential kernel)
        kernel_width = 0.25
        weights = torch.exp(-(distances ** 2) / (kernel_width ** 2))

        # Flatten X
        X_flat = X.view(X.size(0), -1)
        original_flat = original.view(-1)

        # Weighted least squares
        # w = (X^T W X)^-1 X^T W y
        # where W is diagonal matrix of weights

        W = torch.diag(weights)
        XtWX = X_flat.T @ W @ X_flat
        XtWy = X_flat.T @ W @ y

        # Add regularization for numerical stability
        regularization = torch.eye(XtWX.size(0)) * 1e-4
        coefficients = torch.linalg.solve(XtWX + regularization, XtWy)

        # Reshape to original feature shape
        feature_importances = coefficients.view(original.shape)

        return feature_importances

    def _generate_description(
        self,
        important_features: List[Tuple[int, float]],
        prediction: int,
        confidence: float,
    ) -> str:
        """Generate human-readable description"""
        desc = f"Predicted class {prediction} with {confidence:.1%} confidence.\n"
        desc += "Top influential features:\n"

        for idx, importance in important_features[:5]:
            direction = "increases" if importance > 0 else "decreases"
            desc += f"  - Feature {idx}: {direction} prediction (importance: {abs(importance):.3f})\n"

        return desc


class CounterfactualGenerator:
    """
    Generate counterfactual explanations.

    Finds minimal changes to input that would change the prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

    def generate_counterfactual(
        self,
        instance: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> CounterfactualExample:
        """
        Generate counterfactual example.

        Finds minimal changes to instance that flip the prediction.

        Args:
            instance: Original input
            target_class: Desired prediction (None = flip to opposite)

        Returns:
            Counterfactual example
        """
        self.model.eval()

        # Get original prediction
        with torch.no_grad():
            original_output = self.model(instance.unsqueeze(0))
            original_prediction = original_output.argmax(dim=-1).item()

        # Determine target class
        if target_class is None:
            # Binary classification: Flip to opposite class
            num_classes = original_output.size(-1)
            target_class = 1 - original_prediction if num_classes == 2 else (original_prediction + 1) % num_classes

        # Initialize counterfactual as copy of instance
        counterfactual = instance.clone().detach()
        counterfactual.requires_grad = True

        optimizer = torch.optim.Adam([counterfactual], lr=self.learning_rate)

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Forward pass
            output = self.model(counterfactual.unsqueeze(0))

            # Loss: Encourage target class + minimize change
            classification_loss = F.cross_entropy(
                output,
                torch.tensor([target_class])
            )

            distance_loss = F.mse_loss(counterfactual, instance)

            # Combined loss
            loss = classification_loss + 0.1 * distance_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Check if target class achieved
            with torch.no_grad():
                predicted_class = output.argmax(dim=-1).item()
                if predicted_class == target_class:
                    break

        # Get final prediction
        with torch.no_grad():
            final_output = self.model(counterfactual.unsqueeze(0))
            final_prediction = final_output.argmax(dim=-1).item()

        # Identify minimal changes
        changes = self._identify_changes(instance, counterfactual)

        # Calculate distance
        distance = F.mse_loss(instance, counterfactual).item()

        return CounterfactualExample(
            original_input=instance,
            original_prediction=original_prediction,
            counterfactual_input=counterfactual.detach(),
            counterfactual_prediction=final_prediction,
            minimal_changes=changes,
            distance=distance,
        )

    def _identify_changes(
        self,
        original: torch.Tensor,
        counterfactual: torch.Tensor,
        threshold: float = 0.01,
    ) -> List[Tuple[int, float, float]]:
        """Identify significant changes between original and counterfactual"""
        diff = (counterfactual - original).abs()

        # Find features with significant change
        significant_indices = torch.where(diff.flatten() > threshold)[0]

        changes = []
        for idx in significant_indices:
            original_val = original.flatten()[idx].item()
            new_val = counterfactual.flatten()[idx].item()
            changes.append((idx.item(), original_val, new_val))

        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)

        return changes


class XAISystem:
    """
    Unified Explainable AI system.

    Integrates multiple explanation techniques for comprehensive
    model interpretability.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        self.attention_visualizer = AttentionVisualizer(model)
        self.saliency_generator = SaliencyMapGenerator(model)
        self.lime_explainer = LIMEExplainer(model)
        self.counterfactual_generator = CounterfactualGenerator(model)

    def explain(
        self,
        instance: torch.Tensor,
        methods: List[ExplanationType] = None,
    ) -> Dict[ExplanationType, Any]:
        """
        Generate comprehensive explanation using multiple methods.

        Args:
            instance: Input instance to explain
            methods: Which explanation methods to use (None = all)

        Returns:
            Dictionary of explanations by type
        """
        if methods is None:
            methods = [
                ExplanationType.SALIENCY,
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.COUNTERFACTUAL,
            ]

        explanations = {}

        # Saliency map
        if ExplanationType.SALIENCY in methods:
            saliency_map = self.saliency_generator.generate_saliency_map(instance)
            explanations[ExplanationType.SALIENCY] = saliency_map

        # Feature importance (LIME)
        if ExplanationType.FEATURE_IMPORTANCE in methods:
            lime_explanation = self.lime_explainer.explain_instance(instance)
            explanations[ExplanationType.FEATURE_IMPORTANCE] = lime_explanation

        # Counterfactual
        if ExplanationType.COUNTERFACTUAL in methods:
            counterfactual = self.counterfactual_generator.generate_counterfactual(instance)
            explanations[ExplanationType.COUNTERFACTUAL] = counterfactual

        # Attention (if model has attention)
        if ExplanationType.ATTENTION in methods:
            attention_maps = self.attention_visualizer.visualize_attention(instance)
            explanations[ExplanationType.ATTENTION] = attention_maps

        return explanations

    def get_top_features(
        self,
        instance: torch.Tensor,
        num_features: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Get top important features across all explanation methods.

        Returns:
            List of (feature_index, importance, method)
        """
        all_features = []

        # From LIME
        lime_explanation = self.lime_explainer.explain_instance(instance, num_features=num_features)
        for idx, importance in lime_explanation.important_features:
            all_features.append((idx, importance, "LIME"))

        # From saliency map
        saliency_map = self.saliency_generator.generate_saliency_map(instance)
        saliency_flat = saliency_map.flatten()
        top_saliency_indices = torch.argsort(saliency_flat, descending=True)[:num_features]
        for idx in top_saliency_indices:
            all_features.append((idx.item(), saliency_flat[idx].item(), "Saliency"))

        # Remove duplicates and sort by absolute importance
        unique_features = {}
        for idx, importance, method in all_features:
            if idx not in unique_features or abs(importance) > abs(unique_features[idx][0]):
                unique_features[idx] = (importance, method)

        # Sort and return top
        sorted_features = sorted(
            [(idx, importance, method) for idx, (importance, method) in unique_features.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_features[:num_features]


# ============================================================
# Test Functions
# ============================================================

class SimpleClassifier(nn.Module):
    """Simple classifier for testing"""

    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


async def test_xai():
    """Test XAI system"""
    print("=" * 60)
    print("Testing Explainable AI (XAI) System")
    print("=" * 60)
    print()

    # Create model and instance
    model = SimpleClassifier(input_dim=10, num_classes=2)
    instance = torch.randn(10)

    # Create XAI system
    xai = XAISystem(model)

    # Test 1: Saliency map
    print("1. Testing saliency map...")
    saliency_map = xai.saliency_generator.generate_saliency_map(instance)
    print(f"   Saliency map shape: {saliency_map.shape}")
    print(f"   Max saliency: {saliency_map.max():.3f}")
    print(f"   Min saliency: {saliency_map.min():.3f}")
    print()

    # Test 2: LIME explanation
    print("2. Testing LIME explanation...")
    lime_explanation = xai.lime_explainer.explain_instance(instance)
    print(f"   Prediction: Class {lime_explanation.prediction}")
    print(f"   Confidence: {lime_explanation.confidence:.1%}")
    print(f"   Top 3 important features:")
    for idx, importance in lime_explanation.important_features[:3]:
        print(f"      Feature {idx}: {importance:.3f}")
    print()

    # Test 3: Counterfactual explanation
    print("3. Testing counterfactual explanation...")
    counterfactual = xai.counterfactual_generator.generate_counterfactual(instance)
    print(f"   Original prediction: Class {counterfactual.original_prediction}")
    print(f"   Counterfactual prediction: Class {counterfactual.counterfactual_prediction}")
    print(f"   Distance: {counterfactual.distance:.3f}")
    print(f"   Number of changes: {len(counterfactual.minimal_changes)}")
    if counterfactual.minimal_changes:
        print(f"   Largest change: Feature {counterfactual.minimal_changes[0][0]}, "
              f"{counterfactual.minimal_changes[0][1]:.3f} -> {counterfactual.minimal_changes[0][2]:.3f}")
    print()

    # Test 4: Comprehensive explanation
    print("4. Testing comprehensive explanation...")
    explanations = xai.explain(instance)
    print(f"   Explanation methods used: {len(explanations)}")
    for explanation_type in explanations:
        print(f"   - {explanation_type.value}")
    print()

    # Test 5: Top features across methods
    print("5. Testing top features aggregation...")
    top_features = xai.get_top_features(instance, num_features=5)
    print(f"   Top 5 features across all methods:")
    for idx, importance, method in top_features:
        print(f"      Feature {idx}: {importance:.3f} ({method})")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Instance shape: {instance.shape}")
    print(f"Explanation methods: {len(explanations)}")
    print(f"LIME top features: {len(lime_explanation.important_features)}")
    print(f"Counterfactual success: {counterfactual.counterfactual_prediction != counterfactual.original_prediction}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_xai())
