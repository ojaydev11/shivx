"""
Multi-Modal Fusion System

Integrates information from multiple sensory modalities to create unified
perceptual representations.

Key capabilities:
- Multi-modal sensor fusion
- Cross-modal attention and alignment
- Temporal synchronization
- Modality-specific processing
- Unified perceptual representation
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import time


class ModalityType(str, Enum):
    """Types of sensory modalities"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    TEXTUAL = "textual"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


class FusionStrategy(str, Enum):
    """Strategies for fusing multi-modal information"""
    EARLY = "early"  # Fuse at feature level
    LATE = "late"  # Fuse at decision level
    HYBRID = "hybrid"  # Combination of both
    ATTENTION = "attention"  # Attention-based fusion


@dataclass
class SensoryInput:
    """Input from a single sensory modality"""
    input_id: str
    modality: ModalityType
    data: Any  # Raw data (could be array, text, etc.)
    features: Optional[np.ndarray] = None  # Extracted features
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    synchronized: bool = False

    def extract_features(self, feature_extractor: Any = None) -> np.ndarray:
        """Extract features from raw data"""
        if self.features is not None:
            return self.features

        # Simulate feature extraction
        # In production, use modality-specific extractors
        if isinstance(self.data, np.ndarray):
            self.features = self.data.flatten()
        elif isinstance(self.data, str):
            # Text embedding simulation
            self.features = np.random.randn(512).astype(np.float32)
        else:
            # Generic feature extraction
            self.features = np.random.randn(256).astype(np.float32)

        # Normalize
        self.features = self.features / (np.linalg.norm(self.features) + 1e-8)
        return self.features


@dataclass
class ModalityAlignment:
    """Alignment information between modalities"""
    modality1: ModalityType
    modality2: ModalityType
    alignment_score: float  # 0-1, how well aligned
    temporal_offset: float = 0.0  # Time offset in seconds
    spatial_correspondence: Optional[np.ndarray] = None  # Spatial alignment matrix
    confidence: float = 1.0


@dataclass
class FusedRepresentation:
    """Unified multi-modal representation"""
    representation_id: str
    modalities: List[ModalityType]
    fused_features: np.ndarray
    component_features: Dict[ModalityType, np.ndarray]
    fusion_strategy: FusionStrategy
    attention_weights: Optional[Dict[ModalityType, float]] = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_modality_contribution(self, modality: ModalityType) -> float:
        """Get contribution weight of a specific modality"""
        if self.attention_weights and modality in self.attention_weights:
            return self.attention_weights[modality]
        return 1.0 / len(self.modalities) if self.modalities else 0.0

    def dominant_modality(self) -> Optional[ModalityType]:
        """Get the most dominant modality in the fusion"""
        if not self.attention_weights:
            return None
        return max(self.attention_weights.items(), key=lambda x: x[1])[0]


@dataclass
class CrossModalPattern:
    """Learned pattern across modalities"""
    pattern_id: str
    modality_combination: Set[ModalityType]
    pattern_features: np.ndarray
    examples: List[str] = field(default_factory=list)  # Example fusion IDs
    occurrence_count: int = 0
    confidence: float = 1.0
    description: Optional[str] = None


class MultiModalFusion:
    """
    Multi-modal fusion system for unified perception

    Features:
    - Multi-modal input processing
    - Feature extraction and alignment
    - Temporal synchronization
    - Cross-modal attention
    - Fusion strategy selection
    - Learned cross-modal patterns
    """

    def __init__(self):
        self.input_counter = 0
        self.fusion_counter = 0

        # Sensory input buffers (for temporal synchronization)
        self.input_buffers: Dict[ModalityType, deque] = {
            modality: deque(maxlen=100) for modality in ModalityType
        }

        # Fusion history
        self.fusions: Dict[str, FusedRepresentation] = {}

        # Learned cross-modal patterns
        self.patterns: Dict[str, CrossModalPattern] = {}

        # Modality-specific feature extractors (simulated)
        self.feature_dims = {
            ModalityType.VISUAL: 512,
            ModalityType.AUDITORY: 256,
            ModalityType.TACTILE: 128,
            ModalityType.PROPRIOCEPTIVE: 64,
            ModalityType.TEXTUAL: 512,
            ModalityType.NUMERIC: 64,
            ModalityType.TEMPORAL: 32,
            ModalityType.SPATIAL: 128,
        }

        # Fusion parameters
        self.temporal_window = 1.0  # seconds
        self.alignment_threshold = 0.5
        self.default_fusion_strategy = FusionStrategy.ATTENTION

    def add_sensory_input(
        self,
        data: Any,
        modality: ModalityType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SensoryInput:
        """
        Add sensory input to the system

        Args:
            data: Raw sensory data
            modality: Type of sensory modality
            metadata: Optional metadata

        Returns:
            SensoryInput object
        """
        self.input_counter += 1
        input_id = f"{modality.value}_{self.input_counter}"

        sensory_input = SensoryInput(
            input_id=input_id,
            modality=modality,
            data=data,
            metadata=metadata or {},
            confidence=1.0
        )

        # Extract features
        sensory_input.extract_features()

        # Add to buffer
        self.input_buffers[modality].append(sensory_input)

        return sensory_input

    def synchronize_modalities(
        self,
        target_modalities: List[ModalityType],
        time_window: Optional[float] = None
    ) -> Dict[ModalityType, List[SensoryInput]]:
        """
        Synchronize inputs from multiple modalities within a time window

        Args:
            target_modalities: Modalities to synchronize
            time_window: Time window in seconds (default: self.temporal_window)

        Returns:
            Dictionary mapping modalities to synchronized inputs
        """
        if time_window is None:
            time_window = self.temporal_window

        current_time = time.time()
        synchronized = {}

        for modality in target_modalities:
            if modality not in self.input_buffers:
                continue

            # Get inputs within time window
            recent_inputs = [
                inp for inp in self.input_buffers[modality]
                if (current_time - inp.timestamp) <= time_window
            ]

            if recent_inputs:
                synchronized[modality] = recent_inputs

        return synchronized

    def align_modalities(
        self,
        inputs: Dict[ModalityType, List[SensoryInput]]
    ) -> List[ModalityAlignment]:
        """
        Compute alignment between different modalities

        Args:
            inputs: Dictionary of inputs per modality

        Returns:
            List of alignment information
        """
        alignments = []
        modalities = list(inputs.keys())

        # Check all pairs
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                alignment = self._compute_alignment(
                    inputs[mod1],
                    inputs[mod2],
                    mod1,
                    mod2
                )
                if alignment and alignment.alignment_score >= self.alignment_threshold:
                    alignments.append(alignment)

        return alignments

    def _compute_alignment(
        self,
        inputs1: List[SensoryInput],
        inputs2: List[SensoryInput],
        mod1: ModalityType,
        mod2: ModalityType
    ) -> Optional[ModalityAlignment]:
        """Compute alignment between two modalities"""
        if not inputs1 or not inputs2:
            return None

        # Use most recent inputs
        inp1 = inputs1[-1]
        inp2 = inputs2[-1]

        # Temporal alignment
        temporal_offset = inp2.timestamp - inp1.timestamp

        # Feature similarity (as proxy for alignment quality)
        if inp1.features is not None and inp2.features is not None:
            # Project to common dimension
            common_dim = min(len(inp1.features), len(inp2.features))
            feat1 = inp1.features[:common_dim]
            feat2 = inp2.features[:common_dim]

            # Compute similarity
            similarity = np.dot(feat1, feat2) / (
                np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
            )
            alignment_score = (similarity + 1) / 2  # Normalize to 0-1
        else:
            alignment_score = 0.5

        return ModalityAlignment(
            modality1=mod1,
            modality2=mod2,
            alignment_score=alignment_score,
            temporal_offset=temporal_offset,
            confidence=min(inp1.confidence, inp2.confidence)
        )

    def fuse(
        self,
        inputs: Dict[ModalityType, SensoryInput],
        strategy: Optional[FusionStrategy] = None
    ) -> FusedRepresentation:
        """
        Fuse inputs from multiple modalities

        Args:
            inputs: Dictionary mapping modalities to their inputs
            strategy: Fusion strategy (default: self.default_fusion_strategy)

        Returns:
            FusedRepresentation with unified features
        """
        if not inputs:
            raise ValueError("No inputs to fuse")

        if strategy is None:
            strategy = self.default_fusion_strategy

        self.fusion_counter += 1
        representation_id = f"fusion_{self.fusion_counter}"

        modalities = list(inputs.keys())

        # Extract and organize features
        component_features = {}
        for modality, sensory_input in inputs.items():
            if sensory_input.features is None:
                sensory_input.extract_features()
            component_features[modality] = sensory_input.features

        # Apply fusion strategy
        if strategy == FusionStrategy.EARLY:
            fused_features = self._early_fusion(component_features)
            attention_weights = None

        elif strategy == FusionStrategy.LATE:
            fused_features = self._late_fusion(component_features)
            attention_weights = None

        elif strategy == FusionStrategy.ATTENTION:
            fused_features, attention_weights = self._attention_fusion(component_features)

        else:  # HYBRID
            fused_features, attention_weights = self._hybrid_fusion(component_features)

        # Calculate overall confidence
        confidences = [inp.confidence for inp in inputs.values()]
        overall_confidence = np.mean(confidences) if confidences else 0.0

        fused_rep = FusedRepresentation(
            representation_id=representation_id,
            modalities=modalities,
            fused_features=fused_features,
            component_features=component_features,
            fusion_strategy=strategy,
            attention_weights=attention_weights,
            confidence=overall_confidence,
            metadata={
                "num_modalities": len(modalities),
                "feature_dim": len(fused_features)
            }
        )

        self.fusions[representation_id] = fused_rep

        # Learn from this fusion
        self._update_patterns(fused_rep)

        return fused_rep

    def _early_fusion(self, features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """
        Early fusion: concatenate features at feature level

        Simple concatenation of all modality features
        """
        feature_list = [feat for feat in features.values()]

        # Concatenate all features
        fused = np.concatenate(feature_list)

        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def _late_fusion(self, features: Dict[ModalityType, np.ndarray]) -> np.ndarray:
        """
        Late fusion: weighted average of features

        Each modality contributes equally
        """
        feature_list = [feat for feat in features.values()]

        # Average features
        fused = np.mean(feature_list, axis=0)

        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused

    def _attention_fusion(
        self,
        features: Dict[ModalityType, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """
        Attention-based fusion: learn attention weights for each modality

        Weights based on feature statistics and learned patterns
        """
        # Compute attention weights based on feature norms and variance
        attention_weights = {}
        total_weight = 0.0

        for modality, feat in features.items():
            # Use feature norm and variance as attention signal
            norm = np.linalg.norm(feat)
            variance = np.var(feat)
            weight = norm * (1.0 + variance)

            attention_weights[modality] = weight
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            for modality in attention_weights:
                attention_weights[modality] /= total_weight
        else:
            # Equal weights if no signal
            equal_weight = 1.0 / len(features)
            attention_weights = {mod: equal_weight for mod in features.keys()}

        # Apply attention weights
        # First normalize all features to same dimension (use concatenation with weights)
        weighted_features = []
        for modality, feat in features.items():
            weight = attention_weights[modality]
            weighted_features.append(feat * weight)

        # Concatenate weighted features instead of summing (handles different dimensions)
        fused = np.concatenate(weighted_features)

        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused, attention_weights

    def _hybrid_fusion(
        self,
        features: Dict[ModalityType, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[ModalityType, float]]:
        """
        Hybrid fusion: combination of early and attention-based fusion

        Uses both concatenation and attention
        """
        # First apply attention
        attention_fused, attention_weights = self._attention_fusion(features)

        # Also create early fusion
        early_fused = self._early_fusion(features)

        # Combine both (50-50)
        fused = np.concatenate([attention_fused, early_fused])

        # Normalize
        fused = fused / (np.linalg.norm(fused) + 1e-8)

        return fused, attention_weights

    def _update_patterns(self, fusion: FusedRepresentation):
        """
        Update learned cross-modal patterns based on new fusion

        Detects recurring patterns across modalities
        """
        # Create pattern signature
        modality_set = set(fusion.modalities)
        modality_key = "_".join(sorted([m.value for m in modality_set]))

        # Find or create pattern
        if modality_key not in self.patterns:
            pattern_id = f"pattern_{len(self.patterns) + 1}"
            self.patterns[modality_key] = CrossModalPattern(
                pattern_id=pattern_id,
                modality_combination=modality_set,
                pattern_features=fusion.fused_features.copy(),
                examples=[fusion.representation_id],
                occurrence_count=1,
                confidence=0.5,
                description=f"Pattern across {', '.join([m.value for m in modality_set])}"
            )
        else:
            # Update existing pattern
            pattern = self.patterns[modality_key]
            pattern.occurrence_count += 1
            pattern.examples.append(fusion.representation_id)

            # Update pattern features (running average)
            alpha = 0.1  # Learning rate
            pattern.pattern_features = (
                (1 - alpha) * pattern.pattern_features +
                alpha * fusion.fused_features
            )

            # Increase confidence with more examples
            pattern.confidence = min(1.0, pattern.occurrence_count / 10.0)

    def match_pattern(
        self,
        fusion: FusedRepresentation,
        threshold: float = 0.7
    ) -> Optional[CrossModalPattern]:
        """
        Match a fusion against learned patterns

        Args:
            fusion: Fused representation to match
            threshold: Similarity threshold for matching

        Returns:
            Matching pattern or None
        """
        modality_set = set(fusion.modalities)
        modality_key = "_".join(sorted([m.value for m in modality_set]))

        if modality_key not in self.patterns:
            return None

        pattern = self.patterns[modality_key]

        # Compute similarity
        similarity = np.dot(fusion.fused_features, pattern.pattern_features)

        if similarity >= threshold:
            return pattern

        return None

    def get_dominant_modality(
        self,
        time_window: Optional[float] = None
    ) -> Optional[ModalityType]:
        """
        Determine which modality is currently most active

        Args:
            time_window: Time window to consider (default: self.temporal_window)

        Returns:
            Most active modality or None
        """
        if time_window is None:
            time_window = self.temporal_window

        current_time = time.time()
        activity_scores = {}

        for modality, buffer in self.input_buffers.items():
            # Count recent inputs
            recent_count = sum(
                1 for inp in buffer
                if (current_time - inp.timestamp) <= time_window
            )

            # Weight by recency and confidence
            if recent_count > 0:
                recent_inputs = [
                    inp for inp in buffer
                    if (current_time - inp.timestamp) <= time_window
                ]
                avg_confidence = np.mean([inp.confidence for inp in recent_inputs])
                activity_scores[modality] = recent_count * avg_confidence

        if not activity_scores:
            return None

        return max(activity_scores.items(), key=lambda x: x[1])[0]

    def cross_modal_inference(
        self,
        source_modality: ModalityType,
        target_modality: ModalityType,
        source_input: SensoryInput
    ) -> Optional[np.ndarray]:
        """
        Infer features in target modality from source modality

        Uses learned cross-modal patterns to predict one modality from another

        Args:
            source_modality: Source modality
            target_modality: Target modality to infer
            source_input: Input from source modality

        Returns:
            Inferred features for target modality or None
        """
        # Find patterns involving both modalities
        relevant_patterns = [
            p for p in self.patterns.values()
            if source_modality in p.modality_combination
            and target_modality in p.modality_combination
        ]

        if not relevant_patterns:
            return None

        # Use pattern with highest confidence
        best_pattern = max(relevant_patterns, key=lambda p: p.confidence)

        # Simple inference: use pattern features as proxy
        # In production, use learned transformation model
        if source_input.features is not None:
            # Weighted combination of source features and pattern
            alpha = best_pattern.confidence
            inferred = (
                alpha * best_pattern.pattern_features +
                (1 - alpha) * source_input.features[:len(best_pattern.pattern_features)]
            )
            return inferred

        return best_pattern.pattern_features.copy()

    def temporal_integration(
        self,
        modality: ModalityType,
        window_size: int = 5
    ) -> Optional[np.ndarray]:
        """
        Integrate information over time for a single modality

        Args:
            modality: Modality to integrate
            window_size: Number of recent inputs to integrate

        Returns:
            Temporally integrated features or None
        """
        if modality not in self.input_buffers:
            return None

        buffer = self.input_buffers[modality]
        if not buffer:
            return None

        # Get recent inputs
        recent = list(buffer)[-window_size:]

        if not recent:
            return None

        # Extract features
        feature_list = [
            inp.features for inp in recent
            if inp.features is not None
        ]

        if not feature_list:
            return None

        # Temporal pooling: weighted average (more recent = higher weight)
        weights = np.linspace(0.5, 1.0, len(feature_list))
        weights = weights / np.sum(weights)

        integrated = np.average(feature_list, axis=0, weights=weights)

        # Normalize
        integrated = integrated / (np.linalg.norm(integrated) + 1e-8)

        return integrated

    def get_fusion_history(
        self,
        modality_filter: Optional[Set[ModalityType]] = None,
        limit: int = 10
    ) -> List[FusedRepresentation]:
        """
        Get recent fusion history

        Args:
            modality_filter: Only include fusions with these modalities
            limit: Maximum number of fusions to return

        Returns:
            List of recent fusions
        """
        fusions = list(self.fusions.values())

        # Filter by modalities
        if modality_filter:
            fusions = [
                f for f in fusions
                if modality_filter.issubset(set(f.modalities))
            ]

        # Sort by timestamp (most recent first)
        fusions.sort(key=lambda f: f.timestamp, reverse=True)

        return fusions[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get multi-modal fusion statistics"""
        # Count inputs per modality
        input_counts = {
            modality.value: len(buffer)
            for modality, buffer in self.input_buffers.items()
        }

        # Pattern statistics
        pattern_stats = {
            "total_patterns": len(self.patterns),
            "high_confidence_patterns": sum(
                1 for p in self.patterns.values() if p.confidence >= 0.8
            )
        }

        return {
            "total_inputs": self.input_counter,
            "total_fusions": self.fusion_counter,
            "inputs_per_modality": input_counts,
            "learned_patterns": pattern_stats,
            "temporal_window": self.temporal_window,
            "default_strategy": self.default_fusion_strategy.value,
        }
