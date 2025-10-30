"""
Multi-Modal Perception System

Enables AGI to understand and process visual, sensory, and multi-modal inputs,
grounding language understanding in perceptual experience.

Key capabilities:
- Visual perception (image understanding, object detection, scene analysis)
- Multi-modal fusion (integrating different sensory modalities)
- Perceptual grounding (connecting language to perception)
- Visual reasoning and question answering
"""

from .visual_processor import (
    VisualProcessor,
    ImageFeatures,
    ObjectDetection,
    SceneUnderstanding,
    VisualConcept
)
from .multimodal_fusion import (
    MultiModalFusion,
    SensoryInput,
    FusedRepresentation,
    ModalityType
)
from .grounding_engine import (
    GroundingEngine,
    GroundedConcept,
    VisualQuestionAnswer,
    ImageCaption
)

__all__ = [
    "VisualProcessor",
    "ImageFeatures",
    "ObjectDetection",
    "SceneUnderstanding",
    "VisualConcept",
    "MultiModalFusion",
    "SensoryInput",
    "FusedRepresentation",
    "ModalityType",
    "GroundingEngine",
    "GroundedConcept",
    "VisualQuestionAnswer",
    "ImageCaption",
]
