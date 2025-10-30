"""
Grounding Engine - Perception-Language Bridge

Connects language understanding with perceptual experience, enabling
grounded communication about the perceived world.

Key capabilities:
- Visual question answering (VQA)
- Image captioning and description
- Grounded language understanding
- Reference resolution
- Spatial language grounding
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import time
import re


class QuestionType(str, Enum):
    """Types of visual questions"""
    COUNT = "count"  # How many X?
    EXISTENCE = "existence"  # Is there X?
    ATTRIBUTE = "attribute"  # What color/size is X?
    LOCATION = "location"  # Where is X?
    RELATION = "relation"  # What is left of X?
    ACTION = "action"  # What is X doing?
    REASONING = "reasoning"  # Why/how questions
    DESCRIPTION = "description"  # Describe X


class ReferenceType(str, Enum):
    """Types of referring expressions"""
    DEFINITE = "definite"  # "the red car"
    INDEFINITE = "indefinite"  # "a person"
    DEMONSTRATIVE = "demonstrative"  # "that building"
    PRONOUN = "pronoun"  # "it", "they"
    SPATIAL = "spatial"  # "the one on the left"


@dataclass
class GroundedConcept:
    """A concept grounded in perception"""
    concept_id: str
    linguistic_form: str  # Natural language expression
    perceptual_features: np.ndarray  # Visual/sensory features
    object_ids: List[str] = field(default_factory=list)  # Linked objects
    attributes: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def matches(self, query: str) -> bool:
        """Check if concept matches a query string"""
        query_lower = query.lower()
        return (
            self.linguistic_form.lower() in query_lower or
            query_lower in self.linguistic_form.lower()
        )


@dataclass
class ReferringExpression:
    """A linguistic reference to a perceptual entity"""
    expression: str
    reference_type: ReferenceType
    target_object_id: Optional[str] = None
    candidate_objects: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    resolved: bool = False

    def describe(self) -> str:
        """Generate description of reference"""
        parts = [self.expression]
        if self.attributes:
            parts.append(f"with attributes: {self.attributes}")
        if self.spatial_relations:
            parts.append(f"and spatial relations: {len(self.spatial_relations)}")
        return " ".join(parts)


@dataclass
class VisualQuestionAnswer:
    """A visual question and its answer"""
    question_id: str
    question: str
    question_type: QuestionType
    answer: str
    confidence: float
    reasoning: str
    grounded_concepts: List[str] = field(default_factory=list)
    referenced_objects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "type": self.question_type.value
        }


@dataclass
class ImageCaption:
    """Generated caption for an image or scene"""
    caption_id: str
    caption_text: str
    scene_id: Optional[str] = None
    grounded_concepts: List[str] = field(default_factory=list)
    object_mentions: Dict[str, str] = field(default_factory=dict)  # object_id -> mention
    confidence: float = 1.0
    detail_level: str = "medium"  # low, medium, high
    timestamp: float = field(default_factory=time.time)


@dataclass
class GroundedDialog:
    """A dialog exchange grounded in perception"""
    dialog_id: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    active_objects: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_turn(self, speaker: str, utterance: str, grounded_refs: Optional[List[str]] = None):
        """Add a dialog turn"""
        turn = {
            "speaker": speaker,
            "utterance": utterance,
            "grounded_refs": grounded_refs or [],
            "timestamp": time.time()
        }
        self.turns.append(turn)

        if grounded_refs:
            self.active_objects.update(grounded_refs)


class GroundingEngine:
    """
    Perception-Language grounding system

    Features:
    - Visual question answering
    - Image captioning
    - Reference resolution
    - Grounded concept learning
    - Spatial language understanding
    - Multi-turn grounded dialog
    """

    def __init__(self, visual_processor=None, multimodal_fusion=None):
        self.visual_processor = visual_processor
        self.multimodal_fusion = multimodal_fusion

        # Grounded concept vocabulary
        self.grounded_concepts: Dict[str, GroundedConcept] = {}

        # VQA history
        self.vqa_history: Dict[str, VisualQuestionAnswer] = {}

        # Caption history
        self.captions: Dict[str, ImageCaption] = {}

        # Dialog management
        self.dialogs: Dict[str, GroundedDialog] = {}

        # Counters
        self.question_counter = 0
        self.caption_counter = 0
        self.dialog_counter = 0
        self.concept_counter = 0

        # Language patterns for question classification
        self._initialize_question_patterns()

    def _initialize_question_patterns(self):
        """Initialize patterns for question type classification"""
        self.question_patterns = {
            QuestionType.COUNT: [
                r"how many",
                r"count",
                r"number of"
            ],
            QuestionType.EXISTENCE: [
                r"is there",
                r"are there",
                r"does.*contain",
                r"can you see"
            ],
            QuestionType.ATTRIBUTE: [
                r"what (color|size|shape|type)",
                r"which (color|size|shape)",
                r"describe.*appearance"
            ],
            QuestionType.LOCATION: [
                r"where is",
                r"where are",
                r"location of",
                r"position of"
            ],
            QuestionType.RELATION: [
                r"what.*(?:left|right|above|below|next to)",
                r"what is.*(?:in front|behind|near)",
                r"relationship between"
            ],
            QuestionType.ACTION: [
                r"what.*doing",
                r"what.*happening",
                r"what action"
            ],
            QuestionType.REASONING: [
                r"why",
                r"how",
                r"explain",
                r"reason"
            ],
            QuestionType.DESCRIPTION: [
                r"describe",
                r"what do you see",
                r"tell me about",
                r"what's in"
            ]
        }

    def classify_question(self, question: str) -> QuestionType:
        """
        Classify the type of question

        Args:
            question: Natural language question

        Returns:
            Classified question type
        """
        question_lower = question.lower()

        for qtype, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return qtype

        # Default to description
        return QuestionType.DESCRIPTION

    def visual_question_answering(
        self,
        question: str,
        scene_understanding: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> VisualQuestionAnswer:
        """
        Answer a question about visual content

        Args:
            question: Natural language question
            scene_understanding: Scene understanding from visual processor
            context: Optional context information

        Returns:
            VisualQuestionAnswer with answer and reasoning
        """
        self.question_counter += 1
        question_id = f"vqa_{self.question_counter}"

        # Classify question type
        qtype = self.classify_question(question)

        # Route to appropriate handler
        if qtype == QuestionType.COUNT:
            result = self._answer_count_question(question, scene_understanding)
        elif qtype == QuestionType.EXISTENCE:
            result = self._answer_existence_question(question, scene_understanding)
        elif qtype == QuestionType.ATTRIBUTE:
            result = self._answer_attribute_question(question, scene_understanding)
        elif qtype == QuestionType.LOCATION:
            result = self._answer_location_question(question, scene_understanding)
        elif qtype == QuestionType.RELATION:
            result = self._answer_relation_question(question, scene_understanding)
        elif qtype == QuestionType.DESCRIPTION:
            result = self._answer_description_question(question, scene_understanding)
        else:
            result = {
                "answer": "I'm not sure how to answer that.",
                "confidence": 0.3,
                "reasoning": "Question type not fully supported",
                "grounded_concepts": [],
                "referenced_objects": []
            }

        vqa = VisualQuestionAnswer(
            question_id=question_id,
            question=question,
            question_type=qtype,
            answer=result["answer"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            grounded_concepts=result.get("grounded_concepts", []),
            referenced_objects=result.get("referenced_objects", [])
        )

        self.vqa_history[question_id] = vqa
        return vqa

    def _answer_count_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer counting questions"""
        # Extract what to count from question
        question_lower = question.lower()

        if not hasattr(scene, 'objects'):
            return {
                "answer": "0",
                "confidence": 0.5,
                "reasoning": "No objects detected in scene"
            }

        # Try to find object type in question
        for obj in scene.objects:
            label = obj.label.lower()
            if label in question_lower:
                # Count objects with this label
                count = sum(1 for o in scene.objects if o.label.lower() == label)
                return {
                    "answer": str(count),
                    "confidence": 0.85,
                    "reasoning": f"Counted {count} instances of '{obj.label}' in the scene",
                    "grounded_concepts": [obj.label],
                    "referenced_objects": [o.object_id for o in scene.objects if o.label == obj.label]
                }

        return {
            "answer": "0",
            "confidence": 0.6,
            "reasoning": "Could not identify object to count in question"
        }

    def _answer_existence_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer yes/no existence questions"""
        question_lower = question.lower()

        if not hasattr(scene, 'objects'):
            return {
                "answer": "no",
                "confidence": 0.7,
                "reasoning": "No objects detected in scene"
            }

        # Check for each detected object
        for obj in scene.objects:
            if obj.label.lower() in question_lower:
                return {
                    "answer": "yes",
                    "confidence": obj.confidence,
                    "reasoning": f"Found {obj.label} in the scene",
                    "grounded_concepts": [obj.label],
                    "referenced_objects": [obj.object_id]
                }

        return {
            "answer": "no",
            "confidence": 0.75,
            "reasoning": "Did not find the queried object in the scene"
        }

    def _answer_attribute_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer questions about object attributes"""
        question_lower = question.lower()

        if not hasattr(scene, 'objects'):
            return {
                "answer": "unknown",
                "confidence": 0.3,
                "reasoning": "No objects in scene to query attributes"
            }

        # Find object in question
        for obj in scene.objects:
            if obj.label.lower() in question_lower:
                # Determine which attribute
                if "color" in question_lower and "color" in obj.attributes:
                    return {
                        "answer": obj.attributes["color"],
                        "confidence": 0.7,
                        "reasoning": f"The {obj.label} appears to be {obj.attributes['color']}",
                        "grounded_concepts": [obj.label],
                        "referenced_objects": [obj.object_id]
                    }
                elif "size" in question_lower and "size" in obj.attributes:
                    return {
                        "answer": obj.attributes["size"],
                        "confidence": 0.7,
                        "reasoning": f"The {obj.label} appears to be {obj.attributes['size']}",
                        "grounded_concepts": [obj.label],
                        "referenced_objects": [obj.object_id]
                    }

        return {
            "answer": "unknown",
            "confidence": 0.4,
            "reasoning": "Attribute information not available"
        }

    def _answer_location_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer questions about object locations"""
        question_lower = question.lower()

        if not hasattr(scene, 'objects'):
            return {
                "answer": "nowhere",
                "confidence": 0.3,
                "reasoning": "No objects in scene"
            }

        # Find object in question
        for obj in scene.objects:
            if obj.label.lower() in question_lower:
                # Describe location based on bounding box
                center_x, center_y = obj.bounding_box.center()

                # Determine location description
                h_loc = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
                v_loc = "top" if center_y < 0.33 else "bottom" if center_y > 0.66 else "middle"

                location_desc = f"in the {v_loc} {h_loc} of the scene"

                return {
                    "answer": location_desc,
                    "confidence": 0.75,
                    "reasoning": f"The {obj.label} is located {location_desc}",
                    "grounded_concepts": [obj.label],
                    "referenced_objects": [obj.object_id]
                }

        return {
            "answer": "not found",
            "confidence": 0.5,
            "reasoning": "Could not locate the queried object"
        }

    def _answer_relation_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer questions about spatial relationships"""
        question_lower = question.lower()

        if not hasattr(scene, 'spatial_relations') or not scene.spatial_relations:
            return {
                "answer": "no spatial relationships detected",
                "confidence": 0.4,
                "reasoning": "No spatial relations available"
            }

        # Find relevant spatial relation
        for relation in scene.spatial_relations:
            obj1 = next((o for o in scene.objects if o.object_id == relation["object1"]), None)
            obj2 = next((o for o in scene.objects if o.object_id == relation["object2"]), None)

            if obj1 and obj2:
                rel_str = str(relation["relation"].value).replace("_", " ")

                # Check if this relation matches the question
                if obj1.label.lower() in question_lower and rel_str in question_lower:
                    return {
                        "answer": obj2.label,
                        "confidence": relation.get("confidence", 0.7),
                        "reasoning": f"The {obj2.label} is {rel_str} the {obj1.label}",
                        "grounded_concepts": [obj1.label, obj2.label],
                        "referenced_objects": [obj1.object_id, obj2.object_id]
                    }

        return {
            "answer": "unknown",
            "confidence": 0.4,
            "reasoning": "Could not determine the spatial relationship"
        }

    def _answer_description_question(self, question: str, scene: Any) -> Dict[str, Any]:
        """Answer description questions"""
        # Generate caption for the scene
        caption = self.generate_caption(scene, detail_level="medium")

        return {
            "answer": caption.caption_text,
            "confidence": caption.confidence,
            "reasoning": "Generated description from scene understanding",
            "grounded_concepts": caption.grounded_concepts,
            "referenced_objects": list(caption.object_mentions.keys())
        }

    def generate_caption(
        self,
        scene_understanding: Any,
        detail_level: str = "medium"
    ) -> ImageCaption:
        """
        Generate natural language caption for a scene

        Args:
            scene_understanding: Scene understanding from visual processor
            detail_level: Level of detail (low, medium, high)

        Returns:
            ImageCaption with generated text
        """
        self.caption_counter += 1
        caption_id = f"caption_{self.caption_counter}"

        if not hasattr(scene_understanding, 'objects'):
            return ImageCaption(
                caption_id=caption_id,
                caption_text="An empty scene",
                confidence=0.5,
                detail_level=detail_level
            )

        # Build caption based on detail level
        if detail_level == "low":
            caption_text = self._generate_low_detail_caption(scene_understanding)
        elif detail_level == "high":
            caption_text = self._generate_high_detail_caption(scene_understanding)
        else:  # medium
            caption_text = self._generate_medium_detail_caption(scene_understanding)

        # Extract grounded concepts
        grounded_concepts = list(set(obj.label for obj in scene_understanding.objects))

        # Create object mentions
        object_mentions = {obj.object_id: obj.label for obj in scene_understanding.objects}

        caption = ImageCaption(
            caption_id=caption_id,
            caption_text=caption_text,
            scene_id=getattr(scene_understanding, 'scene_id', None),
            grounded_concepts=grounded_concepts,
            object_mentions=object_mentions,
            confidence=0.8,
            detail_level=detail_level
        )

        self.captions[caption_id] = caption
        return caption

    def _generate_low_detail_caption(self, scene: Any) -> str:
        """Generate simple caption"""
        if not scene.objects:
            return f"An empty {scene.scene_type.value} scene"

        # Just mention scene type and count
        return f"A {scene.scene_type.value} scene with {len(scene.objects)} objects"

    def _generate_medium_detail_caption(self, scene: Any) -> str:
        """Generate medium detail caption"""
        if hasattr(scene, 'scene_description') and scene.scene_description:
            return scene.scene_description

        # Build from objects
        if not scene.objects:
            return f"An empty {scene.scene_type.value} scene"

        obj_labels = [obj.label for obj in scene.objects[:5]]  # Top 5
        obj_str = ", ".join(obj_labels)

        return f"A {scene.scene_type.value} scene showing {obj_str}"

    def _generate_high_detail_caption(self, scene: Any) -> str:
        """Generate detailed caption with relationships"""
        if not scene.objects:
            return f"An empty {scene.scene_type.value} scene"

        # Start with scene type
        parts = [f"This is a {scene.scene_type.value} scene."]

        # Describe main objects
        if len(scene.objects) > 0:
            obj_list = ", ".join([obj.label for obj in scene.objects[:5]])
            parts.append(f"It contains {obj_list}.")

        # Add spatial relationships
        if hasattr(scene, 'spatial_relations') and scene.spatial_relations:
            rel = scene.spatial_relations[0]
            obj1 = next((o for o in scene.objects if o.object_id == rel["object1"]), None)
            obj2 = next((o for o in scene.objects if o.object_id == rel["object2"]), None)

            if obj1 and obj2:
                rel_str = str(rel["relation"].value).replace("_", " ")
                parts.append(f"The {obj2.label} is {rel_str} the {obj1.label}.")

        # Add attributes
        if scene.objects and scene.objects[0].attributes:
            obj = scene.objects[0]
            if "color" in obj.attributes:
                parts.append(f"The {obj.label} appears to be {obj.attributes['color']}.")

        return " ".join(parts)

    def resolve_reference(
        self,
        expression: str,
        scene_understanding: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ReferringExpression:
        """
        Resolve a referring expression to specific objects

        Args:
            expression: Referring expression (e.g., "the red car")
            scene_understanding: Scene understanding
            context: Optional context for resolution

        Returns:
            ReferringExpression with resolved reference
        """
        expr_lower = expression.lower()

        # Classify reference type
        if expr_lower.startswith("the "):
            ref_type = ReferenceType.DEFINITE
        elif expr_lower.startswith("a ") or expr_lower.startswith("an "):
            ref_type = ReferenceType.INDEFINITE
        elif any(expr_lower.startswith(w) for w in ["this ", "that ", "these ", "those "]):
            ref_type = ReferenceType.DEMONSTRATIVE
        elif expr_lower in ["it", "they", "them", "this", "that"]:
            ref_type = ReferenceType.PRONOUN
        else:
            ref_type = ReferenceType.DEFINITE

        # Extract attributes from expression
        attributes = self._extract_attributes_from_expression(expr_lower)

        # Find candidate objects
        candidates = []
        if hasattr(scene_understanding, 'objects'):
            for obj in scene_understanding.objects:
                if self._matches_expression(obj, expr_lower, attributes):
                    candidates.append(obj.object_id)

        # Select best candidate
        target = candidates[0] if candidates else None

        referring_expr = ReferringExpression(
            expression=expression,
            reference_type=ref_type,
            target_object_id=target,
            candidate_objects=candidates,
            attributes=attributes,
            confidence=1.0 / (len(candidates) + 1) if candidates else 0.1,
            resolved=target is not None
        )

        return referring_expr

    def _extract_attributes_from_expression(self, expression: str) -> Dict[str, str]:
        """Extract attributes from referring expression"""
        attributes = {}

        # Color patterns
        colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray"]
        for color in colors:
            if color in expression:
                attributes["color"] = color

        # Size patterns
        sizes = ["large", "big", "small", "tiny", "huge"]
        for size in sizes:
            if size in expression:
                attributes["size"] = size

        return attributes

    def _matches_expression(self, obj: Any, expression: str, attributes: Dict[str, str]) -> bool:
        """Check if object matches referring expression"""
        # Check label
        if obj.label.lower() not in expression:
            return False

        # Check attributes
        for attr_name, attr_value in attributes.items():
            if attr_name not in obj.attributes:
                return False
            if obj.attributes[attr_name].lower() != attr_value:
                return False

        return True

    def ground_concept(
        self,
        linguistic_form: str,
        perceptual_features: np.ndarray,
        object_ids: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> GroundedConcept:
        """
        Create or update a grounded concept

        Args:
            linguistic_form: Natural language expression
            perceptual_features: Visual/sensory features
            object_ids: Related object IDs
            attributes: Concept attributes

        Returns:
            GroundedConcept
        """
        # Check if concept already exists
        for concept in self.grounded_concepts.values():
            if concept.linguistic_form.lower() == linguistic_form.lower():
                # Update existing concept
                concept.usage_count += 1
                if object_ids:
                    concept.object_ids.extend(object_ids)
                return concept

        # Create new concept
        self.concept_counter += 1
        concept_id = f"grounded_{self.concept_counter}"

        concept = GroundedConcept(
            concept_id=concept_id,
            linguistic_form=linguistic_form,
            perceptual_features=perceptual_features,
            object_ids=object_ids or [],
            attributes=attributes or {},
            usage_count=1,
            confidence=0.7
        )

        self.grounded_concepts[concept_id] = concept
        return concept

    def start_grounded_dialog(self) -> GroundedDialog:
        """Start a new grounded dialog"""
        self.dialog_counter += 1
        dialog_id = f"dialog_{self.dialog_counter}"

        dialog = GroundedDialog(
            dialog_id=dialog_id,
            context={"started_at": time.time()}
        )

        self.dialogs[dialog_id] = dialog
        return dialog

    def process_grounded_utterance(
        self,
        dialog: GroundedDialog,
        utterance: str,
        scene: Any,
        speaker: str = "user"
    ) -> Dict[str, Any]:
        """
        Process an utterance in grounded dialog context

        Args:
            dialog: Active dialog
            utterance: Natural language utterance
            scene: Current scene understanding
            speaker: Who is speaking

        Returns:
            Processing result with resolved references
        """
        # Resolve any references in the utterance
        references = self._find_references_in_text(utterance)

        grounded_refs = []
        for ref in references:
            resolved = self.resolve_reference(ref, scene, dialog.context)
            if resolved.resolved:
                grounded_refs.append(resolved.target_object_id)

        # Add turn to dialog
        dialog.add_turn(speaker, utterance, grounded_refs)

        return {
            "resolved_references": grounded_refs,
            "unresolved": [r for r in references if r not in grounded_refs],
            "active_objects": list(dialog.active_objects)
        }

    def _find_references_in_text(self, text: str) -> List[str]:
        """Find referring expressions in text"""
        # Simple pattern matching
        # In production, use proper NLP parsing
        patterns = [
            r"the \w+",
            r"a \w+",
            r"an \w+",
            r"this \w+",
            r"that \w+",
        ]

        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            references.extend(matches)

        return references

    def get_stats(self) -> Dict[str, Any]:
        """Get grounding engine statistics"""
        return {
            "total_vqa": len(self.vqa_history),
            "total_captions": len(self.captions),
            "grounded_concepts": len(self.grounded_concepts),
            "active_dialogs": len(self.dialogs),
            "question_types": self._count_question_types()
        }

    def _count_question_types(self) -> Dict[str, int]:
        """Count questions by type"""
        counts = defaultdict(int)
        for vqa in self.vqa_history.values():
            counts[vqa.question_type.value] += 1
        return dict(counts)
