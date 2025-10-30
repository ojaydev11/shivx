"""
Visual Processing System

Comprehensive vision system for understanding images, detecting objects,
analyzing scenes, and performing visual reasoning.

Key capabilities:
- Image feature extraction
- Object detection and recognition
- Scene understanding and segmentation
- Visual reasoning and inference
- Spatial relationship understanding
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import time
import hashlib


class ObjectCategory(str, Enum):
    """Common object categories"""
    PERSON = "person"
    ANIMAL = "animal"
    VEHICLE = "vehicle"
    BUILDING = "building"
    FURNITURE = "furniture"
    FOOD = "food"
    TOOL = "tool"
    PLANT = "plant"
    ELECTRONIC = "electronic"
    CLOTHING = "clothing"
    UNKNOWN = "unknown"


class SceneType(str, Enum):
    """Scene types"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    NATURAL = "natural"
    URBAN = "urban"
    WORKSPACE = "workspace"
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


class SpatialRelation(str, Enum):
    """Spatial relationships between objects"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"
    NEAR = "near"
    FAR_FROM = "far_from"
    TOUCHING = "touching"


@dataclass
class BoundingBox:
    """Bounding box for object detection"""
    x: float  # top-left x
    y: float  # top-left y
    width: float
    height: float
    confidence: float = 1.0

    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        return (self.x + self.width / 2, self.y + self.height / 2)

    def area(self) -> float:
        """Calculate area of bounding box"""
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another box

        Returns overlap ratio between 0 and 1
        """
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class ObjectDetection:
    """Detected object in an image"""
    object_id: str
    category: ObjectCategory
    label: str
    confidence: float
    bounding_box: BoundingBox
    attributes: Dict[str, Any] = field(default_factory=dict)
    features: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)

    def describe(self) -> str:
        """Generate natural language description of object"""
        desc = f"{self.label}"
        if self.attributes:
            attrs = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
            desc += f" ({attrs})"
        return desc


@dataclass
class VisualConcept:
    """A learned visual concept"""
    concept_id: str
    name: str
    category: ObjectCategory
    prototype_features: np.ndarray  # Average/prototype features
    examples: List[str] = field(default_factory=list)  # Example object IDs
    attributes: Set[str] = field(default_factory=set)
    semantic_tags: Set[str] = field(default_factory=set)
    confidence: float = 1.0

    def similarity(self, features: np.ndarray) -> float:
        """Calculate similarity to given features (0-1)"""
        if self.prototype_features is None or features is None:
            return 0.0

        # Cosine similarity
        dot_product = np.dot(self.prototype_features, features)
        norm_product = np.linalg.norm(self.prototype_features) * np.linalg.norm(features)

        if norm_product == 0:
            return 0.0

        similarity = dot_product / norm_product
        return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to 0-1


@dataclass
class ImageFeatures:
    """Extracted features from an image"""
    image_id: str
    global_features: np.ndarray  # Whole image features
    local_features: List[np.ndarray] = field(default_factory=list)  # Region features
    color_histogram: Optional[np.ndarray] = None
    edge_map: Optional[np.ndarray] = None
    texture_features: Optional[np.ndarray] = None
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SceneUnderstanding:
    """Complete scene understanding"""
    scene_id: str
    scene_type: SceneType
    objects: List[ObjectDetection]
    spatial_relations: List[Dict[str, Any]]
    scene_description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def get_object_by_label(self, label: str) -> Optional[ObjectDetection]:
        """Find object by label"""
        for obj in self.objects:
            if obj.label.lower() == label.lower():
                return obj
        return None

    def get_objects_by_category(self, category: ObjectCategory) -> List[ObjectDetection]:
        """Get all objects of a specific category"""
        return [obj for obj in self.objects if obj.category == category]

    def find_relation(self, obj1_id: str, obj2_id: str) -> Optional[SpatialRelation]:
        """Find spatial relation between two objects"""
        for rel in self.spatial_relations:
            if rel["object1"] == obj1_id and rel["object2"] == obj2_id:
                return rel.get("relation")
        return None


class VisualProcessor:
    """
    Comprehensive visual processing system

    Features:
    - Image feature extraction
    - Object detection and recognition
    - Scene understanding
    - Visual concept learning
    - Spatial reasoning
    - Visual question answering support
    """

    def __init__(self):
        self.object_counter = 0
        self.scene_counter = 0
        self.image_counter = 0

        # Feature extraction settings (must be set before initializing concepts)
        self.feature_dim = 512  # Feature vector dimension
        self.detection_threshold = 0.5
        self.nms_threshold = 0.4  # Non-maximum suppression

        # Visual concept library
        self.concepts: Dict[str, VisualConcept] = {}
        self._initialize_base_concepts()

        # Detected objects cache
        self.detected_objects: Dict[str, ObjectDetection] = {}

        # Scene memory
        self.scenes: Dict[str, SceneUnderstanding] = {}

    def _initialize_base_concepts(self):
        """Initialize base visual concepts"""
        base_concepts = [
            ("person", ObjectCategory.PERSON, {"human", "individual"}),
            ("car", ObjectCategory.VEHICLE, {"automobile", "vehicle"}),
            ("dog", ObjectCategory.ANIMAL, {"canine", "pet"}),
            ("cat", ObjectCategory.ANIMAL, {"feline", "pet"}),
            ("chair", ObjectCategory.FURNITURE, {"seat", "furniture"}),
            ("table", ObjectCategory.FURNITURE, {"desk", "furniture"}),
            ("tree", ObjectCategory.PLANT, {"plant", "nature"}),
            ("building", ObjectCategory.BUILDING, {"structure", "architecture"}),
            ("phone", ObjectCategory.ELECTRONIC, {"device", "mobile"}),
            ("laptop", ObjectCategory.ELECTRONIC, {"computer", "device"}),
        ]

        for name, category, tags in base_concepts:
            concept_id = hashlib.md5(name.encode()).hexdigest()[:16]
            # Initialize with random prototype features
            prototype = np.random.randn(self.feature_dim).astype(np.float32)
            prototype = prototype / np.linalg.norm(prototype)  # Normalize

            self.concepts[concept_id] = VisualConcept(
                concept_id=concept_id,
                name=name,
                category=category,
                prototype_features=prototype,
                semantic_tags=tags
            )

    def extract_features(self, image_data: Any, image_id: Optional[str] = None) -> ImageFeatures:
        """
        Extract features from an image

        Args:
            image_data: Image data (numpy array, path, or PIL image)
            image_id: Optional image identifier

        Returns:
            ImageFeatures object with extracted features
        """
        self.image_counter += 1
        if image_id is None:
            image_id = f"img_{self.image_counter}"

        # In production, this would use a real CNN (ResNet, ViT, etc.)
        # For now, simulate feature extraction

        # Global features (simulated)
        global_features = self._simulate_global_features(image_data)

        # Local features from regions (simulated)
        local_features = self._simulate_local_features(image_data, num_regions=16)

        # Color histogram (simulated)
        color_histogram = self._simulate_color_histogram(image_data)

        # Texture features (simulated)
        texture_features = self._simulate_texture_features(image_data)

        features = ImageFeatures(
            image_id=image_id,
            global_features=global_features,
            local_features=local_features,
            color_histogram=color_histogram,
            texture_features=texture_features,
            spatial_layout={"grid_size": 8, "num_regions": 16},
            metadata={"feature_extractor": "simulated_cnn"}
        )

        return features

    def _simulate_global_features(self, image_data: Any) -> np.ndarray:
        """Simulate global image feature extraction"""
        # In production: features = model.encode(image_data)
        features = np.random.randn(self.feature_dim).astype(np.float32)
        features = features / np.linalg.norm(features)
        return features

    def _simulate_local_features(self, image_data: Any, num_regions: int = 16) -> List[np.ndarray]:
        """Simulate local region feature extraction"""
        local_features = []
        for _ in range(num_regions):
            features = np.random.randn(self.feature_dim).astype(np.float32)
            features = features / np.linalg.norm(features)
            local_features.append(features)
        return local_features

    def _simulate_color_histogram(self, image_data: Any, bins: int = 256) -> np.ndarray:
        """Simulate color histogram extraction"""
        # In production: compute actual histogram from image
        return np.random.rand(bins * 3).astype(np.float32)  # RGB histograms

    def _simulate_texture_features(self, image_data: Any) -> np.ndarray:
        """Simulate texture feature extraction"""
        # In production: compute Gabor filters, LBP, etc.
        return np.random.randn(128).astype(np.float32)

    def detect_objects(
        self,
        image_features: ImageFeatures,
        min_confidence: float = None
    ) -> List[ObjectDetection]:
        """
        Detect objects in an image

        Args:
            image_features: Extracted image features
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected objects
        """
        if min_confidence is None:
            min_confidence = self.detection_threshold

        detected = []

        # Simulate object detection using visual concepts
        # In production, this would use a real object detector (YOLO, Faster R-CNN, etc.)

        # Check each local feature against known concepts
        for i, local_feat in enumerate(image_features.local_features):
            # Find best matching concept
            best_match = None
            best_score = 0.0

            for concept in self.concepts.values():
                similarity = concept.similarity(local_feat)
                if similarity > best_score:
                    best_score = similarity
                    best_match = concept

            # If confidence above threshold, create detection
            if best_match and best_score >= min_confidence:
                self.object_counter += 1
                object_id = f"obj_{self.object_counter}"

                # Simulate bounding box based on region index
                grid_size = 4
                row = i // grid_size
                col = i % grid_size
                bbox = BoundingBox(
                    x=col * 0.25,
                    y=row * 0.25,
                    width=0.25,
                    height=0.25,
                    confidence=best_score
                )

                detection = ObjectDetection(
                    object_id=object_id,
                    category=best_match.category,
                    label=best_match.name,
                    confidence=best_score,
                    bounding_box=bbox,
                    features=local_feat,
                    attributes=self._infer_attributes(best_match, local_feat)
                )

                detected.append(detection)
                self.detected_objects[object_id] = detection

        # Apply non-maximum suppression to remove duplicates
        detected = self._non_maximum_suppression(detected)

        return detected

    def _infer_attributes(self, concept: VisualConcept, features: np.ndarray) -> Dict[str, Any]:
        """Infer object attributes from features"""
        attributes = {}

        # Simulate attribute inference
        # In production, use specialized classifiers

        # Size estimation
        size_score = np.mean(features[:10])
        if size_score > 0.3:
            attributes["size"] = "large"
        elif size_score < -0.3:
            attributes["size"] = "small"
        else:
            attributes["size"] = "medium"

        # Color estimation (simplified)
        color_score = np.mean(features[10:20])
        colors = ["red", "blue", "green", "yellow", "black", "white"]
        attributes["color"] = colors[int(abs(color_score * 10)) % len(colors)]

        return attributes

    def _non_maximum_suppression(
        self,
        detections: List[ObjectDetection],
        iou_threshold: float = None
    ) -> List[ObjectDetection]:
        """
        Apply non-maximum suppression to remove duplicate detections

        Keeps highest confidence detection when boxes overlap significantly
        """
        if iou_threshold is None:
            iou_threshold = self.nms_threshold

        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, det in enumerate(detections):
            if i in suppressed:
                continue

            keep.append(det)

            # Suppress overlapping lower-confidence detections
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue

                iou = det.bounding_box.iou(detections[j].bounding_box)
                if iou > iou_threshold and det.label == detections[j].label:
                    suppressed.add(j)

        return keep

    def understand_scene(
        self,
        image_features: ImageFeatures,
        detections: Optional[List[ObjectDetection]] = None
    ) -> SceneUnderstanding:
        """
        Perform comprehensive scene understanding

        Args:
            image_features: Extracted image features
            detections: Pre-computed object detections (optional)

        Returns:
            SceneUnderstanding with full scene analysis
        """
        # Detect objects if not provided
        if detections is None:
            detections = self.detect_objects(image_features)

        self.scene_counter += 1
        scene_id = f"scene_{self.scene_counter}"

        # Classify scene type
        scene_type = self._classify_scene_type(image_features, detections)

        # Analyze spatial relationships
        spatial_relations = self._analyze_spatial_relations(detections)

        # Generate scene description
        description = self._generate_scene_description(scene_type, detections, spatial_relations)

        # Extract scene attributes
        attributes = self._extract_scene_attributes(image_features, detections)

        scene = SceneUnderstanding(
            scene_id=scene_id,
            scene_type=scene_type,
            objects=detections,
            spatial_relations=spatial_relations,
            scene_description=description,
            attributes=attributes,
            confidence=0.85
        )

        self.scenes[scene_id] = scene
        return scene

    def _classify_scene_type(
        self,
        features: ImageFeatures,
        detections: List[ObjectDetection]
    ) -> SceneType:
        """Classify the type of scene"""
        # Simple heuristic-based classification
        # In production, use a scene classifier

        if not detections:
            return SceneType.UNKNOWN

        categories = [d.category for d in detections]

        # Check for indoor indicators
        furniture_count = categories.count(ObjectCategory.FURNITURE)
        electronic_count = categories.count(ObjectCategory.ELECTRONIC)

        if furniture_count >= 2 or electronic_count >= 2:
            return SceneType.INDOOR

        # Check for outdoor indicators
        plant_count = categories.count(ObjectCategory.PLANT)
        vehicle_count = categories.count(ObjectCategory.VEHICLE)
        building_count = categories.count(ObjectCategory.BUILDING)

        if plant_count >= 2:
            return SceneType.NATURAL

        if vehicle_count >= 1 or building_count >= 1:
            return SceneType.URBAN

        return SceneType.OUTDOOR

    def _analyze_spatial_relations(
        self,
        detections: List[ObjectDetection]
    ) -> List[Dict[str, Any]]:
        """Analyze spatial relationships between detected objects"""
        relations = []

        # Check all pairs of objects
        for i, obj1 in enumerate(detections):
            for obj2 in detections[i+1:]:
                relation = self._compute_spatial_relation(obj1, obj2)
                if relation:
                    relations.append({
                        "object1": obj1.object_id,
                        "object2": obj2.object_id,
                        "relation": relation,
                        "confidence": 0.8
                    })

        return relations

    def _compute_spatial_relation(
        self,
        obj1: ObjectDetection,
        obj2: ObjectDetection
    ) -> Optional[SpatialRelation]:
        """Compute spatial relation between two objects"""
        bbox1 = obj1.bounding_box
        bbox2 = obj2.bounding_box

        center1 = bbox1.center()
        center2 = bbox2.center()

        # Calculate relative positions
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Determine primary relation
        if distance < 0.1:
            return SpatialRelation.TOUCHING

        # Check vertical relations
        if abs(dy) > abs(dx):
            if dy < 0:
                return SpatialRelation.ABOVE
            else:
                return SpatialRelation.BELOW
        else:
            # Horizontal relations
            if dx < 0:
                return SpatialRelation.LEFT_OF
            else:
                return SpatialRelation.RIGHT_OF

    def _generate_scene_description(
        self,
        scene_type: SceneType,
        detections: List[ObjectDetection],
        relations: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language description of scene"""
        if not detections:
            return f"An empty {scene_type.value} scene"

        # Count objects by category
        category_counts = defaultdict(int)
        for det in detections:
            category_counts[det.label] += 1

        # Build description
        parts = [f"A {scene_type.value} scene containing"]

        obj_descriptions = []
        for label, count in category_counts.items():
            if count == 1:
                obj_descriptions.append(f"a {label}")
            else:
                obj_descriptions.append(f"{count} {label}s")

        if len(obj_descriptions) > 1:
            parts.append(", ".join(obj_descriptions[:-1]))
            parts.append("and")
            parts.append(obj_descriptions[-1])
        else:
            parts.append(obj_descriptions[0])

        return " ".join(parts)

    def _extract_scene_attributes(
        self,
        features: ImageFeatures,
        detections: List[ObjectDetection]
    ) -> Dict[str, Any]:
        """Extract high-level scene attributes"""
        attributes = {
            "num_objects": len(detections),
            "object_density": len(detections) / 16.0,  # Normalized by num regions
            "complexity": self._estimate_complexity(features, detections),
        }

        # Lighting estimation (from color histogram)
        if features.color_histogram is not None:
            brightness = np.mean(features.color_histogram)
            attributes["brightness"] = "bright" if brightness > 0.6 else "dim"

        return attributes

    def _estimate_complexity(
        self,
        features: ImageFeatures,
        detections: List[ObjectDetection]
    ) -> str:
        """Estimate scene complexity"""
        num_objects = len(detections)
        num_categories = len(set(d.category for d in detections))

        complexity_score = num_objects + num_categories * 2

        if complexity_score > 15:
            return "high"
        elif complexity_score > 8:
            return "medium"
        else:
            return "low"

    def learn_concept(
        self,
        examples: List[ObjectDetection],
        concept_name: str,
        category: ObjectCategory
    ) -> VisualConcept:
        """
        Learn a new visual concept from examples

        Args:
            examples: List of example detections
            concept_name: Name of the concept
            category: Object category

        Returns:
            Learned VisualConcept
        """
        if not examples:
            raise ValueError("Need at least one example to learn concept")

        concept_id = hashlib.md5(f"{concept_name}:{time.time()}".encode()).hexdigest()[:16]

        # Extract features from examples
        feature_list = [ex.features for ex in examples if ex.features is not None]

        if not feature_list:
            # Use random initialization if no features
            prototype = np.random.randn(self.feature_dim).astype(np.float32)
        else:
            # Average features as prototype
            prototype = np.mean(feature_list, axis=0)

        prototype = prototype / np.linalg.norm(prototype)

        # Extract common attributes
        all_attrs = set()
        for ex in examples:
            all_attrs.update(ex.attributes.keys())

        concept = VisualConcept(
            concept_id=concept_id,
            name=concept_name,
            category=category,
            prototype_features=prototype,
            examples=[ex.object_id for ex in examples],
            attributes=all_attrs,
            confidence=min(1.0, len(examples) / 5.0)  # More examples = higher confidence
        )

        self.concepts[concept_id] = concept
        return concept

    def visual_reasoning(
        self,
        scene: SceneUnderstanding,
        query: str
    ) -> Dict[str, Any]:
        """
        Perform visual reasoning on a scene

        Args:
            scene: Scene understanding
            query: Reasoning query (e.g., "What is to the left of the chair?")

        Returns:
            Reasoning result with answer and confidence
        """
        query_lower = query.lower()

        # Simple pattern matching for reasoning
        # In production, use more sophisticated NLP + visual reasoning

        # Count queries
        if "how many" in query_lower:
            return self._answer_count_query(scene, query_lower)

        # Spatial queries
        spatial_keywords = ["left", "right", "above", "below", "near"]
        if any(kw in query_lower for kw in spatial_keywords):
            return self._answer_spatial_query(scene, query_lower)

        # Attribute queries
        if "what color" in query_lower or "what size" in query_lower:
            return self._answer_attribute_query(scene, query_lower)

        # Existence queries
        if "is there" in query_lower or "are there" in query_lower:
            return self._answer_existence_query(scene, query_lower)

        return {
            "answer": "I don't understand the question",
            "confidence": 0.0,
            "reasoning": "Query type not recognized"
        }

    def _answer_count_query(self, scene: SceneUnderstanding, query: str) -> Dict[str, Any]:
        """Answer counting questions"""
        # Extract object type from query
        for concept in self.concepts.values():
            if concept.name in query:
                count = sum(1 for obj in scene.objects if obj.label == concept.name)
                return {
                    "answer": str(count),
                    "confidence": 0.9,
                    "reasoning": f"Counted {count} instances of '{concept.name}' in scene"
                }

        return {
            "answer": "0",
            "confidence": 0.5,
            "reasoning": "Object type not found in scene"
        }

    def _answer_spatial_query(self, scene: SceneUnderstanding, query: str) -> Dict[str, Any]:
        """Answer spatial relationship questions"""
        # This is a simplified implementation
        # In production, parse query more carefully

        for rel in scene.spatial_relations:
            obj1 = next((o for o in scene.objects if o.object_id == rel["object1"]), None)
            obj2 = next((o for o in scene.objects if o.object_id == rel["object2"]), None)

            if obj1 and obj2 and obj1.label in query and str(rel["relation"].value).replace("_", " ") in query:
                return {
                    "answer": obj2.label,
                    "confidence": 0.8,
                    "reasoning": f"{obj2.label} is {rel['relation'].value} {obj1.label}"
                }

        return {
            "answer": "Cannot determine",
            "confidence": 0.3,
            "reasoning": "Spatial relationship not found"
        }

    def _answer_attribute_query(self, scene: SceneUnderstanding, query: str) -> Dict[str, Any]:
        """Answer attribute questions"""
        # Find object in query
        for obj in scene.objects:
            if obj.label in query:
                if "color" in query and "color" in obj.attributes:
                    return {
                        "answer": obj.attributes["color"],
                        "confidence": 0.7,
                        "reasoning": f"The {obj.label} appears to be {obj.attributes['color']}"
                    }
                elif "size" in query and "size" in obj.attributes:
                    return {
                        "answer": obj.attributes["size"],
                        "confidence": 0.7,
                        "reasoning": f"The {obj.label} appears to be {obj.attributes['size']}"
                    }

        return {
            "answer": "Unknown",
            "confidence": 0.2,
            "reasoning": "Attribute not available"
        }

    def _answer_existence_query(self, scene: SceneUnderstanding, query: str) -> Dict[str, Any]:
        """Answer yes/no existence questions"""
        for concept in self.concepts.values():
            if concept.name in query:
                exists = any(obj.label == concept.name for obj in scene.objects)
                return {
                    "answer": "yes" if exists else "no",
                    "confidence": 0.85,
                    "reasoning": f"{'Found' if exists else 'Did not find'} {concept.name} in scene"
                }

        return {
            "answer": "unknown",
            "confidence": 0.1,
            "reasoning": "Cannot identify object in question"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get visual processor statistics"""
        return {
            "total_concepts": len(self.concepts),
            "detected_objects": len(self.detected_objects),
            "processed_scenes": len(self.scenes),
            "feature_dimension": self.feature_dim,
            "detection_threshold": self.detection_threshold,
        }
