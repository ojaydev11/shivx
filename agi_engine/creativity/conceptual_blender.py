"""
Conceptual Blending Engine

Implements conceptual blending (conceptual integration) for creativity.
Combines mental spaces to create emergent meaning and novel concepts.

Based on Fauconnier & Turner's Conceptual Blending Theory:
- Multiple input spaces
- Generic space (common structure)
- Blended space (emergent structure)
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import random
from collections import defaultdict


class BlendType(str, Enum):
    """Types of conceptual blends"""
    SIMPLEX = "simplex"  # Direct mapping
    MIRROR = "mirror"  # Similar structures
    SINGLE_SCOPE = "single_scope"  # One input dominates
    DOUBLE_SCOPE = "double_scope"  # Both inputs contribute structure
    FUSION = "fusion"  # Complete integration


class MappingStrength(str, Enum):
    """Strength of conceptual mapping"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    PERFECT = "perfect"


@dataclass
class Concept:
    """A concept with attributes and relationships"""
    name: str
    attributes: Set[str] = field(default_factory=set)
    relations: Dict[str, str] = field(default_factory=dict)  # relation_type -> target
    properties: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    abstractness: float = 0.5  # 0=concrete, 1=abstract

    def similarity_to(self, other: 'Concept') -> float:
        """Calculate similarity to another concept"""
        if not self.attributes or not other.attributes:
            return 0.0

        # Attribute overlap
        common_attrs = self.attributes.intersection(other.attributes)
        total_attrs = self.attributes.union(other.attributes)

        attr_similarity = len(common_attrs) / len(total_attrs) if total_attrs else 0.0

        # Relation overlap
        common_relations = set(self.relations.keys()).intersection(set(other.relations.keys()))
        total_relations = set(self.relations.keys()).union(set(other.relations.keys()))

        rel_similarity = len(common_relations) / len(total_relations) if total_relations else 0.0

        # Domain similarity
        domain_similarity = 1.0 if self.domain == other.domain else 0.2

        # Weighted combination
        return (attr_similarity * 0.5 + rel_similarity * 0.3 + domain_similarity * 0.2)


@dataclass
class MentalSpace:
    """A mental space containing concepts and relations"""
    space_id: str
    name: str
    concepts: Dict[str, Concept] = field(default_factory=dict)
    frame: str = "default"  # Organizing frame
    perspective: str = "neutral"
    temporal: str = "present"  # past, present, future
    modal: str = "actual"  # actual, hypothetical, counterfactual

    def add_concept(self, concept: Concept):
        """Add a concept to this mental space"""
        self.concepts[concept.name] = concept

    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name"""
        return self.concepts.get(name)

    def get_all_attributes(self) -> Set[str]:
        """Get all attributes from all concepts"""
        attrs = set()
        for concept in self.concepts.values():
            attrs.update(concept.attributes)
        return attrs


@dataclass
class ConceptualMapping:
    """Mapping between elements of mental spaces"""
    source_space: str
    target_space: str
    mappings: Dict[str, str] = field(default_factory=dict)  # source -> target
    strength: MappingStrength = MappingStrength.MODERATE
    reasoning: str = ""

    def get_mapped_element(self, source_element: str) -> Optional[str]:
        """Get the mapped target element"""
        return self.mappings.get(source_element)


@dataclass
class BlendedSpace:
    """The result of blending mental spaces"""
    blend_id: str
    input_spaces: List[str]
    generic_space: MentalSpace
    blended_concepts: Dict[str, Concept] = field(default_factory=dict)
    emergent_structure: List[str] = field(default_factory=list)
    blend_type: BlendType = BlendType.DOUBLE_SCOPE
    creativity_score: float = 0.5
    coherence_score: float = 0.5
    novelty_score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    description: str = ""

    def describe(self) -> str:
        """Generate natural language description of blend"""
        if self.description:
            return self.description

        concept_names = list(self.blended_concepts.keys())
        emergent_text = ", ".join(self.emergent_structure) if self.emergent_structure else "novel properties"

        return f"Blend of {', '.join(self.input_spaces)} creating {', '.join(concept_names[:3])} with {emergent_text}"


class ConceptualBlender:
    """
    Conceptual Blending Engine

    Implements Fauconnier & Turner's Conceptual Integration Networks:

    1. Input Spaces - Source mental spaces to blend
    2. Generic Space - Abstract common structure
    3. Blended Space - Integrated result with emergent structure

    Key operations:
    - Composition: Combine elements from inputs
    - Completion: Fill in implicit structure
    - Elaboration: Extend the blend through imagination

    Produces creative combinations with emergent meaning.
    """

    def __init__(self):
        self.mental_spaces: Dict[str, MentalSpace] = {}
        self.blended_spaces: Dict[str, BlendedSpace] = {}
        self.mappings: List[ConceptualMapping] = []
        self.space_counter = 0
        self.blend_counter = 0

        # Initialize with common mental spaces
        self._initialize_spaces()

    def _initialize_spaces(self):
        """Initialize common mental spaces"""
        # Technology space
        tech_space = self.create_mental_space("technology", frame="digital")
        tech_space.add_concept(Concept(
            name="computer",
            attributes={"digital", "computational", "fast", "logical"},
            relations={"processes": "data", "executes": "programs"},
            domain="technology"
        ))
        tech_space.add_concept(Concept(
            name="network",
            attributes={"connected", "distributed", "communicating"},
            relations={"links": "nodes", "transmits": "information"},
            domain="technology"
        ))

        # Biological space
        bio_space = self.create_mental_space("biology", frame="organic")
        bio_space.add_concept(Concept(
            name="brain",
            attributes={"neural", "adaptive", "learning", "organic"},
            relations={"processes": "signals", "controls": "body"},
            domain="biology"
        ))
        bio_space.add_concept(Concept(
            name="organism",
            attributes={"living", "growing", "adapting", "reproducing"},
            relations={"exists_in": "environment", "consumes": "resources"},
            domain="biology"
        ))

        # Social space
        social_space = self.create_mental_space("social", frame="interpersonal")
        social_space.add_concept(Concept(
            name="community",
            attributes={"collective", "cooperative", "sharing", "communicating"},
            relations={"contains": "individuals", "provides": "support"},
            domain="social"
        ))
        social_space.add_concept(Concept(
            name="market",
            attributes={"exchange", "value-based", "competitive", "dynamic"},
            relations={"facilitates": "trade", "determines": "price"},
            domain="social"
        ))

    def create_mental_space(
        self,
        name: str,
        frame: str = "default",
        perspective: str = "neutral"
    ) -> MentalSpace:
        """Create a new mental space"""
        self.space_counter += 1
        space_id = f"space_{self.space_counter}"

        space = MentalSpace(
            space_id=space_id,
            name=name,
            frame=frame,
            perspective=perspective
        )

        self.mental_spaces[space_id] = space
        return space

    def find_cross_space_mappings(
        self,
        space1: MentalSpace,
        space2: MentalSpace
    ) -> ConceptualMapping:
        """
        Find mappings between two mental spaces

        Maps based on:
        - Structural similarity
        - Functional correspondence
        - Attribute overlap
        """
        mapping = ConceptualMapping(
            source_space=space1.space_id,
            target_space=space2.space_id
        )

        # Find concept correspondences
        for name1, concept1 in space1.concepts.items():
            best_match = None
            best_similarity = 0.0

            for name2, concept2 in space2.concepts.items():
                similarity = concept1.similarity_to(concept2)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name2

            if best_match and best_similarity > 0.2:
                mapping.mappings[name1] = best_match

        # Determine mapping strength
        if len(mapping.mappings) == 0:
            mapping.strength = MappingStrength.WEAK
        elif len(mapping.mappings) <= 2:
            mapping.strength = MappingStrength.MODERATE
        else:
            mapping.strength = MappingStrength.STRONG

        mapping.reasoning = f"Mapped {len(mapping.mappings)} concepts based on structural similarity"

        self.mappings.append(mapping)
        return mapping

    def extract_generic_space(
        self,
        input_spaces: List[MentalSpace]
    ) -> MentalSpace:
        """
        Extract generic space - abstract common structure

        The generic space contains:
        - Common attributes across inputs
        - Shared relational structure
        - Abstract organizing frame
        """
        generic = self.create_mental_space(
            name="generic",
            frame="abstract"
        )

        # Find common attributes
        if input_spaces:
            all_attrs = [space.get_all_attributes() for space in input_spaces]
            common_attrs = set.intersection(*all_attrs) if all_attrs else set()

            # Create abstract concept with common structure
            if common_attrs:
                abstract_concept = Concept(
                    name="abstract_entity",
                    attributes=common_attrs,
                    abstractness=1.0
                )
                generic.add_concept(abstract_concept)

        return generic

    def blend(
        self,
        input_space_ids: List[str],
        blend_type: BlendType = BlendType.DOUBLE_SCOPE,
        selective: bool = False
    ) -> BlendedSpace:
        """
        Blend multiple mental spaces

        Args:
            input_space_ids: IDs of spaces to blend
            blend_type: Type of blend to create
            selective: If True, only blend compatible elements

        Returns:
            Blended space with emergent structure
        """
        input_spaces = [self.mental_spaces[sid] for sid in input_space_ids if sid in self.mental_spaces]

        if len(input_spaces) < 2:
            raise ValueError("Need at least 2 input spaces to blend")

        # Extract generic space
        generic_space = self.extract_generic_space(input_spaces)

        # Create blended space
        self.blend_counter += 1
        blend_id = hashlib.md5(
            f"blend_{self.blend_counter}:{time.time()}".encode()
        ).hexdigest()[:12]

        blended = BlendedSpace(
            blend_id=blend_id,
            input_spaces=input_space_ids,
            generic_space=generic_space,
            blend_type=blend_type
        )

        # Composition: Combine elements from inputs
        blended = self._compose(blended, input_spaces, selective)

        # Completion: Fill in implicit structure
        blended = self._complete(blended)

        # Elaboration: Extend through imagination
        blended = self._elaborate(blended)

        # Evaluate blend
        blended.creativity_score = self._evaluate_creativity(blended)
        blended.coherence_score = self._evaluate_coherence(blended)
        blended.novelty_score = self._evaluate_novelty(blended, input_spaces)

        # Generate description
        blended.description = blended.describe()

        self.blended_spaces[blend_id] = blended
        return blended

    def _compose(
        self,
        blended: BlendedSpace,
        input_spaces: List[MentalSpace],
        selective: bool
    ) -> BlendedSpace:
        """
        Composition: Combine elements from input spaces

        Strategies based on blend type:
        - SIMPLEX: Direct projection from one input
        - MIRROR: Merge similar structures
        - SINGLE_SCOPE: One input provides organizing frame
        - DOUBLE_SCOPE: Both inputs contribute structure
        - FUSION: Complete integration
        """
        if blended.blend_type == BlendType.SIMPLEX:
            # Simple projection from first input
            for concept in input_spaces[0].concepts.values():
                blended.blended_concepts[concept.name] = concept

        elif blended.blend_type == BlendType.MIRROR:
            # Merge similar structures
            for space in input_spaces:
                for concept in space.concepts.values():
                    if concept.name in blended.blended_concepts:
                        # Merge attributes
                        existing = blended.blended_concepts[concept.name]
                        existing.attributes.update(concept.attributes)
                    else:
                        blended.blended_concepts[concept.name] = concept

        elif blended.blend_type == BlendType.SINGLE_SCOPE:
            # First input provides structure, others provide content
            frame_space = input_spaces[0]

            for concept in frame_space.concepts.values():
                # Create new concept with structure from frame
                new_concept = Concept(
                    name=concept.name,
                    attributes=concept.attributes.copy(),
                    relations=concept.relations.copy(),
                    domain="blended"
                )

                # Add attributes from other spaces
                for space in input_spaces[1:]:
                    for other_concept in space.concepts.values():
                        new_concept.attributes.update(other_concept.attributes)

                blended.blended_concepts[concept.name] = new_concept

        elif blended.blend_type == BlendType.DOUBLE_SCOPE:
            # Both inputs contribute structure
            all_concepts = {}
            for space in input_spaces:
                all_concepts.update(space.concepts)

            # Create blended concepts
            for name, concept in all_concepts.items():
                if name not in blended.blended_concepts:
                    # Create new blended concept
                    new_concept = Concept(
                        name=name,
                        attributes=concept.attributes.copy(),
                        relations=concept.relations.copy(),
                        domain="blended"
                    )

                    # Add attributes from all matching concepts
                    for space in input_spaces:
                        if name in space.concepts:
                            new_concept.attributes.update(space.concepts[name].attributes)

                    blended.blended_concepts[name] = new_concept

        elif blended.blend_type == BlendType.FUSION:
            # Complete integration - create unified hybrid
            all_attrs = set()
            all_relations = {}

            for space in input_spaces:
                for concept in space.concepts.values():
                    all_attrs.update(concept.attributes)
                    all_relations.update(concept.relations)

            # Create fusion concept
            fusion_name = "_".join([space.name for space in input_spaces])
            fusion_concept = Concept(
                name=fusion_name,
                attributes=all_attrs,
                relations=all_relations,
                domain="blended"
            )

            blended.blended_concepts[fusion_name] = fusion_concept

        return blended

    def _complete(self, blended: BlendedSpace) -> BlendedSpace:
        """
        Completion: Fill in implicit structure

        Uses background knowledge and inference to complete the blend
        """
        # Infer missing relations
        for concept in blended.blended_concepts.values():
            # Add default relations based on attributes
            if "connected" in concept.attributes and "links" not in concept.relations:
                concept.relations["links"] = "other_entities"

            if "processes" in concept.relations and "produces" not in concept.relations:
                concept.relations["produces"] = "results"

            if "living" in concept.attributes and "requires" not in concept.relations:
                concept.relations["requires"] = "energy"

        return blended

    def _elaborate(self, blended: BlendedSpace) -> BlendedSpace:
        """
        Elaboration: Extend the blend through imagination

        Runs the blend dynamically to discover emergent structure
        """
        emergent = []

        # Discover emergent properties
        all_attributes = set()
        for concept in blended.blended_concepts.values():
            all_attributes.update(concept.attributes)

        # Look for interesting attribute combinations
        attr_list = list(all_attributes)
        for i in range(len(attr_list)):
            for j in range(i + 1, len(attr_list)):
                attr1, attr2 = attr_list[i], attr_list[j]

                # Check for emergent properties
                if self._creates_emergence(attr1, attr2):
                    emergent_property = f"{attr1}_{attr2}_synergy"
                    emergent.append(emergent_property)

        blended.emergent_structure = emergent
        return blended

    def _creates_emergence(self, attr1: str, attr2: str) -> bool:
        """Check if two attributes create emergent properties"""
        # Heuristics for emergence
        complementary_pairs = [
            ("digital", "organic"),
            ("logical", "adaptive"),
            ("collective", "individual"),
            ("static", "dynamic"),
            ("abstract", "concrete")
        ]

        for pair in complementary_pairs:
            if (attr1 in pair and attr2 in pair) or (attr2 in pair and attr1 in pair):
                return True

        return False

    def _evaluate_creativity(self, blended: BlendedSpace) -> float:
        """Evaluate creativity of blend"""
        score = 0.0

        # More concepts = more creative
        score += min(0.3, len(blended.blended_concepts) * 0.1)

        # Emergent structure = more creative
        score += min(0.4, len(blended.emergent_structure) * 0.1)

        # Cross-domain blending = more creative
        domains = set()
        for concept in blended.blended_concepts.values():
            domains.add(concept.domain)
        score += min(0.3, len(domains) * 0.15)

        return min(1.0, score)

    def _evaluate_coherence(self, blended: BlendedSpace) -> float:
        """Evaluate coherence of blend"""
        if not blended.blended_concepts:
            return 0.0

        # Check for conflicting attributes
        all_attrs = []
        for concept in blended.blended_concepts.values():
            all_attrs.extend(list(concept.attributes))

        # Check for contradictions
        conflicts = [
            ("digital", "analog"),
            ("living", "non-living"),
            ("static", "changing")
        ]

        conflict_count = 0
        for attr1, attr2 in conflicts:
            if attr1 in all_attrs and attr2 in all_attrs:
                conflict_count += 1

        # Coherence decreases with conflicts
        coherence = 1.0 - (conflict_count * 0.2)
        return max(0.0, coherence)

    def _evaluate_novelty(self, blended: BlendedSpace, input_spaces: List[MentalSpace]) -> float:
        """Evaluate novelty of blend compared to inputs"""
        # Count new attributes not in any input
        input_attrs = set()
        for space in input_spaces:
            input_attrs.update(space.get_all_attributes())

        blend_attrs = set()
        for concept in blended.blended_concepts.values():
            blend_attrs.update(concept.attributes)

        new_attrs = blend_attrs - input_attrs
        novelty = len(new_attrs) / len(blend_attrs) if blend_attrs else 0.0

        # Emergent structure adds novelty
        if blended.emergent_structure:
            novelty += 0.3

        return min(1.0, novelty)

    def blend_concepts_simple(
        self,
        concept1_name: str,
        concept1_attrs: List[str],
        concept2_name: str,
        concept2_attrs: List[str],
        blend_name: Optional[str] = None
    ) -> Concept:
        """
        Simple utility to blend two concepts directly

        Args:
            concept1_name: Name of first concept
            concept1_attrs: Attributes of first concept
            concept2_name: Name of second concept
            concept2_attrs: Attributes of second concept
            blend_name: Optional name for blended concept

        Returns:
            Blended concept
        """
        # Create concepts
        concept1 = Concept(name=concept1_name, attributes=set(concept1_attrs))
        concept2 = Concept(name=concept2_name, attributes=set(concept2_attrs))

        # Create mental spaces
        space1 = self.create_mental_space(concept1_name)
        space1.add_concept(concept1)

        space2 = self.create_mental_space(concept2_name)
        space2.add_concept(concept2)

        # Blend
        blended_space = self.blend([space1.space_id, space2.space_id], BlendType.FUSION)

        # Extract blended concept
        if not blended_space.blended_concepts:
            # Create manually
            return Concept(
                name=blend_name or f"{concept1_name}_{concept2_name}",
                attributes=set(concept1_attrs + concept2_attrs),
                domain="blended"
            )

        blended_concept = list(blended_space.blended_concepts.values())[0]

        if blend_name:
            blended_concept.name = blend_name

        return blended_concept

    def get_best_blends(self, n: int = 10, min_creativity: float = 0.5) -> List[BlendedSpace]:
        """Get the most creative blends"""
        blends = list(self.blended_spaces.values())
        blends = [b for b in blends if b.creativity_score >= min_creativity]
        blends.sort(key=lambda x: x.creativity_score, reverse=True)
        return blends[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get blender statistics"""
        return {
            "mental_spaces": len(self.mental_spaces),
            "blended_spaces": len(self.blended_spaces),
            "mappings": len(self.mappings),
            "avg_creativity": sum(b.creativity_score for b in self.blended_spaces.values()) / len(self.blended_spaces) if self.blended_spaces else 0,
            "avg_coherence": sum(b.coherence_score for b in self.blended_spaces.values()) / len(self.blended_spaces) if self.blended_spaces else 0,
            "avg_novelty": sum(b.novelty_score for b in self.blended_spaces.values()) / len(self.blended_spaces) if self.blended_spaces else 0,
        }
