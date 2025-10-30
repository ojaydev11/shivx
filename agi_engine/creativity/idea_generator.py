"""
Novel Idea Generator

Generates creative and innovative ideas using multiple techniques:
- Random combination
- Analogical reasoning
- Constraint-based generation
- SCAMPER technique
- Lateral thinking
- Bisociation (connecting unrelated domains)
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import time
import hashlib
import itertools
from collections import defaultdict


class IdeaQuality(str, Enum):
    """Quality/novelty level of an idea"""
    MUNDANE = "mundane"  # Common, obvious
    INCREMENTAL = "incremental"  # Small improvement
    NOVEL = "novel"  # New and interesting
    BREAKTHROUGH = "breakthrough"  # Revolutionary
    RADICAL = "radical"  # Paradigm-shifting


class GenerationTechnique(str, Enum):
    """Techniques for idea generation"""
    RANDOM_COMBINATION = "random_combination"
    ANALOGICAL = "analogical"
    CONSTRAINT_BASED = "constraint_based"
    SCAMPER = "scamper"
    LATERAL_THINKING = "lateral_thinking"
    BISOCIATION = "bisociation"
    MORPHOLOGICAL = "morphological"
    PROVOCATION = "provocation"


@dataclass
class Idea:
    """A generated creative idea"""
    idea_id: str
    description: str
    components: List[str]  # Building blocks
    technique: GenerationTechnique
    domain: str
    quality: IdeaQuality = IdeaQuality.NOVEL
    novelty_score: float = 0.5  # 0-1
    feasibility_score: float = 0.5  # 0-1
    impact_score: float = 0.5  # 0-1
    timestamp: float = field(default_factory=time.time)
    inspiration_sources: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    constraints_satisfied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate overall idea score"""
        return (self.novelty_score * 0.4 +
                self.feasibility_score * 0.3 +
                self.impact_score * 0.3)


@dataclass
class ConceptSpace:
    """A space of concepts for ideation"""
    domain: str
    concepts: Set[str]
    attributes: Dict[str, List[str]]  # Concept -> attributes
    relationships: Dict[Tuple[str, str], str]  # (concept1, concept2) -> relationship
    constraints: List[str] = field(default_factory=list)


class IdeaGenerator:
    """
    Novel Idea Generation Engine

    Uses multiple creativity techniques to generate innovative ideas:
    1. Random Combination - Merge unrelated concepts
    2. Analogical Reasoning - Apply patterns from other domains
    3. Constraint-Based - Generate within specific constraints
    4. SCAMPER - Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse
    5. Lateral Thinking - Break conventional patterns
    6. Bisociation - Connect distant conceptual frameworks
    7. Morphological Analysis - Systematic exploration of combinations
    8. Provocation - Use deliberate disruption
    """

    def __init__(self):
        # Knowledge base of concepts and domains
        self.concept_spaces: Dict[str, ConceptSpace] = {}
        self.idea_history: List[Idea] = []
        self.idea_counter = 0

        # Cross-domain mappings for analogical reasoning
        self.domain_analogies: Dict[str, List[str]] = defaultdict(list)

        # SCAMPER operators
        self.scamper_operators = {
            "substitute": ["replace", "swap", "exchange", "switch"],
            "combine": ["merge", "unite", "blend", "integrate"],
            "adapt": ["adjust", "modify", "change", "alter"],
            "modify": ["magnify", "minify", "transform", "reshape"],
            "put_to_other_use": ["repurpose", "redirect", "reapply"],
            "eliminate": ["remove", "simplify", "streamline", "reduce"],
            "reverse": ["invert", "flip", "opposite", "backwards"]
        }

        # Seed with common concept spaces
        self._initialize_concept_spaces()

    def _initialize_concept_spaces(self):
        """Initialize common concept spaces"""
        # Technology domain
        tech_concepts = {
            "artificial intelligence", "blockchain", "quantum computing",
            "internet of things", "virtual reality", "augmented reality",
            "machine learning", "neural networks", "cloud computing",
            "edge computing", "5G", "robotics", "nanotechnology",
            "biotechnology", "renewable energy", "autonomous vehicles"
        }

        tech_space = ConceptSpace(
            domain="technology",
            concepts=tech_concepts,
            attributes={
                "artificial intelligence": ["adaptive", "learning", "autonomous"],
                "blockchain": ["decentralized", "immutable", "transparent"],
                "quantum computing": ["superposition", "fast", "parallel"]
            },
            relationships={}
        )
        self.concept_spaces["technology"] = tech_space

        # Nature domain
        nature_concepts = {
            "ecosystem", "evolution", "adaptation", "symbiosis",
            "photosynthesis", "migration", "hibernation", "swarm behavior",
            "fractal patterns", "biomimicry", "natural selection",
            "mutualism", "predator-prey", "food chain"
        }

        nature_space = ConceptSpace(
            domain="nature",
            concepts=nature_concepts,
            attributes={
                "ecosystem": ["balanced", "interconnected", "resilient"],
                "evolution": ["gradual", "adaptive", "selective"],
                "swarm behavior": ["collective", "emergent", "coordinated"]
            },
            relationships={}
        )
        self.concept_spaces["nature"] = nature_space

        # Social domain
        social_concepts = {
            "collaboration", "community", "governance", "marketplace",
            "education", "healthcare", "entertainment", "communication",
            "democracy", "hierarchy", "network", "reputation"
        }

        social_space = ConceptSpace(
            domain="social",
            concepts=social_concepts,
            attributes={
                "collaboration": ["cooperative", "shared", "synergistic"],
                "marketplace": ["exchange", "value-based", "competitive"],
                "education": ["learning", "transformative", "developmental"]
            },
            relationships={}
        )
        self.concept_spaces["social"] = social_space

    def register_concept_space(self, space: ConceptSpace):
        """Register a new concept space"""
        self.concept_spaces[space.domain] = space

    def generate_ideas(
        self,
        prompt: str,
        domain: str = "general",
        num_ideas: int = 10,
        techniques: Optional[List[GenerationTechnique]] = None,
        constraints: Optional[List[str]] = None,
        min_novelty: float = 0.3
    ) -> List[Idea]:
        """
        Generate creative ideas based on prompt

        Args:
            prompt: What to generate ideas for
            domain: Target domain
            num_ideas: Number of ideas to generate
            techniques: Specific techniques to use (None = all)
            constraints: Constraints to satisfy
            min_novelty: Minimum novelty threshold

        Returns:
            List of generated ideas
        """
        ideas = []
        techniques_to_use = techniques or list(GenerationTechnique)

        # Generate ideas using different techniques
        for technique in techniques_to_use:
            if technique == GenerationTechnique.RANDOM_COMBINATION:
                ideas.extend(self._generate_random_combinations(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.ANALOGICAL:
                ideas.extend(self._generate_analogical(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.CONSTRAINT_BASED:
                ideas.extend(self._generate_constraint_based(prompt, domain, constraints or [], num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.SCAMPER:
                ideas.extend(self._generate_scamper(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.LATERAL_THINKING:
                ideas.extend(self._generate_lateral(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.BISOCIATION:
                ideas.extend(self._generate_bisociation(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.MORPHOLOGICAL:
                ideas.extend(self._generate_morphological(prompt, domain, num_ideas // len(techniques_to_use)))

            elif technique == GenerationTechnique.PROVOCATION:
                ideas.extend(self._generate_provocative(prompt, domain, num_ideas // len(techniques_to_use)))

        # Filter by novelty
        ideas = [idea for idea in ideas if idea.novelty_score >= min_novelty]

        # Sort by overall score
        ideas.sort(key=lambda x: x.overall_score(), reverse=True)

        # Store in history
        self.idea_history.extend(ideas[:num_ideas])

        return ideas[:num_ideas]

    def _generate_random_combinations(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas by randomly combining concepts"""
        ideas = []

        # Get concepts from multiple domains
        all_concepts = []
        for space in self.concept_spaces.values():
            all_concepts.extend(list(space.concepts))

        if len(all_concepts) < 2:
            return ideas

        for _ in range(count):
            # Pick 2-4 random concepts
            num_components = random.randint(2, 4)
            components = random.sample(all_concepts, min(num_components, len(all_concepts)))

            # Generate description
            description = f"Combine {' and '.join(components)} for {prompt}"

            idea = self._create_idea(
                description=description,
                components=components,
                technique=GenerationTechnique.RANDOM_COMBINATION,
                domain=domain
            )

            # Random combinations tend to be more novel but less feasible
            idea.novelty_score = random.uniform(0.6, 0.9)
            idea.feasibility_score = random.uniform(0.3, 0.6)
            idea.impact_score = random.uniform(0.4, 0.8)

            ideas.append(idea)

        return ideas

    def _generate_analogical(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas using analogical reasoning"""
        ideas = []

        # Find source domains for analogy
        source_domains = [d for d in self.concept_spaces.keys() if d != domain]

        if not source_domains:
            return ideas

        for _ in range(count):
            # Pick a source domain
            source_domain = random.choice(source_domains)
            source_space = self.concept_spaces[source_domain]

            # Pick concepts from source domain
            if not source_space.concepts:
                continue

            source_concepts = random.sample(
                list(source_space.concepts),
                min(2, len(source_space.concepts))
            )

            # Map to target domain using analogy
            description = f"Apply {source_domain} concept of {' and '.join(source_concepts)} to {prompt} in {domain}"

            idea = self._create_idea(
                description=description,
                components=source_concepts,
                technique=GenerationTechnique.ANALOGICAL,
                domain=domain
            )

            # Analogies are moderately novel and feasible
            idea.novelty_score = random.uniform(0.5, 0.8)
            idea.feasibility_score = random.uniform(0.5, 0.8)
            idea.impact_score = random.uniform(0.5, 0.9)
            idea.inspiration_sources = [source_domain]

            ideas.append(idea)

        return ideas

    def _generate_constraint_based(self, prompt: str, domain: str, constraints: List[str], count: int) -> List[Idea]:
        """Generate ideas within specific constraints"""
        ideas = []

        # Get relevant concepts
        relevant_concepts = []
        if domain in self.concept_spaces:
            relevant_concepts = list(self.concept_spaces[domain].concepts)

        if not relevant_concepts:
            # Use all concepts
            for space in self.concept_spaces.values():
                relevant_concepts.extend(list(space.concepts))

        if not relevant_concepts:
            return ideas

        for _ in range(count):
            # Pick concepts that could satisfy constraints
            components = random.sample(relevant_concepts, min(2, len(relevant_concepts)))

            # Build description incorporating constraints
            constraint_text = ", ".join(constraints) if constraints else "given constraints"
            description = f"Solution for {prompt} using {' and '.join(components)} while satisfying {constraint_text}"

            idea = self._create_idea(
                description=description,
                components=components,
                technique=GenerationTechnique.CONSTRAINT_BASED,
                domain=domain
            )

            idea.constraints_satisfied = constraints
            # Constraint-based ideas are more feasible but less novel
            idea.novelty_score = random.uniform(0.3, 0.6)
            idea.feasibility_score = random.uniform(0.6, 0.9)
            idea.impact_score = random.uniform(0.4, 0.7)

            ideas.append(idea)

        return ideas

    def _generate_scamper(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas using SCAMPER technique"""
        ideas = []

        # Get base concepts
        base_concepts = []
        if domain in self.concept_spaces:
            base_concepts = list(self.concept_spaces[domain].concepts)

        if not base_concepts:
            return ideas

        scamper_techniques = list(self.scamper_operators.keys())

        for _ in range(count):
            # Pick SCAMPER operator
            operator = random.choice(scamper_techniques)
            actions = self.scamper_operators[operator]
            action = random.choice(actions)

            # Pick concept to apply operator to
            concept = random.choice(base_concepts)

            # Generate description
            description = f"{action.capitalize()} {concept} to {prompt}"

            idea = self._create_idea(
                description=description,
                components=[concept, operator],
                technique=GenerationTechnique.SCAMPER,
                domain=domain
            )

            # SCAMPER produces practical, incremental ideas
            idea.novelty_score = random.uniform(0.4, 0.7)
            idea.feasibility_score = random.uniform(0.6, 0.9)
            idea.impact_score = random.uniform(0.4, 0.7)
            idea.metadata["scamper_operator"] = operator

            ideas.append(idea)

        return ideas

    def _generate_lateral(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas using lateral thinking"""
        ideas = []

        # Lateral thinking involves breaking assumptions
        provocations = [
            "What if we did the opposite?",
            "What if there were no constraints?",
            "What if we started from scratch?",
            "What if we combined it with something completely different?",
            "What if we removed the main component?"
        ]

        # Get concepts
        concepts = []
        for space in self.concept_spaces.values():
            concepts.extend(list(space.concepts))

        if not concepts:
            return ideas

        for _ in range(count):
            provocation = random.choice(provocations)
            concept = random.choice(concepts)

            description = f"{provocation} Apply to {prompt} using {concept}"

            idea = self._create_idea(
                description=description,
                components=[concept, "lateral_thinking"],
                technique=GenerationTechnique.LATERAL_THINKING,
                domain=domain
            )

            # Lateral thinking produces highly novel but risky ideas
            idea.novelty_score = random.uniform(0.7, 1.0)
            idea.feasibility_score = random.uniform(0.2, 0.5)
            idea.impact_score = random.uniform(0.5, 1.0)
            idea.metadata["provocation"] = provocation

            ideas.append(idea)

        return ideas

    def _generate_bisociation(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas by connecting distant conceptual frameworks"""
        ideas = []

        # Get two distant domains
        domains = list(self.concept_spaces.keys())
        if len(domains) < 2:
            return ideas

        for _ in range(count):
            # Pick two random domains
            domain1, domain2 = random.sample(domains, 2)
            space1 = self.concept_spaces[domain1]
            space2 = self.concept_spaces[domain2]

            if not space1.concepts or not space2.concepts:
                continue

            # Pick concepts from each
            concept1 = random.choice(list(space1.concepts))
            concept2 = random.choice(list(space2.concepts))

            description = f"Connect {domain1} concept '{concept1}' with {domain2} concept '{concept2}' for {prompt}"

            idea = self._create_idea(
                description=description,
                components=[concept1, concept2],
                technique=GenerationTechnique.BISOCIATION,
                domain=domain
            )

            # Bisociation produces breakthrough ideas
            idea.novelty_score = random.uniform(0.7, 0.95)
            idea.feasibility_score = random.uniform(0.3, 0.6)
            idea.impact_score = random.uniform(0.6, 0.95)
            idea.inspiration_sources = [domain1, domain2]

            ideas.append(idea)

        return ideas

    def _generate_morphological(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas using morphological analysis"""
        ideas = []

        # Define dimensions and options
        dimensions = {
            "input": ["data", "user action", "sensor", "external event"],
            "processing": ["algorithm", "neural network", "rule-based", "hybrid"],
            "output": ["visualization", "action", "prediction", "recommendation"],
            "interface": ["voice", "touch", "gesture", "neural"]
        }

        for _ in range(count):
            # Pick one option from each dimension
            combination = {}
            for dim, options in dimensions.items():
                combination[dim] = random.choice(options)

            description = f"System for {prompt}: input via {combination['input']}, process with {combination['processing']}, output as {combination['output']}, interface through {combination['interface']}"

            idea = self._create_idea(
                description=description,
                components=list(combination.values()),
                technique=GenerationTechnique.MORPHOLOGICAL,
                domain=domain
            )

            # Morphological analysis produces systematic, feasible ideas
            idea.novelty_score = random.uniform(0.4, 0.7)
            idea.feasibility_score = random.uniform(0.6, 0.9)
            idea.impact_score = random.uniform(0.5, 0.8)
            idea.metadata["morphological_dims"] = combination

            ideas.append(idea)

        return ideas

    def _generate_provocative(self, prompt: str, domain: str, count: int) -> List[Idea]:
        """Generate ideas using provocative statements"""
        ideas = []

        # Provocative statements challenge assumptions
        provocations = [
            "must be impossible",
            "should cost nothing",
            "needs to happen instantly",
            "everyone must love it",
            "requires no resources",
            "works without power",
            "creates infinite value"
        ]

        for _ in range(count):
            provocation = random.choice(provocations)

            description = f"Design {prompt} that {provocation}"

            idea = self._create_idea(
                description=description,
                components=["provocation", provocation],
                technique=GenerationTechnique.PROVOCATION,
                domain=domain
            )

            # Provocations are highly novel but very challenging
            idea.novelty_score = random.uniform(0.8, 1.0)
            idea.feasibility_score = random.uniform(0.1, 0.3)
            idea.impact_score = random.uniform(0.7, 1.0)
            idea.metadata["provocation"] = provocation

            ideas.append(idea)

        return ideas

    def _create_idea(
        self,
        description: str,
        components: List[str],
        technique: GenerationTechnique,
        domain: str
    ) -> Idea:
        """Create an idea object"""
        self.idea_counter += 1

        # Generate unique ID
        idea_id = hashlib.md5(
            f"{description}:{time.time()}:{self.idea_counter}".encode()
        ).hexdigest()[:12]

        return Idea(
            idea_id=idea_id,
            description=description,
            components=components,
            technique=technique,
            domain=domain
        )

    def evaluate_idea(self, idea: Idea, criteria: Optional[Dict[str, float]] = None) -> float:
        """
        Evaluate an idea's quality

        Args:
            idea: The idea to evaluate
            criteria: Optional custom criteria weights

        Returns:
            Overall quality score (0-1)
        """
        if criteria is None:
            criteria = {
                "novelty": 0.35,
                "feasibility": 0.35,
                "impact": 0.30
            }

        score = (
            idea.novelty_score * criteria.get("novelty", 0.35) +
            idea.feasibility_score * criteria.get("feasibility", 0.35) +
            idea.impact_score * criteria.get("impact", 0.30)
        )

        return min(1.0, max(0.0, score))

    def refine_idea(self, idea: Idea, feedback: str) -> Idea:
        """
        Refine an existing idea based on feedback

        Args:
            idea: Original idea
            feedback: Feedback to incorporate

        Returns:
            Refined idea
        """
        # Create refined version
        refined_description = f"{idea.description} [Refined: {feedback}]"

        refined = self._create_idea(
            description=refined_description,
            components=idea.components,
            technique=idea.technique,
            domain=idea.domain
        )

        # Adjust scores based on refinement
        refined.novelty_score = idea.novelty_score * 0.9  # Slightly less novel
        refined.feasibility_score = min(1.0, idea.feasibility_score * 1.2)  # More feasible
        refined.impact_score = idea.impact_score

        refined.inspiration_sources = idea.inspiration_sources + [idea.idea_id]

        return refined

    def get_best_ideas(self, n: int = 10, min_quality: float = 0.5) -> List[Idea]:
        """Get the best ideas generated so far"""
        filtered = [idea for idea in self.idea_history if self.evaluate_idea(idea) >= min_quality]
        filtered.sort(key=lambda x: self.evaluate_idea(x), reverse=True)
        return filtered[:n]

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        technique_counts = defaultdict(int)
        quality_counts = defaultdict(int)

        for idea in self.idea_history:
            technique_counts[idea.technique.value] += 1
            quality_counts[idea.quality.value] += 1

        return {
            "total_ideas_generated": len(self.idea_history),
            "unique_techniques_used": len(technique_counts),
            "by_technique": dict(technique_counts),
            "by_quality": dict(quality_counts),
            "avg_novelty": sum(i.novelty_score for i in self.idea_history) / len(self.idea_history) if self.idea_history else 0,
            "avg_feasibility": sum(i.feasibility_score for i in self.idea_history) / len(self.idea_history) if self.idea_history else 0,
            "avg_impact": sum(i.impact_score for i in self.idea_history) / len(self.idea_history) if self.idea_history else 0,
            "concept_spaces": len(self.concept_spaces),
        }
