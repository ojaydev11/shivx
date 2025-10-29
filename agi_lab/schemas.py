"""
AGI Lab - Schema Definitions
Brain-inspired parallel AGI exploration framework
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class AGIApproachType(str, Enum):
    """Different AGI research approaches"""
    WORLD_MODEL = "world_model"  # Learn physics and causality
    META_LEARNING = "meta_learning"  # Learn to learn
    NEUROSYMBOLIC = "neurosymbolic"  # Neural + symbolic reasoning
    ACTIVE_INFERENCE = "active_inference"  # Free Energy Principle
    HTM = "hierarchical_temporal_memory"  # Numenta's approach
    NEURAL_TURING = "neural_turing_machine"  # Memory-augmented
    TRANSFORMER_WM = "transformer_world_model"  # JEPA-style
    EVOLUTIONARY = "evolutionary"  # Genetic programming
    OPEN_RL = "open_ended_rl"  # Curiosity-driven RL
    HYBRID = "hybrid"  # Multiple modules
    PREDICTIVE_CODING = "predictive_coding"  # Brain-inspired prediction
    CONSCIOUSNESS = "global_workspace"  # GWT/IIT inspired
    COMPOSITIONAL = "compositional_reasoning"  # Part-whole reasoning
    CAUSAL = "causal_inference"  # Pearl's do-calculus
    ANALOGICAL = "analogical_reasoning"  # Structure mapping
    EMBODIED = "embodied_cognition"  # Body-environment interaction


class NeuralPattern(BaseModel):
    """Records a computational pattern (like neural activation)"""
    pattern_id: str = Field(default_factory=lambda: str(uuid4()))
    approach_type: AGIApproachType
    pattern_type: str  # "activation", "decision", "prediction", "error"
    context: str  # What task/situation
    data: Dict[str, Any]  # The actual pattern data
    timestamp: datetime = Field(default_factory=datetime.now)
    success_score: float = Field(default=0.0, ge=0.0, le=1.0)
    generalization_score: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExperimentResult(BaseModel):
    """Result from running one AGI approach"""
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    approach_type: AGIApproachType
    config: Dict[str, Any]
    patterns: List[NeuralPattern] = Field(default_factory=list)

    # AGI-ness metrics
    task_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    generalization_score: float = Field(default=0.0, ge=0.0, le=1.0)
    transfer_learning_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_depth: int = Field(default=0, ge=0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)

    # Meta metrics
    training_time_sec: float
    memory_usage_mb: float
    convergence_step: Optional[int] = None

    timestamp: datetime = Field(default_factory=datetime.now)


class AGIFitnessMetrics(BaseModel):
    """Composite fitness for AGI-ness"""
    overall_score: float  # Weighted combination

    # Core AGI capabilities
    general_reasoning: float  # Solve diverse problems
    transfer_learning: float  # Apply knowledge across domains
    causal_understanding: float  # Know why, not just what
    abstraction: float  # Handle concepts at multiple levels
    creativity: float  # Generate novel solutions
    metacognition: float  # Reason about own reasoning

    # Emergent properties
    sample_efficiency: float  # Learn from few examples
    robustness: float  # Handle distribution shift
    interpretability: float  # Explain decisions

    weights: Dict[str, float] = Field(default_factory=lambda: {
        "general_reasoning": 0.20,
        "transfer_learning": 0.20,
        "causal_understanding": 0.15,
        "abstraction": 0.10,
        "creativity": 0.10,
        "metacognition": 0.10,
        "sample_efficiency": 0.05,
        "robustness": 0.05,
        "interpretability": 0.05,
    })

    def compute_overall(self) -> float:
        """Weighted sum of all metrics"""
        total = (
            self.general_reasoning * self.weights["general_reasoning"] +
            self.transfer_learning * self.weights["transfer_learning"] +
            self.causal_understanding * self.weights["causal_understanding"] +
            self.abstraction * self.weights["abstraction"] +
            self.creativity * self.weights["creativity"] +
            self.metacognition * self.weights["metacognition"] +
            self.sample_efficiency * self.weights["sample_efficiency"] +
            self.robustness * self.weights["robustness"] +
            self.interpretability * self.weights["interpretability"]
        )
        self.overall_score = total
        return total


class CrossPollinationResult(BaseModel):
    """Result of combining multiple successful approaches"""
    hybrid_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_approaches: List[AGIApproachType]
    combined_patterns: List[NeuralPattern]
    fitness: AGIFitnessMetrics
    generation: int  # Evolution generation number
    mutations: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class ExplorationSession(BaseModel):
    """Tracks one parallel exploration session"""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    num_parallel: int = Field(default=20, ge=1, le=100)
    max_generations: int = Field(default=10, ge=1)
    current_generation: int = Field(default=0)

    all_results: List[ExperimentResult] = Field(default_factory=list)
    best_approaches: List[str] = Field(default_factory=list)

    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    convergence_reached: bool = False
    best_fitness: float = Field(default=0.0)
