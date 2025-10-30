"""
Theory of Mind - Model Other Agents' Mental States

This module enables AGI to build and maintain models of other agents' beliefs,
intentions, goals, and mental states. This is crucial for predicting behavior,
understanding motivations, and effective social interaction.

Key capabilities:
- Mental state tracking and inference
- Belief attribution and updating
- Intention recognition
- Perspective taking
- False belief reasoning
- Emotional state modeling
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class BeliefType(str, Enum):
    """Types of beliefs an agent can hold"""
    FACTUAL = "factual"  # Beliefs about facts
    NORMATIVE = "normative"  # Beliefs about what should be
    CAUSAL = "causal"  # Beliefs about cause-effect
    INTENTIONAL = "intentional"  # Beliefs about others' intentions
    EPISTEMIC = "epistemic"  # Beliefs about knowledge


class IntentionType(str, Enum):
    """Types of intentions"""
    GOAL = "goal"  # Achieve a goal
    COMMUNICATE = "communicate"  # Share information
    COOPERATE = "cooperate"  # Work together
    COMPETE = "compete"  # Compete for resources
    OBSERVE = "observe"  # Gather information
    AVOID = "avoid"  # Avoid something


class EmotionalState(str, Enum):
    """Basic emotional states"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    CURIOUS = "curious"
    FRUSTRATED = "frustrated"


@dataclass
class Belief:
    """A belief held by an agent"""
    content: str  # What is believed
    belief_type: BeliefType
    confidence: float = 0.8  # 0-1, how strongly believed
    source: Optional[str] = None  # Where belief came from
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

    def strength(self) -> float:
        """Calculate belief strength based on evidence and contradictions"""
        evidence_score = min(1.0, len(self.evidence) * 0.1)
        contradiction_penalty = len(self.contradictions) * 0.15
        return max(0.0, min(1.0, self.confidence + evidence_score - contradiction_penalty))


@dataclass
class Intention:
    """An intention/goal of an agent"""
    description: str
    intention_type: IntentionType
    target: Optional[str] = None  # What/who the intention is about
    priority: float = 0.5  # 0-1
    confidence: float = 0.7  # How confident we are about this inference
    indicators: List[str] = field(default_factory=list)  # Observed behaviors
    timestamp: float = field(default_factory=time.time)


@dataclass
class MentalState:
    """Complete mental state of an agent"""
    agent_id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    intentions: Dict[str, Intention] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    emotional_state: EmotionalState = EmotionalState.NEUTRAL
    knowledge: Set[str] = field(default_factory=set)  # What agent knows
    attention: Optional[str] = None  # Current focus
    last_updated: float = field(default_factory=time.time)

    def add_belief(self, belief: Belief) -> str:
        """Add or update a belief"""
        belief_id = f"belief_{len(self.beliefs)}_{int(time.time())}"
        self.beliefs[belief_id] = belief
        self.last_updated = time.time()
        return belief_id

    def add_intention(self, intention: Intention) -> str:
        """Add or update an intention"""
        intention_id = f"intention_{len(self.intentions)}_{int(time.time())}"
        self.intentions[intention_id] = intention
        self.last_updated = time.time()
        return intention_id

    def get_strongest_intention(self) -> Optional[Intention]:
        """Get the most likely current intention"""
        if not self.intentions:
            return None
        return max(self.intentions.values(), key=lambda i: i.priority * i.confidence)


@dataclass
class AgentModel:
    """Model of another agent"""
    agent_id: str
    name: str
    mental_state: MentalState
    observed_behaviors: List[Dict[str, Any]] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)  # e.g., "cooperative": 0.8
    capabilities: Set[str] = field(default_factory=set)
    last_interaction: Optional[float] = None
    trust_level: float = 0.5  # 0-1

    def record_behavior(self, behavior: Dict[str, Any]):
        """Record an observed behavior"""
        behavior["timestamp"] = time.time()
        self.observed_behaviors.append(behavior)
        # Keep only recent behaviors
        if len(self.observed_behaviors) > 100:
            self.observed_behaviors = self.observed_behaviors[-100:]

    def record_interaction(self, interaction: Dict[str, Any]):
        """Record an interaction"""
        interaction["timestamp"] = time.time()
        self.interaction_history.append(interaction)
        self.last_interaction = time.time()
        # Keep only recent interactions
        if len(self.interaction_history) > 50:
            self.interaction_history = self.interaction_history[-50:]


class TheoryOfMind:
    """
    Theory of Mind system for modeling other agents

    Features:
    - Build and maintain mental models of other agents
    - Infer beliefs, intentions, and goals
    - Predict behavior based on mental states
    - Track perspective differences
    - Detect false beliefs
    - Update models based on observations
    """

    def __init__(self):
        self.agent_models: Dict[str, AgentModel] = {}
        self.self_model: Optional[MentalState] = None
        self.perspective_differences: Dict[Tuple[str, str], Set[str]] = {}

        # Initialize self model
        self._init_self_model()

    def _init_self_model(self):
        """Initialize model of self"""
        self.self_model = MentalState(
            agent_id="self",
            emotional_state=EmotionalState.NEUTRAL
        )

    def register_agent(
        self,
        agent_id: str,
        name: str,
        initial_beliefs: Optional[List[Belief]] = None,
        capabilities: Optional[Set[str]] = None
    ) -> AgentModel:
        """
        Register a new agent and create initial mental model

        Args:
            agent_id: Unique identifier for agent
            name: Human-readable name
            initial_beliefs: Starting beliefs about agent
            capabilities: Known capabilities of agent

        Returns:
            AgentModel for the new agent
        """
        mental_state = MentalState(agent_id=agent_id)

        # Add initial beliefs
        if initial_beliefs:
            for belief in initial_beliefs:
                mental_state.add_belief(belief)

        agent_model = AgentModel(
            agent_id=agent_id,
            name=name,
            mental_state=mental_state,
            capabilities=capabilities or set()
        )

        self.agent_models[agent_id] = agent_model
        return agent_model

    def observe_behavior(
        self,
        agent_id: str,
        behavior: str,
        context: Dict[str, Any]
    ) -> Optional[Intention]:
        """
        Observe agent behavior and infer intentions

        Args:
            agent_id: Agent being observed
            behavior: Description of behavior
            context: Contextual information

        Returns:
            Inferred intention, if any
        """
        if agent_id not in self.agent_models:
            # Unknown agent - create model
            self.register_agent(agent_id, f"Agent_{agent_id}")

        agent_model = self.agent_models[agent_id]

        # Record behavior
        agent_model.record_behavior({
            "behavior": behavior,
            "context": context
        })

        # Infer intention from behavior
        intention = self._infer_intention(behavior, context)

        if intention:
            agent_model.mental_state.add_intention(intention)

        # Update beliefs based on behavior
        self._update_beliefs_from_behavior(agent_model, behavior, context)

        return intention

    def _infer_intention(
        self,
        behavior: str,
        context: Dict[str, Any]
    ) -> Optional[Intention]:
        """
        Infer intention from observed behavior

        Uses heuristics and pattern matching to guess at intentions
        """
        behavior_lower = behavior.lower()

        # Goal-oriented behavior
        if any(word in behavior_lower for word in ["move toward", "reach for", "approach"]):
            target = context.get("target", "unknown")
            return Intention(
                description=f"Achieve goal related to {target}",
                intention_type=IntentionType.GOAL,
                target=target,
                confidence=0.7,
                indicators=[behavior]
            )

        # Communicative behavior
        elif any(word in behavior_lower for word in ["say", "signal", "gesture", "communicate"]):
            return Intention(
                description="Communicate information",
                intention_type=IntentionType.COMMUNICATE,
                confidence=0.8,
                indicators=[behavior]
            )

        # Cooperative behavior
        elif any(word in behavior_lower for word in ["help", "assist", "cooperate", "share"]):
            return Intention(
                description="Cooperate with others",
                intention_type=IntentionType.COOPERATE,
                confidence=0.75,
                indicators=[behavior]
            )

        # Competitive behavior
        elif any(word in behavior_lower for word in ["compete", "race", "oppose", "block"]):
            return Intention(
                description="Compete for resources",
                intention_type=IntentionType.COMPETE,
                confidence=0.7,
                indicators=[behavior]
            )

        # Observational behavior
        elif any(word in behavior_lower for word in ["look", "watch", "observe", "scan"]):
            return Intention(
                description="Gather information",
                intention_type=IntentionType.OBSERVE,
                confidence=0.6,
                indicators=[behavior]
            )

        # Avoidance behavior
        elif any(word in behavior_lower for word in ["avoid", "retreat", "flee", "escape"]):
            threat = context.get("threat", "unknown")
            return Intention(
                description=f"Avoid {threat}",
                intention_type=IntentionType.AVOID,
                target=threat,
                confidence=0.75,
                indicators=[behavior]
            )

        return None

    def _update_beliefs_from_behavior(
        self,
        agent_model: AgentModel,
        behavior: str,
        context: Dict[str, Any]
    ):
        """Update agent's beliefs based on observed behavior"""
        # If agent moved toward X, they likely believe X is desirable/necessary
        if "move toward" in behavior.lower():
            target = context.get("target")
            if target:
                belief = Belief(
                    content=f"{target} is desirable or necessary",
                    belief_type=BeliefType.NORMATIVE,
                    confidence=0.6,
                    source="observed_behavior",
                    evidence=[behavior]
                )
                agent_model.mental_state.add_belief(belief)

        # If agent avoided X, they likely believe X is dangerous/undesirable
        elif "avoid" in behavior.lower():
            threat = context.get("threat")
            if threat:
                belief = Belief(
                    content=f"{threat} is dangerous or undesirable",
                    belief_type=BeliefType.NORMATIVE,
                    confidence=0.7,
                    source="observed_behavior",
                    evidence=[behavior]
                )
                agent_model.mental_state.add_belief(belief)

    def predict_behavior(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """
        Predict likely behaviors based on mental model

        Args:
            agent_id: Agent to predict
            context: Current context

        Returns:
            List of (behavior, probability) tuples
        """
        if agent_id not in self.agent_models:
            return []

        agent_model = self.agent_models[agent_id]
        predictions = []

        # Get current intention
        intention = agent_model.mental_state.get_strongest_intention()

        if intention:
            # Predict behavior based on intention type
            if intention.intention_type == IntentionType.GOAL:
                if intention.target:
                    predictions.append((f"Move toward {intention.target}", 0.8))
                    predictions.append((f"Acquire {intention.target}", 0.6))

            elif intention.intention_type == IntentionType.COMMUNICATE:
                predictions.append(("Initiate communication", 0.8))
                predictions.append(("Signal or gesture", 0.6))

            elif intention.intention_type == IntentionType.COOPERATE:
                predictions.append(("Offer assistance", 0.7))
                predictions.append(("Share resources", 0.5))

            elif intention.intention_type == IntentionType.COMPETE:
                predictions.append(("Compete for resources", 0.7))
                predictions.append(("Block others", 0.4))

            elif intention.intention_type == IntentionType.OBSERVE:
                predictions.append(("Continue observing", 0.8))
                predictions.append(("Gather more data", 0.6))

            elif intention.intention_type == IntentionType.AVOID:
                if intention.target:
                    predictions.append((f"Retreat from {intention.target}", 0.9))
                    predictions.append((f"Find alternative route", 0.6))

        # Factor in personality traits
        if "cooperative" in agent_model.personality_traits:
            coop_score = agent_model.personality_traits["cooperative"]
            predictions.append(("Cooperate with others", coop_score * 0.5))

        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:5]  # Top 5 predictions

    def take_perspective(
        self,
        agent_id: str,
        situation: str
    ) -> Dict[str, Any]:
        """
        Take another agent's perspective on a situation

        Returns what the agent likely perceives, believes, and feels
        """
        if agent_id not in self.agent_models:
            return {"error": "Unknown agent"}

        agent_model = self.agent_models[agent_id]
        mental_state = agent_model.mental_state

        # What does agent know/perceive?
        perceived_info = self._infer_perception(agent_model, situation)

        # What are agent's relevant beliefs?
        relevant_beliefs = self._get_relevant_beliefs(mental_state, situation)

        # What is agent's emotional response?
        emotional_response = self._infer_emotion(agent_model, situation)

        # What might agent intend to do?
        likely_intentions = self._infer_likely_intentions(agent_model, situation)

        return {
            "agent_id": agent_id,
            "perception": perceived_info,
            "beliefs": relevant_beliefs,
            "emotion": emotional_response,
            "intentions": likely_intentions,
            "knowledge_gaps": self._identify_knowledge_gaps(agent_model, situation)
        }

    def _infer_perception(
        self,
        agent_model: AgentModel,
        situation: str
    ) -> List[str]:
        """Infer what agent can perceive in situation"""
        # Simple heuristic: agent perceives based on capabilities and attention
        perceptions = []

        # Add what agent is focusing on
        if agent_model.mental_state.attention:
            perceptions.append(f"Focused on: {agent_model.mental_state.attention}")

        # Add general perception based on capabilities
        if "vision" in agent_model.capabilities:
            perceptions.append(f"Visual information about: {situation}")

        if "hearing" in agent_model.capabilities:
            perceptions.append(f"Auditory information about: {situation}")

        return perceptions

    def _get_relevant_beliefs(
        self,
        mental_state: MentalState,
        situation: str
    ) -> List[str]:
        """Get beliefs relevant to situation"""
        relevant = []

        for belief in mental_state.beliefs.values():
            # Simple keyword matching
            if any(word in belief.content.lower() for word in situation.lower().split()):
                if belief.strength() > 0.3:  # Only strong enough beliefs
                    relevant.append(belief.content)

        return relevant

    def _infer_emotion(
        self,
        agent_model: AgentModel,
        situation: str
    ) -> EmotionalState:
        """Infer emotional response to situation"""
        # Current emotional state is baseline
        current_emotion = agent_model.mental_state.emotional_state

        # Adjust based on situation and personality
        situation_lower = situation.lower()

        if any(word in situation_lower for word in ["danger", "threat", "risk"]):
            return EmotionalState.FEARFUL

        if any(word in situation_lower for word in ["success", "achievement", "reward"]):
            return EmotionalState.HAPPY

        if any(word in situation_lower for word in ["failure", "loss", "defeat"]):
            return EmotionalState.SAD

        if any(word in situation_lower for word in ["obstacle", "interference", "blocked"]):
            return EmotionalState.FRUSTRATED

        if any(word in situation_lower for word in ["unexpected", "novel", "new"]):
            return EmotionalState.SURPRISED

        return current_emotion

    def _infer_likely_intentions(
        self,
        agent_model: AgentModel,
        situation: str
    ) -> List[str]:
        """Infer what agent might intend to do"""
        intentions = []

        # Get current strongest intention
        current_intention = agent_model.mental_state.get_strongest_intention()
        if current_intention:
            intentions.append(f"Continue: {current_intention.description}")

        # Infer new intentions based on situation
        situation_lower = situation.lower()

        if "opportunity" in situation_lower:
            intentions.append("Seize opportunity")

        if "problem" in situation_lower:
            intentions.append("Solve problem")

        if "other agent" in situation_lower:
            if agent_model.personality_traits.get("cooperative", 0.5) > 0.6:
                intentions.append("Cooperate with other agent")
            else:
                intentions.append("Compete with other agent")

        return intentions

    def _identify_knowledge_gaps(
        self,
        agent_model: AgentModel,
        situation: str
    ) -> List[str]:
        """Identify what agent likely doesn't know"""
        gaps = []

        # What does self know that agent might not?
        if self.self_model:
            for knowledge_item in self.self_model.knowledge:
                if knowledge_item not in agent_model.mental_state.knowledge:
                    gaps.append(knowledge_item)

        return gaps

    def detect_false_belief(
        self,
        agent_id: str,
        actual_state: Dict[str, Any]
    ) -> List[Tuple[str, str]]:
        """
        Detect false beliefs (beliefs that differ from reality)

        Args:
            agent_id: Agent to check
            actual_state: True state of the world

        Returns:
            List of (belief_content, actual_truth) tuples
        """
        if agent_id not in self.agent_models:
            return []

        agent_model = self.agent_models[agent_id]
        false_beliefs = []

        for belief in agent_model.mental_state.beliefs.values():
            # Check if belief contradicts actual state
            for key, actual_value in actual_state.items():
                if key.lower() in belief.content.lower():
                    # Belief is about this aspect of state
                    if str(actual_value).lower() not in belief.content.lower():
                        # Belief doesn't match reality
                        false_beliefs.append((belief.content, f"{key}: {actual_value}"))

        return false_beliefs

    def update_from_communication(
        self,
        agent_id: str,
        message: str,
        context: Dict[str, Any]
    ):
        """
        Update agent model based on communication

        Messages reveal beliefs, intentions, and knowledge
        """
        if agent_id not in self.agent_models:
            self.register_agent(agent_id, f"Agent_{agent_id}")

        agent_model = self.agent_models[agent_id]

        # Record interaction
        agent_model.record_interaction({
            "type": "communication",
            "message": message,
            "context": context
        })

        # Extract beliefs from message
        # In production, this would use NLP
        message_lower = message.lower()

        if "i believe" in message_lower or "i think" in message_lower:
            # Extract belief content (simplified)
            belief_content = message.split("believe")[-1].strip() if "believe" in message_lower else message
            belief = Belief(
                content=belief_content,
                belief_type=BeliefType.FACTUAL,
                confidence=0.8,
                source="communication",
                evidence=[message]
            )
            agent_model.mental_state.add_belief(belief)

        # Extract intentions
        if "i want" in message_lower or "i intend" in message_lower:
            intention_desc = message.split("want")[-1].strip() if "want" in message_lower else message
            intention = Intention(
                description=intention_desc,
                intention_type=IntentionType.GOAL,
                confidence=0.9,
                indicators=[message]
            )
            agent_model.mental_state.add_intention(intention)

        # Update knowledge
        # Anything agent says, they likely know
        agent_model.mental_state.knowledge.add(message)

        # Update trust based on communication honesty
        # (This is simplified - real version would track and verify claims)
        agent_model.trust_level = min(1.0, agent_model.trust_level + 0.05)

    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of agent's mental model"""
        if agent_id not in self.agent_models:
            return {"error": "Unknown agent"}

        agent_model = self.agent_models[agent_id]
        mental_state = agent_model.mental_state

        return {
            "agent_id": agent_id,
            "name": agent_model.name,
            "beliefs_count": len(mental_state.beliefs),
            "intentions_count": len(mental_state.intentions),
            "current_emotion": mental_state.emotional_state.value,
            "current_intention": mental_state.get_strongest_intention().description if mental_state.get_strongest_intention() else None,
            "trust_level": agent_model.trust_level,
            "capabilities": list(agent_model.capabilities),
            "personality": agent_model.personality_traits,
            "recent_behaviors": agent_model.observed_behaviors[-5:],
        }
