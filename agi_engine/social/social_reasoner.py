"""
Social Reasoning - Understanding Social Norms and Behavior

This module enables AGI to understand social norms, recognize intent from
behavior, predict social outcomes, and reason about social situations.

Key capabilities:
- Social norm understanding and compliance
- Intent recognition from observed behavior
- Behavior prediction in social contexts
- Social appropriateness assessment
- Cultural awareness
- Group dynamics understanding
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class NormType(str, Enum):
    """Types of social norms"""
    MORAL = "moral"  # Right/wrong
    CONVENTIONAL = "conventional"  # Social customs
    ETIQUETTE = "etiquette"  # Polite behavior
    LEGAL = "legal"  # Laws and regulations
    PROFESSIONAL = "professional"  # Workplace norms
    CULTURAL = "cultural"  # Culture-specific norms


class BehaviorType(str, Enum):
    """Types of social behaviors"""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    ALTRUISTIC = "altruistic"
    SELFISH = "selfish"
    AGGRESSIVE = "aggressive"
    SUBMISSIVE = "submissive"
    NEUTRAL = "neutral"


class SocialRole(str, Enum):
    """Social roles in interactions"""
    LEADER = "leader"
    FOLLOWER = "follower"
    MEDIATOR = "mediator"
    OBSERVER = "observer"
    PARTICIPANT = "participant"
    AUTHORITY = "authority"


@dataclass
class SocialNorm:
    """A social norm or rule"""
    norm_id: str
    description: str
    norm_type: NormType
    context: str  # When this norm applies
    importance: float = 0.7  # 0-1, how critical is compliance
    universality: float = 0.5  # 0-1, how universal vs. context-specific
    consequences: List[str] = field(default_factory=list)  # What happens if violated
    examples: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    learned_from: Optional[str] = None
    confidence: float = 0.8

    def applies_to_context(self, context: str) -> bool:
        """Check if norm applies to given context"""
        context_lower = context.lower()
        norm_context_lower = self.context.lower()

        # Simple keyword matching
        context_keywords = set(norm_context_lower.split())
        given_keywords = set(context_lower.split())

        overlap = context_keywords.intersection(given_keywords)
        return len(overlap) > 0 or self.universality > 0.8


@dataclass
class Intent:
    """Recognized intent from behavior"""
    description: str
    intent_type: str  # goal, communication, emotion, etc.
    confidence: float
    evidence: List[str] = field(default_factory=list)
    predicted_outcome: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Behavior:
    """Observed or predicted behavior"""
    actor: str
    action: str
    behavior_type: BehaviorType
    context: Dict[str, Any]
    social_appropriateness: float = 0.5  # 0-1
    norm_compliance: Dict[str, bool] = field(default_factory=dict)
    predicted_reactions: List[Tuple[str, str, float]] = field(default_factory=list)  # (agent, reaction, probability)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SocialContext:
    """Context of a social situation"""
    context_id: str
    description: str
    participants: List[str]
    roles: Dict[str, SocialRole] = field(default_factory=dict)  # agent -> role
    relationships: Dict[Tuple[str, str], str] = field(default_factory=dict)  # (agent1, agent2) -> relationship type
    norms_active: List[str] = field(default_factory=list)  # Active norm IDs
    power_dynamics: Dict[str, float] = field(default_factory=dict)  # agent -> power level (0-1)
    formality: float = 0.5  # 0-1, informal to formal
    timestamp: float = field(default_factory=time.time)


class SocialReasoner:
    """
    Social reasoning system for understanding norms and behavior

    Features:
    - Learn and apply social norms
    - Recognize intent from behavior
    - Predict social outcomes
    - Assess appropriateness
    - Understand group dynamics
    - Cultural awareness
    """

    def __init__(self):
        self.norms: Dict[str, SocialNorm] = {}
        self.contexts: Dict[str, SocialContext] = {}
        self.behavior_history: List[Behavior] = []
        self.norm_violations: List[Dict[str, Any]] = []

        # Initialize with common universal norms
        self._init_basic_norms()

    def _init_basic_norms(self):
        """Initialize with fundamental social norms"""
        basic_norms = [
            SocialNorm(
                norm_id="norm_001",
                description="Don't harm others",
                norm_type=NormType.MORAL,
                context="all situations",
                importance=1.0,
                universality=1.0,
                consequences=["loss of trust", "social exclusion", "legal consequences"],
                examples=["physical harm", "emotional harm", "deception"]
            ),
            SocialNorm(
                norm_id="norm_002",
                description="Keep promises and commitments",
                norm_type=NormType.MORAL,
                context="all situations",
                importance=0.9,
                universality=0.9,
                consequences=["loss of trust", "damaged relationships"],
                examples=["honoring agreements", "following through on commitments"]
            ),
            SocialNorm(
                norm_id="norm_003",
                description="Respect personal space",
                norm_type=NormType.ETIQUETTE,
                context="physical interactions",
                importance=0.7,
                universality=0.8,
                consequences=["discomfort", "offense"],
                examples=["maintain appropriate distance", "don't touch without permission"]
            ),
            SocialNorm(
                norm_id="norm_004",
                description="Take turns in conversation",
                norm_type=NormType.CONVENTIONAL,
                context="conversations",
                importance=0.6,
                universality=0.85,
                consequences=["frustration", "exclusion from conversation"],
                examples=["don't interrupt", "allow others to speak"]
            ),
            SocialNorm(
                norm_id="norm_005",
                description="Help others in need",
                norm_type=NormType.MORAL,
                context="when able and safe",
                importance=0.8,
                universality=0.9,
                consequences=["guilt", "social judgment"],
                examples=["assist someone struggling", "offer support"]
            ),
            SocialNorm(
                norm_id="norm_006",
                description="Be honest in communication",
                norm_type=NormType.MORAL,
                context="most situations",
                importance=0.85,
                universality=0.85,
                consequences=["loss of trust", "damaged credibility"],
                examples=["tell the truth", "don't deceive"],
                exceptions=["white lies to protect feelings", "necessary deception for safety"]
            ),
            SocialNorm(
                norm_id="norm_007",
                description="Share resources fairly",
                norm_type=NormType.MORAL,
                context="group situations",
                importance=0.75,
                universality=0.8,
                consequences=["conflict", "resentment"],
                examples=["equitable distribution", "consider others' needs"]
            ),
            SocialNorm(
                norm_id="norm_008",
                description="Respect authority in formal settings",
                norm_type=NormType.CONVENTIONAL,
                context="formal hierarchical situations",
                importance=0.6,
                universality=0.7,
                consequences=["sanctions", "loss of privileges"],
                examples=["follow instructions from supervisors", "show deference"]
            ),
        ]

        for norm in basic_norms:
            self.norms[norm.norm_id] = norm

    def learn_norm(
        self,
        description: str,
        norm_type: NormType,
        context: str,
        importance: float = 0.7,
        examples: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> SocialNorm:
        """
        Learn a new social norm from observation or instruction

        Args:
            description: What the norm prescribes
            norm_type: Type of norm
            context: When it applies
            importance: How critical compliance is
            examples: Example situations
            source: How norm was learned

        Returns:
            The created SocialNorm
        """
        norm_id = f"norm_{len(self.norms) + 1:03d}"

        norm = SocialNorm(
            norm_id=norm_id,
            description=description,
            norm_type=norm_type,
            context=context,
            importance=importance,
            examples=examples or [],
            learned_from=source
        )

        self.norms[norm_id] = norm
        return norm

    def recognize_intent(
        self,
        behavior: str,
        actor: str,
        context: Dict[str, Any]
    ) -> List[Intent]:
        """
        Recognize possible intents from observed behavior

        Args:
            behavior: Description of behavior
            actor: Who performed behavior
            context: Contextual information

        Returns:
            List of possible intents, ordered by confidence
        """
        intents = []
        behavior_lower = behavior.lower()

        # Goal-oriented intent
        if any(word in behavior_lower for word in ["move", "reach", "take", "acquire"]):
            target = context.get("target", "object")
            intents.append(Intent(
                description=f"Acquire or interact with {target}",
                intent_type="goal",
                confidence=0.8,
                evidence=[behavior],
                predicted_outcome=f"{actor} obtains {target}"
            ))

        # Communication intent
        if any(word in behavior_lower for word in ["speak", "say", "signal", "gesture", "write"]):
            intents.append(Intent(
                description="Communicate information or feelings",
                intent_type="communication",
                confidence=0.85,
                evidence=[behavior],
                predicted_outcome="Information is transmitted"
            ))

        # Helping intent
        if any(word in behavior_lower for word in ["help", "assist", "support", "aid"]):
            target = context.get("beneficiary", "others")
            intents.append(Intent(
                description=f"Help {target}",
                intent_type="altruistic",
                confidence=0.8,
                evidence=[behavior],
                predicted_outcome=f"{target} receives assistance"
            ))

        # Avoidance intent
        if any(word in behavior_lower for word in ["avoid", "flee", "retreat", "hide"]):
            threat = context.get("threat", "danger")
            intents.append(Intent(
                description=f"Avoid {threat}",
                intent_type="self_preservation",
                confidence=0.85,
                evidence=[behavior],
                predicted_outcome=f"{actor} escapes {threat}"
            ))

        # Aggressive intent
        if any(word in behavior_lower for word in ["attack", "strike", "threaten", "intimidate"]):
            target = context.get("target", "others")
            intents.append(Intent(
                description=f"Harm or intimidate {target}",
                intent_type="aggressive",
                confidence=0.75,
                evidence=[behavior],
                predicted_outcome=f"{target} is harmed or deterred"
            ))

        # Cooperative intent
        if any(word in behavior_lower for word in ["cooperate", "collaborate", "work together", "coordinate"]):
            intents.append(Intent(
                description="Work together with others",
                intent_type="cooperative",
                confidence=0.8,
                evidence=[behavior],
                predicted_outcome="Joint goal is achieved"
            ))

        # Sort by confidence
        intents.sort(key=lambda i: i.confidence, reverse=True)

        return intents

    def assess_appropriateness(
        self,
        behavior: str,
        context: SocialContext
    ) -> Tuple[float, List[str]]:
        """
        Assess social appropriateness of behavior in context

        Args:
            behavior: Behavior to assess
            context: Social context

        Returns:
            (appropriateness_score, reasons) tuple
        """
        score = 0.5  # Neutral baseline
        reasons = []

        # Check against active norms
        active_norms = [self.norms[norm_id] for norm_id in context.norms_active if norm_id in self.norms]

        behavior_lower = behavior.lower()

        for norm in active_norms:
            # Check for violations
            if self._violates_norm(behavior_lower, norm):
                penalty = norm.importance * 0.3
                score -= penalty
                reasons.append(f"Violates: {norm.description}")

            # Check for compliance
            elif self._complies_with_norm(behavior_lower, norm):
                bonus = norm.importance * 0.2
                score += bonus
                reasons.append(f"Complies with: {norm.description}")

        # Check formality match
        if context.formality > 0.7:  # Formal context
            informal_markers = ["hey", "yeah", "gonna", "wanna"]
            if any(marker in behavior_lower for marker in informal_markers):
                score -= 0.2
                reasons.append("Too informal for context")

        elif context.formality < 0.3:  # Informal context
            formal_markers = ["salutations", "herewith", "pursuant"]
            if any(marker in behavior_lower for marker in formal_markers):
                score -= 0.1
                reasons.append("Overly formal for context")

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        return score, reasons

    def _violates_norm(self, behavior: str, norm: SocialNorm) -> bool:
        """Check if behavior violates a norm"""
        # This is simplified - production would use NLP and semantic understanding

        behavior_lower = behavior.lower()

        # Check against known violations
        if norm.norm_id == "norm_001" and any(word in behavior_lower for word in ["harm", "hurt", "attack", "deceive"]):
            return True

        if norm.norm_id == "norm_003" and "invade personal space" in behavior_lower:
            return True

        if norm.norm_id == "norm_004" and "interrupt" in behavior_lower:
            return True

        if norm.norm_id == "norm_006" and any(word in behavior_lower for word in ["lie", "deceive", "mislead"]):
            # Check for exceptions
            if any(exc in behavior_lower for exc in ["protect", "safety"]):
                return False
            return True

        return False

    def _complies_with_norm(self, behavior: str, norm: SocialNorm) -> bool:
        """Check if behavior complies with a norm"""
        behavior_lower = behavior.lower()

        if norm.norm_id == "norm_005" and any(word in behavior_lower for word in ["help", "assist", "support"]):
            return True

        if norm.norm_id == "norm_006" and any(word in behavior_lower for word in ["honest", "truthful", "transparent"]):
            return True

        if norm.norm_id == "norm_007" and any(word in behavior_lower for word in ["share", "distribute fairly"]):
            return True

        return False

    def predict_social_outcome(
        self,
        action: str,
        actor: str,
        context: SocialContext
    ) -> Dict[str, Any]:
        """
        Predict social outcomes of an action

        Args:
            action: Proposed action
            actor: Who will perform action
            context: Social context

        Returns:
            Dictionary with predictions
        """
        # Assess appropriateness
        appropriateness, reasons = self.assess_appropriateness(action, context)

        # Predict reactions from other participants
        reactions = []
        for participant in context.participants:
            if participant != actor:
                reaction = self._predict_reaction(
                    participant, action, appropriateness, context
                )
                reactions.append(reaction)

        # Predict relationship changes
        relationship_changes = self._predict_relationship_changes(
            actor, action, appropriateness, context
        )

        # Predict norm reinforcement or change
        norm_effects = []
        if appropriateness > 0.7:
            norm_effects.append("Reinforces positive norms")
        elif appropriateness < 0.3:
            norm_effects.append("May weaken norm compliance")

        return {
            "action": action,
            "actor": actor,
            "appropriateness": appropriateness,
            "appropriateness_reasons": reasons,
            "predicted_reactions": reactions,
            "relationship_changes": relationship_changes,
            "norm_effects": norm_effects,
            "overall_outcome": "positive" if appropriateness > 0.6 else "negative" if appropriateness < 0.4 else "neutral"
        }

    def _predict_reaction(
        self,
        participant: str,
        action: str,
        appropriateness: float,
        context: SocialContext
    ) -> Dict[str, Any]:
        """Predict how a participant will react"""
        # Get participant's role and power
        role = context.roles.get(participant, SocialRole.PARTICIPANT)
        power = context.power_dynamics.get(participant, 0.5)

        # Base reaction on appropriateness
        if appropriateness > 0.7:
            reaction = "positive"
            emotion = "pleased"
            probability = 0.8
        elif appropriateness < 0.3:
            reaction = "negative"
            emotion = "displeased"
            probability = 0.8
        else:
            reaction = "neutral"
            emotion = "neutral"
            probability = 0.6

        # Modify based on role
        if role == SocialRole.AUTHORITY and appropriateness < 0.5:
            reaction = "enforcement"
            probability = 0.9

        elif role == SocialRole.MEDIATOR:
            reaction = "mediation"
            probability = 0.7

        return {
            "participant": participant,
            "reaction": reaction,
            "emotion": emotion,
            "probability": probability,
            "role": role.value
        }

    def _predict_relationship_changes(
        self,
        actor: str,
        action: str,
        appropriateness: float,
        context: SocialContext
    ) -> List[Dict[str, Any]]:
        """Predict how relationships will change"""
        changes = []

        for participant in context.participants:
            if participant != actor:
                # Get current relationship if exists
                rel_key = (actor, participant)
                current_rel = context.relationships.get(rel_key, "neutral")

                # Predict change
                if appropriateness > 0.7:
                    change = "improved"
                    magnitude = 0.1
                elif appropriateness < 0.3:
                    change = "worsened"
                    magnitude = -0.15
                else:
                    change = "unchanged"
                    magnitude = 0.0

                changes.append({
                    "with": participant,
                    "current": current_rel,
                    "change": change,
                    "magnitude": magnitude
                })

        return changes

    def create_context(
        self,
        description: str,
        participants: List[str],
        formality: float = 0.5
    ) -> SocialContext:
        """
        Create a new social context

        Args:
            description: Description of situation
            participants: List of participant IDs
            formality: How formal the context is (0-1)

        Returns:
            Created SocialContext
        """
        context_id = f"context_{int(time.time())}_{len(self.contexts)}"

        # Determine active norms based on context
        active_norms = []
        for norm_id, norm in self.norms.items():
            if norm.applies_to_context(description):
                active_norms.append(norm_id)

        context = SocialContext(
            context_id=context_id,
            description=description,
            participants=participants,
            norms_active=active_norms,
            formality=formality
        )

        self.contexts[context_id] = context
        return context

    def assign_roles(
        self,
        context_id: str,
        role_assignments: Dict[str, SocialRole]
    ):
        """Assign social roles to participants in context"""
        if context_id not in self.contexts:
            return

        context = self.contexts[context_id]
        context.roles.update(role_assignments)

    def set_relationship(
        self,
        context_id: str,
        agent1: str,
        agent2: str,
        relationship_type: str
    ):
        """Set relationship between two agents in context"""
        if context_id not in self.contexts:
            return

        context = self.contexts[context_id]
        context.relationships[(agent1, agent2)] = relationship_type
        # Relationships are typically symmetric
        context.relationships[(agent2, agent1)] = relationship_type

    def record_behavior(
        self,
        actor: str,
        action: str,
        behavior_type: BehaviorType,
        context: SocialContext
    ) -> Behavior:
        """Record observed behavior"""
        # Assess appropriateness
        appropriateness, reasons = self.assess_appropriateness(action, context)

        # Check norm compliance
        norm_compliance = {}
        for norm_id in context.norms_active:
            if norm_id in self.norms:
                norm = self.norms[norm_id]
                complies = self._complies_with_norm(action.lower(), norm)
                violates = self._violates_norm(action.lower(), norm)

                if complies:
                    norm_compliance[norm_id] = True
                elif violates:
                    norm_compliance[norm_id] = False
                    # Record violation
                    self.norm_violations.append({
                        "actor": actor,
                        "action": action,
                        "norm": norm.description,
                        "context": context.description,
                        "timestamp": time.time()
                    })

        behavior = Behavior(
            actor=actor,
            action=action,
            behavior_type=behavior_type,
            context={"context_id": context.context_id, "description": context.description},
            social_appropriateness=appropriateness,
            norm_compliance=norm_compliance
        )

        self.behavior_history.append(behavior)

        # Keep only recent history
        if len(self.behavior_history) > 1000:
            self.behavior_history = self.behavior_history[-1000:]

        return behavior

    def get_norms_for_context(self, context: str) -> List[SocialNorm]:
        """Get all norms applicable to a context"""
        applicable = []

        for norm in self.norms.values():
            if norm.applies_to_context(context):
                applicable.append(norm)

        # Sort by importance
        applicable.sort(key=lambda n: n.importance, reverse=True)

        return applicable

    def get_statistics(self) -> Dict[str, Any]:
        """Get social reasoning statistics"""
        return {
            "total_norms": len(self.norms),
            "norms_by_type": {
                norm_type.value: sum(1 for n in self.norms.values() if n.norm_type == norm_type)
                for norm_type in NormType
            },
            "behaviors_recorded": len(self.behavior_history),
            "norm_violations": len(self.norm_violations),
            "active_contexts": len(self.contexts),
            "recent_violations": self.norm_violations[-5:] if self.norm_violations else []
        }
