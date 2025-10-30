"""
Dialogue Manager

Manages multi-turn conversations:
- Dialogue state tracking
- Context management
- Turn-taking
- Conversation flow control
- Dialogue acts
- Topic tracking
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque, defaultdict


class DialogueAct(str, Enum):
    """Types of dialogue acts"""
    INFORM = "inform"
    REQUEST = "request"
    CONFIRM = "confirm"
    DENY = "deny"
    GREET = "greet"
    GOODBYE = "goodbye"
    THANK = "thank"
    APOLOGY = "apology"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"


class DialoguePhase(str, Enum):
    """Phases of dialogue"""
    OPENING = "opening"
    INFORMATION_GATHERING = "information_gathering"
    TASK_EXECUTION = "task_execution"
    CONFIRMATION = "confirmation"
    CLOSING = "closing"


class SpeakerRole(str, Enum):
    """Role of speaker"""
    USER = "user"
    SYSTEM = "system"


@dataclass
class Turn:
    """A single turn in dialogue"""
    turn_id: int
    speaker: SpeakerRole
    utterance: str
    dialogue_act: DialogueAct
    timestamp: float = field(default_factory=time.time)
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DialogueState:
    """Current state of dialogue"""
    dialogue_id: str
    phase: DialoguePhase = DialoguePhase.OPENING
    active_topic: Optional[str] = None
    mentioned_topics: List[str] = field(default_factory=list)
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    required_slots: Set[str] = field(default_factory=set)
    user_goals: List[str] = field(default_factory=list)
    system_goals: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_system_act: Optional[DialogueAct] = None
    last_user_act: Optional[DialogueAct] = None
    pending_confirmations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


class DialogueManager:
    """
    Dialogue Manager for Multi-Turn Conversations

    Features:
    - State tracking across turns
    - Context management
    - Dialogue flow control
    - Topic tracking
    - Slot filling
    - Grounding and confirmation
    - Error recovery
    """

    def __init__(self):
        # Dialogue history
        self.dialogues: Dict[str, List[Turn]] = defaultdict(list)
        self.states: Dict[str, DialogueState] = {}

        # Dialogue policies
        self.phase_transitions = self._build_phase_transitions()
        self.act_responses = self._build_act_responses()

        # Context window
        self.context_window_size = 5

        # Current active dialogue
        self.current_dialogue_id: Optional[str] = None
        self.dialogue_counter = 0

    def _build_phase_transitions(self) -> Dict[DialoguePhase, List[DialoguePhase]]:
        """Define valid phase transitions"""
        return {
            DialoguePhase.OPENING: [DialoguePhase.INFORMATION_GATHERING],
            DialoguePhase.INFORMATION_GATHERING: [
                DialoguePhase.TASK_EXECUTION,
                DialoguePhase.CONFIRMATION,
                DialoguePhase.CLOSING
            ],
            DialoguePhase.TASK_EXECUTION: [
                DialoguePhase.CONFIRMATION,
                DialoguePhase.INFORMATION_GATHERING,
                DialoguePhase.CLOSING
            ],
            DialoguePhase.CONFIRMATION: [
                DialoguePhase.TASK_EXECUTION,
                DialoguePhase.CLOSING
            ],
            DialoguePhase.CLOSING: [],
        }

    def _build_act_responses(self) -> Dict[DialogueAct, DialogueAct]:
        """Map user dialogue acts to system response acts"""
        return {
            DialogueAct.GREET: DialogueAct.GREET,
            DialogueAct.GOODBYE: DialogueAct.GOODBYE,
            DialogueAct.THANK: DialogueAct.ACKNOWLEDGE,
            DialogueAct.APOLOGY: DialogueAct.ACKNOWLEDGE,
            DialogueAct.REQUEST: DialogueAct.INFORM,
            DialogueAct.INFORM: DialogueAct.ACKNOWLEDGE,
            DialogueAct.CONFIRM: DialogueAct.ACKNOWLEDGE,
            DialogueAct.DENY: DialogueAct.CLARIFY,
            DialogueAct.OFFER: DialogueAct.ACCEPT,
        }

    def start_dialogue(self, dialogue_id: Optional[str] = None) -> DialogueState:
        """
        Start a new dialogue

        Args:
            dialogue_id: Optional custom dialogue ID

        Returns:
            Initial dialogue state
        """
        if dialogue_id is None:
            self.dialogue_counter += 1
            dialogue_id = f"dialogue_{self.dialogue_counter}"

        # Create initial state
        state = DialogueState(
            dialogue_id=dialogue_id,
            phase=DialoguePhase.OPENING
        )

        self.states[dialogue_id] = state
        self.dialogues[dialogue_id] = []
        self.current_dialogue_id = dialogue_id

        return state

    def process_turn(
        self,
        utterance: str,
        dialogue_act: DialogueAct,
        intent: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        slots: Optional[Dict[str, Any]] = None
    ) -> Turn:
        """
        Process user turn

        Args:
            utterance: User's utterance
            dialogue_act: Dialogue act type
            intent: Optional intent
            entities: Optional extracted entities
            slots: Optional filled slots

        Returns:
            Turn object
        """
        if not self.current_dialogue_id:
            self.start_dialogue()

        state = self.states[self.current_dialogue_id]
        state.turn_count += 1

        # Create turn
        turn = Turn(
            turn_id=state.turn_count,
            speaker=SpeakerRole.USER,
            utterance=utterance,
            dialogue_act=dialogue_act,
            intent=intent,
            entities=entities or [],
            slots=slots or {}
        )

        # Add to history
        self.dialogues[self.current_dialogue_id].append(turn)

        # Update state
        self._update_state(turn)

        return turn

    def _update_state(self, turn: Turn):
        """Update dialogue state based on turn"""
        state = self.states[self.current_dialogue_id]

        # Update last act
        if turn.speaker == SpeakerRole.USER:
            state.last_user_act = turn.dialogue_act
        else:
            state.last_system_act = turn.dialogue_act

        # Update slots
        for slot, value in turn.slots.items():
            state.filled_slots[slot] = value

        # Extract topics
        if turn.intent:
            if turn.intent not in state.mentioned_topics:
                state.mentioned_topics.append(turn.intent)
                state.active_topic = turn.intent

        # Update phase based on dialogue act
        self._update_phase(turn.dialogue_act)

        # Update context
        state.context["last_utterance"] = turn.utterance
        state.context["last_intent"] = turn.intent

    def _update_phase(self, dialogue_act: DialogueAct):
        """Update dialogue phase based on dialogue act"""
        state = self.states[self.current_dialogue_id]
        current_phase = state.phase

        # Phase transition logic
        if dialogue_act == DialogueAct.GREET and current_phase == DialoguePhase.OPENING:
            # Stay in opening
            pass
        elif dialogue_act in [DialogueAct.REQUEST, DialogueAct.INFORM]:
            if current_phase == DialoguePhase.OPENING:
                state.phase = DialoguePhase.INFORMATION_GATHERING
            elif current_phase == DialoguePhase.CONFIRMATION:
                state.phase = DialoguePhase.TASK_EXECUTION
        elif dialogue_act == DialogueAct.CONFIRM:
            if current_phase == DialoguePhase.INFORMATION_GATHERING:
                state.phase = DialoguePhase.TASK_EXECUTION
        elif dialogue_act == DialogueAct.GOODBYE:
            state.phase = DialoguePhase.CLOSING

    def generate_system_turn(
        self,
        content: Dict[str, Any],
        dialogue_act: Optional[DialogueAct] = None
    ) -> Turn:
        """
        Generate system turn

        Args:
            content: Content to communicate
            dialogue_act: Optional dialogue act (auto-determined if not provided)

        Returns:
            System turn
        """
        state = self.states[self.current_dialogue_id]
        state.turn_count += 1

        # Determine dialogue act if not provided
        if dialogue_act is None:
            dialogue_act = self._determine_system_act()

        # Generate utterance (placeholder - would use NLG in real system)
        utterance = self._generate_utterance(content, dialogue_act)

        # Create turn
        turn = Turn(
            turn_id=state.turn_count,
            speaker=SpeakerRole.SYSTEM,
            utterance=utterance,
            dialogue_act=dialogue_act,
            slots=content
        )

        # Add to history
        self.dialogues[self.current_dialogue_id].append(turn)

        # Update state
        self._update_state(turn)

        return turn

    def _determine_system_act(self) -> DialogueAct:
        """Determine appropriate system dialogue act"""
        state = self.states[self.current_dialogue_id]

        # Respond based on last user act
        if state.last_user_act:
            return self.act_responses.get(state.last_user_act, DialogueAct.INFORM)

        # Default based on phase
        phase_to_act = {
            DialoguePhase.OPENING: DialogueAct.GREET,
            DialoguePhase.INFORMATION_GATHERING: DialogueAct.REQUEST,
            DialoguePhase.TASK_EXECUTION: DialogueAct.INFORM,
            DialoguePhase.CONFIRMATION: DialogueAct.CONFIRM,
            DialoguePhase.CLOSING: DialogueAct.GOODBYE,
        }

        return phase_to_act.get(state.phase, DialogueAct.INFORM)

    def _generate_utterance(
        self,
        content: Dict[str, Any],
        dialogue_act: DialogueAct
    ) -> str:
        """Generate utterance from content (placeholder)"""
        # In real system, this would use NLG engine
        if dialogue_act == DialogueAct.GREET:
            return "Hello! How can I help you today?"
        elif dialogue_act == DialogueAct.GOODBYE:
            return "Goodbye! Have a great day!"
        elif dialogue_act == DialogueAct.REQUEST:
            missing_slots = self.get_missing_slots()
            if missing_slots:
                return f"Could you provide the {missing_slots[0]}?"
            return "What would you like to do?"
        elif dialogue_act == DialogueAct.INFORM:
            if "result" in content:
                return f"Here's the result: {content['result']}"
            return str(content)
        elif dialogue_act == DialogueAct.CONFIRM:
            return "Is this correct?"
        elif dialogue_act == DialogueAct.ACKNOWLEDGE:
            return "Got it!"
        else:
            return str(content)

    def get_context(self, window_size: Optional[int] = None) -> List[Turn]:
        """
        Get recent dialogue context

        Args:
            window_size: Number of recent turns (default: context_window_size)

        Returns:
            List of recent turns
        """
        if not self.current_dialogue_id:
            return []

        size = window_size or self.context_window_size
        history = self.dialogues[self.current_dialogue_id]

        return history[-size:]

    def get_state(self) -> Optional[DialogueState]:
        """Get current dialogue state"""
        if not self.current_dialogue_id:
            return None

        return self.states.get(self.current_dialogue_id)

    def get_missing_slots(self) -> List[str]:
        """Get list of required slots that haven't been filled"""
        if not self.current_dialogue_id:
            return []

        state = self.states[self.current_dialogue_id]
        return [
            slot for slot in state.required_slots
            if slot not in state.filled_slots
        ]

    def add_required_slot(self, slot_name: str):
        """Add a required slot to current dialogue"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            state.required_slots.add(slot_name)

    def fill_slot(self, slot_name: str, value: Any):
        """Fill a slot with value"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            state.filled_slots[slot_name] = value

    def is_slot_filled(self, slot_name: str) -> bool:
        """Check if slot is filled"""
        if not self.current_dialogue_id:
            return False

        state = self.states[self.current_dialogue_id]
        return slot_name in state.filled_slots

    def get_slot_value(self, slot_name: str) -> Optional[Any]:
        """Get value of filled slot"""
        if not self.current_dialogue_id:
            return None

        state = self.states[self.current_dialogue_id]
        return state.filled_slots.get(slot_name)

    def all_slots_filled(self) -> bool:
        """Check if all required slots are filled"""
        missing = self.get_missing_slots()
        return len(missing) == 0

    def request_confirmation(self, item: str):
        """Request user confirmation for item"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            state.pending_confirmations.append(item)

    def confirm(self, item: str):
        """Mark item as confirmed"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            if item in state.pending_confirmations:
                state.pending_confirmations.remove(item)

    def has_pending_confirmations(self) -> bool:
        """Check if there are pending confirmations"""
        if not self.current_dialogue_id:
            return False

        state = self.states[self.current_dialogue_id]
        return len(state.pending_confirmations) > 0

    def get_topic_history(self) -> List[str]:
        """Get history of mentioned topics"""
        if not self.current_dialogue_id:
            return []

        state = self.states[self.current_dialogue_id]
        return state.mentioned_topics.copy()

    def get_active_topic(self) -> Optional[str]:
        """Get current active topic"""
        if not self.current_dialogue_id:
            return None

        state = self.states[self.current_dialogue_id]
        return state.active_topic

    def switch_topic(self, new_topic: str):
        """Switch to new topic"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            if new_topic not in state.mentioned_topics:
                state.mentioned_topics.append(new_topic)
            state.active_topic = new_topic

    def end_dialogue(self) -> Dict[str, Any]:
        """
        End current dialogue

        Returns:
            Summary of dialogue
        """
        if not self.current_dialogue_id:
            return {}

        state = self.states[self.current_dialogue_id]
        history = self.dialogues[self.current_dialogue_id]

        summary = {
            "dialogue_id": self.current_dialogue_id,
            "turn_count": state.turn_count,
            "duration": time.time() - state.started_at,
            "topics": state.mentioned_topics,
            "filled_slots": state.filled_slots.copy(),
            "final_phase": state.phase,
            "user_turns": sum(1 for t in history if t.speaker == SpeakerRole.USER),
            "system_turns": sum(1 for t in history if t.speaker == SpeakerRole.SYSTEM),
        }

        # Mark as closed
        state.phase = DialoguePhase.CLOSING

        # Don't delete dialogue (keep for history), just clear current
        self.current_dialogue_id = None

        return summary

    def get_dialogue_history(self, dialogue_id: str) -> List[Turn]:
        """Get complete history of a dialogue"""
        return self.dialogues.get(dialogue_id, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get dialogue manager statistics"""
        total_turns = sum(len(h) for h in self.dialogues.values())
        total_dialogues = len(self.dialogues)

        if total_dialogues > 0:
            avg_turns_per_dialogue = total_turns / total_dialogues
        else:
            avg_turns_per_dialogue = 0

        return {
            "total_dialogues": total_dialogues,
            "total_turns": total_turns,
            "avg_turns_per_dialogue": avg_turns_per_dialogue,
            "active_dialogue": self.current_dialogue_id,
        }

    def recover_from_error(self, error_type: str) -> DialogueAct:
        """
        Recover from dialogue error

        Args:
            error_type: Type of error encountered

        Returns:
            Appropriate recovery dialogue act
        """
        if error_type == "misunderstanding":
            return DialogueAct.CLARIFY
        elif error_type == "missing_information":
            return DialogueAct.REQUEST
        elif error_type == "invalid_action":
            return DialogueAct.INFORM
        else:
            return DialogueAct.APOLOGY

    def handle_interruption(self):
        """Handle user interruption"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            # Save current state to context
            state.context["interrupted_phase"] = state.phase
            state.context["interrupted_topic"] = state.active_topic

    def resume_from_interruption(self):
        """Resume dialogue after interruption"""
        if self.current_dialogue_id:
            state = self.states[self.current_dialogue_id]
            # Restore state
            if "interrupted_phase" in state.context:
                state.phase = state.context["interrupted_phase"]
            if "interrupted_topic" in state.context:
                state.active_topic = state.context["interrupted_topic"]

    def get_grounding_status(self) -> Dict[str, Any]:
        """
        Get grounding status (mutual understanding check)

        Returns:
            Status of mutual understanding
        """
        if not self.current_dialogue_id:
            return {}

        state = self.states[self.current_dialogue_id]

        return {
            "filled_slots": len(state.filled_slots),
            "required_slots": len(state.required_slots),
            "completion": len(state.filled_slots) / max(len(state.required_slots), 1),
            "pending_confirmations": len(state.pending_confirmations),
            "topics_covered": len(state.mentioned_topics),
        }
