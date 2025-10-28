"""
Agent Handoff Mechanism - Multi-Agent Orchestration Framework
==============================================================

Manages state transfer and communication between agents with:
- State serialization/deserialization
- Context preservation
- Handoff triggers and validation
- Audit trail logging
- Recovery from failed handoffs

Handoff Triggers:
- Task completion
- Capability mismatch
- Resource constraints
- Explicit agent request

Features:
- Full context preservation
- Versioned state format
- Rollback support
- Audit chain integration
"""

import json
import logging
import pickle
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class HandoffTrigger(Enum):
    """Reasons for agent handoff"""
    TASK_COMPLETED = "task_completed"
    CAPABILITY_MISMATCH = "capability_mismatch"
    RESOURCE_CONSTRAINT = "resource_constraint"
    AGENT_REQUEST = "agent_request"
    ERROR_RECOVERY = "error_recovery"
    TIMEOUT = "timeout"


class HandoffStatus(Enum):
    """Handoff execution status"""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class HandoffContext:
    """Context transferred between agents"""
    handoff_id: str
    from_agent: str
    to_agent: str
    trigger: HandoffTrigger
    timestamp: str

    # State data
    task_state: Dict[str, Any]  # Current task state
    shared_memory: Dict[str, Any]  # Shared memory/context
    execution_history: List[Dict[str, Any]]  # Previous actions

    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 0
    deadline: Optional[str] = None

    # Serialization format version
    version: str = "1.0"

    # Additional context
    notes: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Files, data, etc.


@dataclass
class HandoffResult:
    """Result of handoff operation"""
    handoff_id: str
    status: HandoffStatus
    from_agent: str
    to_agent: str
    trigger: HandoffTrigger
    initiated_at: str
    completed_at: Optional[str] = None
    duration_sec: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    context_hash: Optional[str] = None  # For integrity verification


class HandoffManager:
    """
    Manages agent-to-agent handoffs with state preservation.

    Features:
    - Serializes and deserializes agent state
    - Validates handoff compatibility
    - Logs handoff audit trail
    - Handles handoff failures
    """

    def __init__(self, audit_log_path: str = "var/orchestration/handoff_audit.ndjson"):
        """
        Initialize handoff manager.

        Args:
            audit_log_path: Path to handoff audit log
        """
        from pathlib import Path

        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Active handoffs
        self.active_handoffs: Dict[str, HandoffContext] = {}

        # Handoff history
        self.handoff_history: List[HandoffResult] = []

        # Statistics
        self.total_handoffs = 0
        self.successful_handoffs = 0
        self.failed_handoffs = 0

        logger.info(f"HandoffManager initialized (audit: {audit_log_path})")

    def initiate_handoff(
        self,
        from_agent: str,
        to_agent: str,
        trigger: HandoffTrigger,
        task_state: Dict[str, Any],
        shared_memory: Optional[Dict[str, Any]] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        notes: str = ""
    ) -> HandoffContext:
        """
        Initiate agent handoff.

        Args:
            from_agent: Source agent identifier
            to_agent: Target agent identifier
            trigger: Reason for handoff
            task_state: Current task state to transfer
            shared_memory: Shared context/memory
            execution_history: History of actions
            user_id: User identifier
            session_id: Session identifier
            notes: Additional handoff notes

        Returns:
            HandoffContext with transfer details
        """
        handoff_id = str(uuid4())

        context = HandoffContext(
            handoff_id=handoff_id,
            from_agent=from_agent,
            to_agent=to_agent,
            trigger=trigger,
            timestamp=datetime.now().isoformat(),
            task_state=task_state,
            shared_memory=shared_memory or {},
            execution_history=execution_history or [],
            user_id=user_id,
            session_id=session_id,
            notes=notes
        )

        # Store active handoff
        self.active_handoffs[handoff_id] = context

        # Log handoff
        result = HandoffResult(
            handoff_id=handoff_id,
            status=HandoffStatus.INITIATED,
            from_agent=from_agent,
            to_agent=to_agent,
            trigger=trigger,
            initiated_at=context.timestamp
        )

        self._log_handoff(result)

        self.total_handoffs += 1

        logger.info(
            f"Handoff initiated: {from_agent} -> {to_agent} "
            f"(trigger: {trigger.value}, id: {handoff_id})"
        )

        return context

    def complete_handoff(
        self,
        handoff_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> HandoffResult:
        """
        Complete handoff operation.

        Args:
            handoff_id: Handoff identifier
            success: Whether handoff succeeded
            error: Error message if failed

        Returns:
            HandoffResult with completion details
        """
        if handoff_id not in self.active_handoffs:
            raise ValueError(f"Handoff not found: {handoff_id}")

        context = self.active_handoffs[handoff_id]

        # Calculate duration
        initiated_at = datetime.fromisoformat(context.timestamp)
        completed_at = datetime.now()
        duration_sec = (completed_at - initiated_at).total_seconds()

        # Create result
        result = HandoffResult(
            handoff_id=handoff_id,
            status=HandoffStatus.COMPLETED if success else HandoffStatus.FAILED,
            from_agent=context.from_agent,
            to_agent=context.to_agent,
            trigger=context.trigger,
            initiated_at=context.timestamp,
            completed_at=completed_at.isoformat(),
            duration_sec=duration_sec,
            success=success,
            error=error,
            context_hash=self._compute_context_hash(context)
        )

        # Update statistics
        if success:
            self.successful_handoffs += 1
            logger.info(f"Handoff completed successfully: {handoff_id} ({duration_sec:.2f}s)")
        else:
            self.failed_handoffs += 1
            logger.error(f"Handoff failed: {handoff_id} - {error}")

        # Remove from active handoffs
        del self.active_handoffs[handoff_id]

        # Add to history
        self.handoff_history.append(result)

        # Log completion
        self._log_handoff(result)

        return result

    def serialize_context(self, context: HandoffContext) -> str:
        """
        Serialize handoff context to JSON string.

        Args:
            context: Context to serialize

        Returns:
            JSON string representation
        """
        # Convert to dict
        context_dict = asdict(context)

        # Encode trigger as string
        context_dict["trigger"] = context.trigger.value

        # Serialize to JSON
        return json.dumps(context_dict, ensure_ascii=False, sort_keys=True)

    def deserialize_context(self, context_str: str) -> HandoffContext:
        """
        Deserialize handoff context from JSON string.

        Args:
            context_str: JSON string

        Returns:
            HandoffContext object
        """
        context_dict = json.loads(context_str)

        # Convert trigger back to enum
        context_dict["trigger"] = HandoffTrigger(context_dict["trigger"])

        # Reconstruct HandoffContext
        return HandoffContext(**context_dict)

    def _compute_context_hash(self, context: HandoffContext) -> str:
        """Compute hash of context for integrity verification"""
        import hashlib

        context_str = self.serialize_context(context)
        return hashlib.sha256(context_str.encode()).hexdigest()

    def verify_context_integrity(self, context: HandoffContext, expected_hash: str) -> bool:
        """
        Verify context integrity using hash.

        Args:
            context: Context to verify
            expected_hash: Expected SHA256 hash

        Returns:
            True if integrity verified, False otherwise
        """
        actual_hash = self._compute_context_hash(context)
        return actual_hash == expected_hash

    def rollback_handoff(self, handoff_id: str) -> bool:
        """
        Rollback failed handoff.

        Args:
            handoff_id: Handoff to rollback

        Returns:
            True if rollback successful, False otherwise
        """
        if handoff_id not in self.active_handoffs:
            logger.warning(f"Cannot rollback - handoff not active: {handoff_id}")
            return False

        context = self.active_handoffs[handoff_id]

        logger.warning(f"Rolling back handoff: {handoff_id}")

        # Mark as rolled back
        result = HandoffResult(
            handoff_id=handoff_id,
            status=HandoffStatus.ROLLED_BACK,
            from_agent=context.from_agent,
            to_agent=context.to_agent,
            trigger=context.trigger,
            initiated_at=context.timestamp,
            completed_at=datetime.now().isoformat(),
            success=False,
            error="Handoff rolled back"
        )

        # Remove from active
        del self.active_handoffs[handoff_id]

        # Log rollback
        self._log_handoff(result)

        logger.info(f"Handoff rolled back: {handoff_id}")

        return True

    def _log_handoff(self, result: HandoffResult):
        """Log handoff to audit trail"""
        try:
            # Log to audit chain
            from utils.audit_chain import append_jsonl

            log_entry = {
                "event_type": "agent_handoff",
                "handoff_id": result.handoff_id,
                "status": result.status.value,
                "from_agent": result.from_agent,
                "to_agent": result.to_agent,
                "trigger": result.trigger.value,
                "initiated_at": result.initiated_at,
                "completed_at": result.completed_at,
                "duration_sec": result.duration_sec,
                "success": result.success,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            }

            append_jsonl(str(self.audit_log_path), log_entry)

        except Exception as e:
            logger.error(f"Failed to log handoff: {e}")

    def get_handoff_stats(self) -> Dict[str, Any]:
        """Get handoff statistics"""
        return {
            "total_handoffs": self.total_handoffs,
            "successful": self.successful_handoffs,
            "failed": self.failed_handoffs,
            "active": len(self.active_handoffs),
            "success_rate": self.successful_handoffs / max(self.total_handoffs, 1),
            "avg_duration_sec": (
                sum(h.duration_sec for h in self.handoff_history if h.duration_sec)
                / max(len(self.handoff_history), 1)
            ) if self.handoff_history else 0.0
        }

    def get_active_handoffs(self) -> List[HandoffContext]:
        """Get currently active handoffs"""
        return list(self.active_handoffs.values())

    def get_handoff_history(
        self,
        limit: Optional[int] = None,
        agent: Optional[str] = None
    ) -> List[HandoffResult]:
        """
        Get handoff history.

        Args:
            limit: Maximum number of results
            agent: Filter by agent (from_agent or to_agent)

        Returns:
            List of handoff results
        """
        history = self.handoff_history

        if agent:
            history = [
                h for h in history
                if h.from_agent == agent or h.to_agent == agent
            ]

        if limit:
            history = history[-limit:]

        return history


# Singleton instance
_handoff_manager: Optional[HandoffManager] = None


def get_handoff_manager() -> HandoffManager:
    """Get singleton handoff manager instance"""
    global _handoff_manager
    if _handoff_manager is None:
        _handoff_manager = HandoffManager()
    return _handoff_manager
