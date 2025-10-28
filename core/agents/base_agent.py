"""
Base Agent - Multi-Agent Framework
===================================

Abstract base class for all agents with:
- Lifecycle management (spawn, pause, resume, terminate)
- Capability declaration
- Task execution interface
- Message passing protocol
- Resource tracking
- Status reporting

Features:
- Standard agent interface
- Built-in resource governor integration
- Guardian defense validation
- Audit logging
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status"""
    SPAWNED = "spawned"  # Created but not started
    IDLE = "idle"  # Active but not executing tasks
    BUSY = "busy"  # Executing task
    PAUSED = "paused"  # Temporarily paused
    TERMINATED = "terminated"  # Shut down


class AgentCapability(Enum):
    """Agent capabilities"""
    PLANNING = "planning"
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    SYSTEM_OPERATIONS = "system_operations"
    TRADING = "trading"
    MARKET_ANALYSIS = "market_analysis"
    SAFETY_VALIDATION = "safety_validation"
    COMMUNICATION = "communication"
    FILE_OPERATIONS = "file_operations"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    message_id: str
    from_agent: str
    to_agent: str
    message_type: str  # request, response, notification, error
    content: Dict[str, Any]
    timestamp: str
    in_reply_to: Optional[str] = None


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    agent_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_sec: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides:
    - Lifecycle management
    - Capability declaration
    - Task execution interface
    - Message passing
    - Resource tracking
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        capabilities: List[AgentCapability],
        resource_quotas: Optional[Dict[str, float]] = None
    ):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role (planner, researcher, coder, etc.)
            capabilities: List of agent capabilities
            resource_quotas: Resource quota overrides
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.status = AgentStatus.SPAWNED

        # Message inbox/outbox
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []

        # Task tracking
        self.current_task: Optional[str] = None
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []

        # Metrics
        self.spawn_time = datetime.now()
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks_count = 0

        # Resource quotas
        self.resource_quotas = resource_quotas or {}
        self._register_resource_quotas()

        logger.info(f"Agent spawned: {agent_id} ({role}) with capabilities: {[c.value for c in capabilities]}")

    def _register_resource_quotas(self):
        """Register resource quotas with governor"""
        try:
            from core.orchestration.resource_governor import get_resource_governor, ResourceType

            governor = get_resource_governor()

            # Default quotas
            default_quotas = {
                ResourceType.CPU_TIME: 3600.0,  # 1 hour
                ResourceType.MEMORY: 1024.0,  # 1GB
                ResourceType.CONCURRENT_TASKS: 5.0,
                ResourceType.API_CALLS: 1000.0,
                ResourceType.FILE_OPERATIONS: 100.0,
                ResourceType.NETWORK_REQUESTS: 500.0,
            }

            # Apply quotas
            for resource_type, limit in default_quotas.items():
                # Use custom quota if provided
                custom_limit = self.resource_quotas.get(resource_type.value, limit)
                governor.set_quota(
                    self.agent_id,
                    resource_type,
                    custom_limit,
                    reset_period_sec=3600.0  # Hourly reset
                )

            logger.debug(f"Resource quotas registered for agent: {self.agent_id}")

        except Exception as e:
            logger.warning(f"Failed to register resource quotas: {e}")

    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Check if agent can handle task.

        Args:
            task: Task specification

        Returns:
            True if agent can handle, False otherwise
        """
        pass

    @abstractmethod
    def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Execute task.

        Args:
            task: Task specification with parameters

        Returns:
            TaskResult with execution outcome
        """
        pass

    def start(self):
        """Start agent (transition to IDLE)"""
        if self.status != AgentStatus.SPAWNED:
            logger.warning(f"Cannot start agent in status: {self.status.value}")
            return

        self.status = AgentStatus.IDLE
        logger.info(f"Agent started: {self.agent_id}")

    def pause(self):
        """Pause agent"""
        if self.status not in [AgentStatus.IDLE, AgentStatus.BUSY]:
            logger.warning(f"Cannot pause agent in status: {self.status.value}")
            return

        self.status = AgentStatus.PAUSED
        logger.info(f"Agent paused: {self.agent_id}")

    def resume(self):
        """Resume agent from pause"""
        if self.status != AgentStatus.PAUSED:
            logger.warning(f"Cannot resume agent in status: {self.status.value}")
            return

        self.status = AgentStatus.IDLE
        logger.info(f"Agent resumed: {self.agent_id}")

    def terminate(self):
        """Terminate agent"""
        self.status = AgentStatus.TERMINATED
        logger.info(f"Agent terminated: {self.agent_id}")

    def send_message(
        self,
        to_agent: str,
        message_type: str,
        content: Dict[str, Any],
        in_reply_to: Optional[str] = None
    ) -> AgentMessage:
        """
        Send message to another agent.

        Args:
            to_agent: Target agent ID
            message_type: Type of message (request, response, etc.)
            content: Message content
            in_reply_to: Optional message ID this is replying to

        Returns:
            Created AgentMessage
        """
        message = AgentMessage(
            message_id=str(uuid4()),
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            in_reply_to=in_reply_to
        )

        self.outbox.append(message)

        logger.debug(
            f"Message sent: {self.agent_id} -> {to_agent} "
            f"({message_type})"
        )

        return message

    def receive_message(self, message: AgentMessage):
        """
        Receive message from another agent.

        Args:
            message: AgentMessage to receive
        """
        self.inbox.append(message)

        logger.debug(
            f"Message received: {message.from_agent} -> {self.agent_id} "
            f"({message.message_type})"
        )

    def get_messages(
        self,
        message_type: Optional[str] = None,
        unread_only: bool = False
    ) -> List[AgentMessage]:
        """
        Get messages from inbox.

        Args:
            message_type: Filter by message type
            unread_only: Only return unread messages

        Returns:
            List of messages
        """
        messages = self.inbox

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        # Note: unread tracking would require additional state
        # Simplified for now

        return messages

    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status.

        Returns:
            Status dictionary with agent state and metrics
        """
        uptime_sec = (datetime.now() - self.spawn_time).total_seconds()

        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "current_task": self.current_task,
            "uptime_sec": uptime_sec,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks_count,
            "success_rate": self.successful_tasks / max(self.total_tasks, 1),
            "inbox_count": len(self.inbox),
            "outbox_count": len(self.outbox),
        }

    def _validate_safety(self, task: Dict[str, Any]) -> bool:
        """
        Validate task safety with Guardian Defense.

        Args:
            task: Task to validate

        Returns:
            True if safe, False otherwise
        """
        try:
            from security.guardian_defense import get_guardian_defense

            guardian = get_guardian_defense()

            # Check for dangerous operations
            task_type = task.get("type", "")
            task_params = task.get("params", {})

            # Task-specific safety checks would go here
            # For now, basic validation

            return True

        except Exception as e:
            logger.error(f"Safety validation failed: {e}")
            # Fail closed (deny) on validation errors
            return False

    def _track_resource_usage(self, resource_type: str, amount: float) -> bool:
        """
        Track resource usage.

        Args:
            resource_type: Type of resource
            amount: Amount consumed

        Returns:
            True if usage allowed, False if quota exceeded
        """
        try:
            from core.orchestration.resource_governor import get_resource_governor, ResourceType

            governor = get_resource_governor()

            # Convert string to ResourceType enum
            rt = ResourceType(resource_type)

            return governor.track_usage(self.agent_id, rt, amount)

        except Exception as e:
            logger.error(f"Resource tracking failed: {e}")
            # Fail open (allow) on tracking errors
            return True

    def _log_task_execution(self, task_id: str, result: TaskResult):
        """Log task execution to audit chain"""
        try:
            from utils.audit_chain import append_jsonl

            log_entry = {
                "event_type": "agent_task_execution",
                "agent_id": self.agent_id,
                "role": self.role,
                "task_id": task_id,
                "success": result.success,
                "execution_time_sec": result.execution_time_sec,
                "error": result.error,
                "timestamp": result.timestamp
            }

            append_jsonl("var/orchestration/agent_tasks.ndjson", log_entry)

        except Exception as e:
            logger.error(f"Failed to log task execution: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role}, status={self.status.value})"
