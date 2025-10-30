"""
Collaboration Engine - Cooperative Behavior and Teamwork

This module enables AGI to collaborate effectively with other agents,
including cooperative planning, communication, and conflict resolution.

Key capabilities:
- Cooperative task planning and execution
- Communication strategy selection
- Conflict detection and resolution
- Resource sharing and allocation
- Team coordination
- Role assignment and adaptation
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json


class TaskStatus(str, Enum):
    """Status of collaborative task"""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CommunicationType(str, Enum):
    """Types of communication"""
    REQUEST = "request"  # Ask for something
    INFORM = "inform"  # Share information
    PROPOSE = "propose"  # Suggest action
    QUERY = "query"  # Ask question
    AGREE = "agree"  # Accept proposal
    REFUSE = "refuse"  # Reject proposal
    ACKNOWLEDGE = "acknowledge"  # Confirm receipt
    NEGOTIATE = "negotiate"  # Discuss terms


class ConflictType(str, Enum):
    """Types of conflicts"""
    RESOURCE = "resource"  # Competition for resources
    GOAL = "goal"  # Incompatible goals
    METHOD = "method"  # Disagreement on approach
    PRIORITY = "priority"  # Different priorities
    BELIEF = "belief"  # Different beliefs about facts
    VALUE = "value"  # Different values/preferences


class ResolutionStrategy(str, Enum):
    """Conflict resolution strategies"""
    COMPROMISE = "compromise"  # Meet in middle
    COLLABORATION = "collaboration"  # Find win-win
    ACCOMMODATION = "accommodation"  # Yield to other
    COMPETITION = "competition"  # Assert own position
    AVOIDANCE = "avoidance"  # Postpone or ignore


@dataclass
class CommunicationStrategy:
    """Strategy for communication"""
    message_type: CommunicationType
    content: str
    recipient: str
    priority: float = 0.5  # 0-1
    urgency: float = 0.5  # 0-1
    formality: float = 0.5  # 0-1
    expected_response: Optional[str] = None
    timeout: float = 60.0  # seconds
    retries: int = 3


@dataclass
class Message:
    """A communication message"""
    message_id: str
    sender: str
    recipient: str
    message_type: CommunicationType
    content: str
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborativeTask:
    """A task requiring collaboration"""
    task_id: str
    description: str
    goal: str
    participants: List[str]
    role_assignments: Dict[str, str] = field(default_factory=dict)  # participant -> role
    status: TaskStatus = TaskStatus.PROPOSED
    subtasks: List[str] = field(default_factory=list)  # Subtask IDs
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # task -> [dependencies]
    resource_allocation: Dict[str, Dict[str, float]] = field(default_factory=dict)  # participant -> {resource: amount}
    progress: float = 0.0  # 0-1
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    coordinator: Optional[str] = None

    def assign_role(self, participant: str, role: str):
        """Assign role to participant"""
        self.role_assignments[participant] = role

    def update_progress(self, new_progress: float):
        """Update task progress"""
        self.progress = max(0.0, min(1.0, new_progress))
        if self.progress >= 1.0:
            self.status = TaskStatus.COMPLETED
            self.completed_at = time.time()


@dataclass
class ConflictResolution:
    """Resolution of a conflict"""
    conflict_id: str
    conflict_type: ConflictType
    parties: List[str]  # Agents in conflict
    description: str
    strategy: ResolutionStrategy
    proposed_solution: str
    accepted: bool = False
    outcome: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SharedResource:
    """A resource shared among agents"""
    resource_id: str
    name: str
    total_amount: float
    allocated: Dict[str, float] = field(default_factory=dict)  # agent -> amount
    requests: List[Tuple[str, float, float]] = field(default_factory=list)  # (agent, amount, priority)

    def available(self) -> float:
        """Get available amount"""
        allocated_total = sum(self.allocated.values())
        return max(0.0, self.total_amount - allocated_total)

    def allocate(self, agent: str, amount: float) -> bool:
        """Allocate resource to agent"""
        if amount <= self.available():
            self.allocated[agent] = self.allocated.get(agent, 0) + amount
            return True
        return False

    def release(self, agent: str, amount: Optional[float] = None):
        """Release allocated resource"""
        if agent in self.allocated:
            if amount is None:
                # Release all
                del self.allocated[agent]
            else:
                # Release partial
                self.allocated[agent] = max(0, self.allocated[agent] - amount)
                if self.allocated[agent] == 0:
                    del self.allocated[agent]


class CollaborationEngine:
    """
    Collaboration engine for multi-agent cooperation

    Features:
    - Cooperative task planning and coordination
    - Strategic communication
    - Conflict detection and resolution
    - Resource sharing and allocation
    - Role assignment and management
    - Team performance tracking
    """

    def __init__(self):
        self.tasks: Dict[str, CollaborativeTask] = {}
        self.messages: List[Message] = []
        self.conflicts: Dict[str, ConflictResolution] = {}
        self.resources: Dict[str, SharedResource] = {}
        self.task_counter = 0
        self.message_counter = 0
        self.conflict_counter = 0

        # Collaboration metrics
        self.collaboration_history: List[Dict[str, Any]] = []
        self.agent_reliability: Dict[str, float] = {}  # agent -> reliability score

    def create_collaborative_task(
        self,
        description: str,
        goal: str,
        participants: List[str],
        coordinator: Optional[str] = None
    ) -> CollaborativeTask:
        """
        Create a new collaborative task

        Args:
            description: What needs to be done
            goal: Desired outcome
            participants: Agents involved
            coordinator: Optional coordinator agent

        Returns:
            Created CollaborativeTask
        """
        self.task_counter += 1
        task_id = f"collab_task_{self.task_counter}"

        task = CollaborativeTask(
            task_id=task_id,
            description=description,
            goal=goal,
            participants=participants,
            coordinator=coordinator or participants[0] if participants else None
        )

        self.tasks[task_id] = task
        return task

    def decompose_task(
        self,
        task: CollaborativeTask,
        strategy: str = "balanced"
    ) -> List[CollaborativeTask]:
        """
        Decompose task into subtasks for participants

        Args:
            task: Task to decompose
            strategy: Decomposition strategy ("balanced", "specialized", "parallel")

        Returns:
            List of subtasks
        """
        subtasks = []

        if strategy == "balanced":
            # Divide work equally among participants
            num_parts = len(task.participants)
            for i, participant in enumerate(task.participants):
                self.task_counter += 1
                subtask_id = f"subtask_{self.task_counter}"

                subtask = CollaborativeTask(
                    task_id=subtask_id,
                    description=f"Part {i+1}/{num_parts} of {task.description}",
                    goal=f"Complete portion {i+1} of {task.goal}",
                    participants=[participant],
                    coordinator=task.coordinator
                )

                subtask.assign_role(participant, "executor")
                subtasks.append(subtask)
                task.subtasks.append(subtask_id)
                self.tasks[subtask_id] = subtask

        elif strategy == "specialized":
            # Assign based on capabilities (simplified)
            # In production, would use agent capability models
            roles = ["planner", "executor", "validator", "integrator"]
            for i, participant in enumerate(task.participants):
                role = roles[i % len(roles)]

                self.task_counter += 1
                subtask_id = f"subtask_{self.task_counter}"

                subtask = CollaborativeTask(
                    task_id=subtask_id,
                    description=f"{role.capitalize()} role for {task.description}",
                    goal=f"Fulfill {role} responsibilities",
                    participants=[participant],
                    coordinator=task.coordinator
                )

                subtask.assign_role(participant, role)
                subtasks.append(subtask)
                task.subtasks.append(subtask_id)
                self.tasks[subtask_id] = subtask

        elif strategy == "parallel":
            # All participants work on same task simultaneously
            for participant in task.participants:
                self.task_counter += 1
                subtask_id = f"subtask_{self.task_counter}"

                subtask = CollaborativeTask(
                    task_id=subtask_id,
                    description=f"Parallel work on {task.description}",
                    goal=task.goal,
                    participants=[participant],
                    coordinator=task.coordinator
                )

                subtask.assign_role(participant, "contributor")
                subtasks.append(subtask)
                task.subtasks.append(subtask_id)
                self.tasks[subtask_id] = subtask

        return subtasks

    def plan_communication(
        self,
        sender: str,
        recipient: str,
        purpose: str,
        context: Dict[str, Any]
    ) -> CommunicationStrategy:
        """
        Plan communication strategy

        Args:
            sender: Who is sending
            recipient: Who will receive
            purpose: Purpose of communication
            context: Contextual information

        Returns:
            Communication strategy
        """
        # Determine message type based on purpose
        purpose_lower = purpose.lower()

        if any(word in purpose_lower for word in ["ask", "request", "need"]):
            msg_type = CommunicationType.REQUEST
            priority = 0.7
        elif any(word in purpose_lower for word in ["tell", "inform", "notify"]):
            msg_type = CommunicationType.INFORM
            priority = 0.5
        elif any(word in purpose_lower for word in ["suggest", "propose"]):
            msg_type = CommunicationType.PROPOSE
            priority = 0.6
        elif any(word in purpose_lower for word in ["question", "query", "ask about"]):
            msg_type = CommunicationType.QUERY
            priority = 0.5
        else:
            msg_type = CommunicationType.INFORM
            priority = 0.5

        # Determine formality based on context
        formality = context.get("formality", 0.5)

        # Determine urgency
        urgency = context.get("urgency", 0.5)

        strategy = CommunicationStrategy(
            message_type=msg_type,
            content=purpose,
            recipient=recipient,
            priority=priority,
            urgency=urgency,
            formality=formality
        )

        return strategy

    def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: CommunicationType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Send a message to another agent"""
        self.message_counter += 1
        message_id = f"msg_{self.message_counter}"

        message = Message(
            message_id=message_id,
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )

        self.messages.append(message)

        # Keep only recent messages
        if len(self.messages) > 1000:
            self.messages = self.messages[-1000:]

        return message

    def respond_to_message(
        self,
        message: Message,
        response_content: str,
        accept: bool = True
    ) -> Message:
        """Respond to a received message"""
        # Determine response type
        if message.message_type == CommunicationType.REQUEST:
            response_type = CommunicationType.AGREE if accept else CommunicationType.REFUSE
        elif message.message_type == CommunicationType.PROPOSE:
            response_type = CommunicationType.AGREE if accept else CommunicationType.REFUSE
        elif message.message_type == CommunicationType.QUERY:
            response_type = CommunicationType.INFORM
        else:
            response_type = CommunicationType.ACKNOWLEDGE

        # Create response
        response = self.send_message(
            sender=message.recipient,
            recipient=message.sender,
            message_type=response_type,
            content=response_content,
            metadata={"in_response_to": message.message_id}
        )

        # Mark original message as delivered with response
        message.delivered = True
        message.response = response.message_id

        return response

    def detect_conflict(
        self,
        agent1: str,
        agent2: str,
        context: Dict[str, Any]
    ) -> Optional[ConflictResolution]:
        """
        Detect potential conflict between agents

        Args:
            agent1: First agent
            agent2: Second agent
            context: Context information

        Returns:
            ConflictResolution if conflict detected, None otherwise
        """
        conflict_type = None
        description = ""

        # Check for resource conflicts
        if "resource" in context:
            resource_id = context["resource"]
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                # Check if both agents want more than available
                agent1_request = context.get(f"{agent1}_request", 0)
                agent2_request = context.get(f"{agent2}_request", 0)

                if agent1_request + agent2_request > resource.available():
                    conflict_type = ConflictType.RESOURCE
                    description = f"Both agents want {resource.name} but insufficient available"

        # Check for goal conflicts
        if "goals" in context:
            agent1_goal = context["goals"].get(agent1)
            agent2_goal = context["goals"].get(agent2)

            if agent1_goal and agent2_goal:
                # Simplified check - in production would use semantic comparison
                if "compete" in agent1_goal.lower() or "compete" in agent2_goal.lower():
                    conflict_type = ConflictType.GOAL
                    description = "Agents have competing goals"

        # Check for method disagreements
        if "proposed_methods" in context:
            methods = context["proposed_methods"]
            if len(set(methods.values())) > 1:  # Different methods proposed
                conflict_type = ConflictType.METHOD
                description = "Agents disagree on approach"

        if conflict_type:
            self.conflict_counter += 1
            conflict_id = f"conflict_{self.conflict_counter}"

            # Determine resolution strategy
            strategy = self._select_resolution_strategy(conflict_type, context)

            conflict = ConflictResolution(
                conflict_id=conflict_id,
                conflict_type=conflict_type,
                parties=[agent1, agent2],
                description=description,
                strategy=strategy,
                proposed_solution=""  # Will be filled by resolution process
            )

            self.conflicts[conflict_id] = conflict
            return conflict

        return None

    def _select_resolution_strategy(
        self,
        conflict_type: ConflictType,
        context: Dict[str, Any]
    ) -> ResolutionStrategy:
        """Select appropriate resolution strategy"""
        # Strategy selection based on conflict type
        if conflict_type == ConflictType.RESOURCE:
            # Try to find collaborative solution first
            return ResolutionStrategy.COLLABORATION

        elif conflict_type == ConflictType.GOAL:
            # Look for compromise
            return ResolutionStrategy.COMPROMISE

        elif conflict_type == ConflictType.METHOD:
            # Collaborate to find best approach
            return ResolutionStrategy.COLLABORATION

        elif conflict_type == ConflictType.PRIORITY:
            # Negotiate priorities
            return ResolutionStrategy.COMPROMISE

        else:
            # Default to compromise
            return ResolutionStrategy.COMPROMISE

    def resolve_conflict(
        self,
        conflict: ConflictResolution
    ) -> bool:
        """
        Resolve a conflict

        Args:
            conflict: Conflict to resolve

        Returns:
            True if resolved successfully
        """
        if conflict.strategy == ResolutionStrategy.COMPROMISE:
            # Find middle ground
            conflict.proposed_solution = f"Compromise: Each party adjusts expectations"

        elif conflict.strategy == ResolutionStrategy.COLLABORATION:
            # Find win-win solution
            if conflict.conflict_type == ConflictType.RESOURCE:
                conflict.proposed_solution = "Increase available resources or share in turns"
            else:
                conflict.proposed_solution = "Combine approaches to satisfy both parties"

        elif conflict.strategy == ResolutionStrategy.ACCOMMODATION:
            # One party yields
            conflict.proposed_solution = f"{conflict.parties[1]} accommodates {conflict.parties[0]}"

        elif conflict.strategy == ResolutionStrategy.COMPETITION:
            # Assert position (typically used as last resort)
            conflict.proposed_solution = f"{conflict.parties[0]} proceeds with original plan"

        elif conflict.strategy == ResolutionStrategy.AVOIDANCE:
            # Postpone decision
            conflict.proposed_solution = "Postpone decision until more information available"

        # In production, would negotiate with parties
        # For now, assume 70% acceptance rate
        import random
        conflict.accepted = random.random() < 0.7

        if conflict.accepted:
            conflict.outcome = "resolved"
            return True
        else:
            conflict.outcome = "unresolved"
            return False

    def allocate_resources(
        self,
        task: CollaborativeTask,
        available_resources: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Allocate resources to task participants

        Args:
            task: Task requiring resources
            available_resources: Available resources

        Returns:
            Allocation map: participant -> {resource: amount}
        """
        allocation = {}

        # Equal distribution by default
        num_participants = len(task.participants)

        for resource_name, total_amount in available_resources.items():
            # Create or get shared resource
            if resource_name not in self.resources:
                self.resources[resource_name] = SharedResource(
                    resource_id=resource_name,
                    name=resource_name,
                    total_amount=total_amount
                )

            resource = self.resources[resource_name]
            amount_per_participant = total_amount / num_participants

            for participant in task.participants:
                if resource.allocate(participant, amount_per_participant):
                    if participant not in allocation:
                        allocation[participant] = {}
                    allocation[participant][resource_name] = amount_per_participant

        task.resource_allocation = allocation
        return allocation

    def coordinate_execution(
        self,
        task: CollaborativeTask
    ) -> Dict[str, Any]:
        """
        Coordinate execution of collaborative task

        Returns status and next actions
        """
        if task.status == TaskStatus.PROPOSED:
            # Start task
            task.status = TaskStatus.ACCEPTED
            task.started_at = time.time()

            # Send start messages to participants
            for participant in task.participants:
                self.send_message(
                    sender=task.coordinator or "system",
                    recipient=participant,
                    message_type=CommunicationType.INFORM,
                    content=f"Task started: {task.description}",
                    metadata={"task_id": task.task_id}
                )

            return {
                "status": "started",
                "message": "Task initiated, participants notified"
            }

        elif task.status == TaskStatus.IN_PROGRESS:
            # Check progress of subtasks
            if task.subtasks:
                completed_subtasks = sum(
                    1 for st_id in task.subtasks
                    if st_id in self.tasks and self.tasks[st_id].status == TaskStatus.COMPLETED
                )
                task.update_progress(completed_subtasks / len(task.subtasks))

            # Check if blocked
            # (simplified - would check dependencies, resources, etc.)

            return {
                "status": "in_progress",
                "progress": task.progress,
                "message": f"Task {int(task.progress * 100)}% complete"
            }

        elif task.status == TaskStatus.COMPLETED:
            # Record collaboration history
            self._record_collaboration(task)

            return {
                "status": "completed",
                "message": "Task successfully completed"
            }

        return {"status": task.status.value}

    def _record_collaboration(self, task: CollaborativeTask):
        """Record collaboration for learning"""
        duration = (task.completed_at or time.time()) - task.created_at

        record = {
            "task_id": task.task_id,
            "participants": task.participants,
            "duration": duration,
            "success": task.status == TaskStatus.COMPLETED,
            "timestamp": time.time()
        }

        self.collaboration_history.append(record)

        # Update reliability scores
        for participant in task.participants:
            if participant not in self.agent_reliability:
                self.agent_reliability[participant] = 0.5

            if task.status == TaskStatus.COMPLETED:
                # Increase reliability
                self.agent_reliability[participant] = min(
                    1.0,
                    self.agent_reliability[participant] + 0.1
                )
            else:
                # Decrease reliability slightly
                self.agent_reliability[participant] = max(
                    0.0,
                    self.agent_reliability[participant] - 0.05
                )

    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get collaboration performance metrics"""
        total_tasks = len(self.tasks)
        completed_tasks = sum(
            1 for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED
        )
        failed_tasks = sum(
            1 for t in self.tasks.values()
            if t.status == TaskStatus.FAILED
        )

        total_conflicts = len(self.conflicts)
        resolved_conflicts = sum(
            1 for c in self.conflicts.values()
            if c.accepted
        )

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "total_messages": len(self.messages),
            "total_conflicts": total_conflicts,
            "resolved_conflicts": resolved_conflicts,
            "resolution_rate": resolved_conflicts / total_conflicts if total_conflicts > 0 else 0,
            "agent_reliability": self.agent_reliability.copy(),
            "active_resources": len(self.resources),
        }
