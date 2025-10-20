"""
Training Data Collector - Capture real decisions and outcomes for AGI training

This system automatically collects training data from ShivX operations:
- Every decision made (tool selection, workflow planning, etc.)
- Task outcomes (success/failure)
- User feedback and corrections
- Empire-specific operations (Sewago, Halobuzz, SolsniperPro)

This data feeds the RL policies, continual learner, and transfer learner.

Part of ShivX Personal Empire AGI (Week 1).
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid


logger = logging.getLogger(__name__)


class TaskDomain(Enum):
    """Task domains matching empire businesses"""
    SEWAGO = "sewago"  # SewaAI - Core platform
    HALOBUZZ = "halobuzz"  # HaloAI - Marketing/social
    SOLSNIPER = "solsniper"  # Aayan AI - Trading/crypto
    NEPVEST = "nepvest"  # Future expansion
    SHIVX_CORE = "shivx_core"  # General ShivX operations
    UNKNOWN = "unknown"


class TaskType(Enum):
    """Types of tasks"""
    TOOL_SELECTION = "tool_selection"
    WORKFLOW_PLANNING = "workflow_planning"
    CODE_GENERATION = "code_generation"
    BUG_FIXING = "bug_fixing"
    DECISION_MAKING = "decision_making"
    CONTENT_CREATION = "content_creation"
    TRADING_DECISION = "trading_decision"
    USER_INTERACTION = "user_interaction"
    SYSTEM_OPTIMIZATION = "system_optimization"


@dataclass
class TaskExample:
    """A single training example from real usage"""
    id: str
    domain: TaskDomain
    task_type: TaskType

    # Input
    context: Dict[str, Any]  # What was the situation?
    query: str  # What was the user asking for?

    # Action taken
    action_taken: str  # What did we do?
    reasoning: str  # Why did we do it?
    alternatives_considered: List[str] = field(default_factory=list)

    # Outcome
    outcome: Optional[str] = None  # What happened?
    success: Optional[bool] = None  # Did it work?
    user_feedback: Optional[str] = None  # What did user say?

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    duration_seconds: Optional[float] = None
    cost_estimate: Optional[float] = None  # For ROI tracking

    # For RL training
    reward: Optional[float] = None  # Computed reward signal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['domain'] = self.domain.value
        data['task_type'] = self.task_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskExample':
        """Create from dictionary"""
        data = data.copy()
        data['domain'] = TaskDomain(data['domain'])
        data['task_type'] = TaskType(data['task_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Dataset:
    """A versioned dataset of training examples"""
    name: str
    version: str
    examples: List[TaskExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_example(self, example: TaskExample):
        """Add example to dataset"""
        self.examples.append(example)

    def filter_by_domain(self, domain: TaskDomain) -> List[TaskExample]:
        """Get examples for specific domain"""
        return [ex for ex in self.examples if ex.domain == domain]

    def filter_by_type(self, task_type: TaskType) -> List[TaskExample]:
        """Get examples for specific task type"""
        return [ex for ex in self.examples if ex.task_type == task_type]

    def get_successful(self) -> List[TaskExample]:
        """Get successful examples only"""
        return [ex for ex in self.examples if ex.success is True]

    def get_failed(self) -> List[TaskExample]:
        """Get failed examples"""
        return [ex for ex in self.examples if ex.success is False]

    def compute_stats(self) -> Dict[str, Any]:
        """Compute dataset statistics"""
        total = len(self.examples)
        if total == 0:
            return {"total": 0}

        successful = len(self.get_successful())
        failed = len(self.get_failed())
        unlabeled = total - successful - failed

        # Domain distribution
        domain_counts = {}
        for domain in TaskDomain:
            count = len(self.filter_by_domain(domain))
            if count > 0:
                domain_counts[domain.value] = count

        # Task type distribution
        type_counts = {}
        for task_type in TaskType:
            count = len(self.filter_by_type(task_type))
            if count > 0:
                type_counts[task_type.value] = count

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "unlabeled": unlabeled,
            "success_rate": successful / total if total > 0 else 0,
            "domains": domain_counts,
            "task_types": type_counts,
        }


class DataCollector:
    """
    Automatic data collection from ShivX operations.

    Usage:
        collector = DataCollector()

        # Start task
        task_id = collector.start_task(
            domain=TaskDomain.SEWAGO,
            task_type=TaskType.CODE_GENERATION,
            context={...},
            query="Build user authentication"
        )

        # Record action
        collector.record_action(
            task_id,
            action="Generated auth module",
            reasoning="User needs secure login"
        )

        # Complete task
        collector.complete_task(
            task_id,
            outcome="Auth module deployed",
            success=True
        )
    """

    def __init__(
        self,
        storage_dir: str = "data/agi_training",
        auto_save: bool = True,
    ):
        """
        Initialize Data Collector.

        Args:
            storage_dir: Directory for storing collected data
            auto_save: Auto-save after each task completion
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.auto_save = auto_save

        # Active tasks (task_id -> partial TaskExample)
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        # Current dataset
        self.current_dataset = Dataset(
            name="shivx_live_training",
            version=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        )

        # Load existing data if available
        self._load_latest_dataset()

        logger.info(f"Data Collector initialized: {len(self.current_dataset.examples)} examples loaded")

    def start_task(
        self,
        domain: TaskDomain,
        task_type: TaskType,
        context: Dict[str, Any],
        query: str,
        task_id: Optional[str] = None,
    ) -> str:
        """
        Start tracking a new task.

        Args:
            domain: Task domain
            task_type: Task type
            context: Contextual information
            query: User query/request
            task_id: Optional task ID (generated if not provided)

        Returns:
            Task ID
        """
        if task_id is None:
            task_id = str(uuid.uuid4())

        self.active_tasks[task_id] = {
            "id": task_id,
            "domain": domain,
            "task_type": task_type,
            "context": context,
            "query": query,
            "start_time": datetime.utcnow(),
        }

        logger.debug(f"Started task: {task_id} ({domain.value}/{task_type.value})")

        return task_id

    def record_action(
        self,
        task_id: str,
        action_taken: str,
        reasoning: str,
        alternatives: Optional[List[str]] = None,
        confidence: float = 1.0,
    ):
        """Record action taken for a task"""
        if task_id not in self.active_tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        self.active_tasks[task_id].update({
            "action_taken": action_taken,
            "reasoning": reasoning,
            "alternatives_considered": alternatives or [],
            "confidence": confidence,
        })

        logger.debug(f"Recorded action for task: {task_id}")

    def complete_task(
        self,
        task_id: str,
        outcome: str,
        success: bool,
        user_feedback: Optional[str] = None,
        reward: Optional[float] = None,
    ):
        """
        Mark task as complete and store example.

        Args:
            task_id: Task ID
            outcome: What happened
            success: Whether task succeeded
            user_feedback: User's feedback
            reward: Computed reward (for RL)
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task not found: {task_id}")
            return

        task_data = self.active_tasks[task_id]

        # Compute duration
        start_time = task_data.get("start_time")
        if start_time:
            duration = (datetime.utcnow() - start_time).total_seconds()
        else:
            duration = None

        # Compute reward if not provided
        if reward is None:
            reward = self._compute_reward(success, duration, task_data.get("confidence", 1.0))

        # Create TaskExample
        example = TaskExample(
            id=task_id,
            domain=task_data["domain"],
            task_type=task_data["task_type"],
            context=task_data["context"],
            query=task_data["query"],
            action_taken=task_data.get("action_taken", "unknown"),
            reasoning=task_data.get("reasoning", ""),
            alternatives_considered=task_data.get("alternatives_considered", []),
            outcome=outcome,
            success=success,
            user_feedback=user_feedback,
            confidence=task_data.get("confidence", 1.0),
            duration_seconds=duration,
            reward=reward,
        )

        # Add to dataset
        self.current_dataset.add_example(example)

        # Remove from active tasks
        del self.active_tasks[task_id]

        # Auto-save if enabled
        if self.auto_save:
            self.save_dataset()

        logger.info(
            f"Completed task: {task_id} (success={success}, reward={reward:.2f})"
        )

    def _compute_reward(
        self,
        success: bool,
        duration: Optional[float],
        confidence: float,
    ) -> float:
        """
        Compute reward signal for RL training.

        Reward formula:
        - Base: +1 for success, -1 for failure
        - Speed bonus: +0.5 if fast (<60s), -0.2 if slow (>300s)
        - Confidence bonus: +0.3 if high confidence (>0.8) and correct
        """
        reward = 1.0 if success else -1.0

        # Speed bonus/penalty
        if duration is not None:
            if duration < 60:
                reward += 0.5
            elif duration > 300:
                reward -= 0.2

        # Confidence bonus (only if successful)
        if success and confidence > 0.8:
            reward += 0.3

        # Confidence penalty (if overconfident and failed)
        if not success and confidence > 0.8:
            reward -= 0.3

        return reward

    def save_dataset(self, filename: Optional[str] = None):
        """Save current dataset to disk"""
        if filename is None:
            filename = f"dataset_{self.current_dataset.version}.json"

        filepath = self.storage_dir / filename

        data = {
            "name": self.current_dataset.name,
            "version": self.current_dataset.version,
            "created_at": self.current_dataset.created_at.isoformat(),
            "metadata": self.current_dataset.metadata,
            "examples": [ex.to_dict() for ex in self.current_dataset.examples],
            "stats": self.current_dataset.compute_stats(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved dataset: {filepath} ({len(self.current_dataset.examples)} examples)")

    def _load_latest_dataset(self):
        """Load most recent dataset from disk"""
        try:
            # Find latest dataset file
            dataset_files = sorted(self.storage_dir.glob("dataset_*.json"), reverse=True)

            if not dataset_files:
                logger.info("No existing dataset found")
                return

            latest_file = dataset_files[0]

            with open(latest_file, 'r') as f:
                data = json.load(f)

            # Load examples
            examples = [TaskExample.from_dict(ex) for ex in data["examples"]]

            self.current_dataset = Dataset(
                name=data["name"],
                version=data["version"],
                examples=examples,
                metadata=data.get("metadata", {}),
                created_at=datetime.fromisoformat(data["created_at"]),
            )

            logger.info(f"Loaded dataset: {latest_file.name} ({len(examples)} examples)")

        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            "active_tasks": len(self.active_tasks),
            "dataset": self.current_dataset.compute_stats(),
            "storage_dir": str(self.storage_dir),
        }

    def export_for_training(
        self,
        domain: Optional[TaskDomain] = None,
        task_type: Optional[TaskType] = None,
        success_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Export examples for training.

        Args:
            domain: Filter by domain
            task_type: Filter by task type
            success_only: Only include successful examples

        Returns:
            List of training examples
        """
        examples = self.current_dataset.examples

        if domain:
            examples = [ex for ex in examples if ex.domain == domain]

        if task_type:
            examples = [ex for ex in examples if ex.task_type == task_type]

        if success_only:
            examples = [ex for ex in examples if ex.success is True]

        return [
            {
                "input": {
                    "context": ex.context,
                    "query": ex.query,
                },
                "output": ex.action_taken,
                "reward": ex.reward or 0.0,
                "success": ex.success,
                "metadata": {
                    "domain": ex.domain.value,
                    "task_type": ex.task_type.value,
                    "reasoning": ex.reasoning,
                }
            }
            for ex in examples
        ]


# Decorator for automatic data collection
def collect_task_data(
    domain: TaskDomain,
    task_type: TaskType,
):
    """
    Decorator to automatically collect data from function execution.

    Usage:
        @collect_task_data(TaskDomain.SEWAGO, TaskType.CODE_GENERATION)
        def generate_code(query: str, context: Dict) -> str:
            # ... implementation
            return result
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_collector()

            # Extract context and query from args/kwargs
            context = kwargs.get('context', {})
            query = kwargs.get('query', str(args[0]) if args else "")

            # Start task
            task_id = collector.start_task(domain, task_type, context, query)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record success
                collector.record_action(
                    task_id,
                    action_taken=str(result),
                    reasoning=f"Executed {func.__name__}",
                )

                collector.complete_task(
                    task_id,
                    outcome=f"Function executed successfully",
                    success=True,
                )

                return result

            except Exception as e:
                # Record failure
                collector.complete_task(
                    task_id,
                    outcome=f"Function failed: {str(e)}",
                    success=False,
                )
                raise

        return wrapper
    return decorator


# Global singleton
_collector = None


def get_collector() -> DataCollector:
    """Get or create global data collector"""
    global _collector

    if _collector is None:
        _collector = DataCollector()

    return _collector


# Quick test
def test_data_collector():
    """Test data collection"""
    collector = DataCollector(storage_dir="data/agi_training/test")

    # Simulate some tasks
    print("\n=== Data Collector Test ===\n")

    # Task 1: Successful Sewago operation
    task1 = collector.start_task(
        domain=TaskDomain.SEWAGO,
        task_type=TaskType.CODE_GENERATION,
        context={"user_id": "123", "project": "auth_system"},
        query="Create user authentication module",
    )

    collector.record_action(
        task1,
        action_taken="Generated auth module with JWT tokens",
        reasoning="JWT is secure and widely supported",
        alternatives=["Session-based auth", "OAuth only"],
        confidence=0.9,
    )

    collector.complete_task(
        task1,
        outcome="Auth module deployed and tested",
        success=True,
        user_feedback="Works great!",
    )

    # Task 2: Failed trading decision
    task2 = collector.start_task(
        domain=TaskDomain.SOLSNIPER,
        task_type=TaskType.TRADING_DECISION,
        context={"market": "SOL/USDT", "price": 150.0},
        query="Should I buy SOL now?",
    )

    collector.record_action(
        task2,
        action_taken="Recommended buy",
        reasoning="Technical indicators show upward trend",
        confidence=0.7,
    )

    collector.complete_task(
        task2,
        outcome="Price dropped 5%",
        success=False,
        user_feedback="Bad timing",
    )

    # Task 3: Content creation
    task3 = collector.start_task(
        domain=TaskDomain.HALOBUZZ,
        task_type=TaskType.CONTENT_CREATION,
        context={"platform": "twitter", "audience": "tech_professionals"},
        query="Create engaging tweet about AI",
    )

    collector.record_action(
        task3,
        action_taken="Generated tweet about AGI progress",
        reasoning="Trending topic, relevant to audience",
        confidence=0.85,
    )

    collector.complete_task(
        task3,
        outcome="Tweet got 500+ engagements",
        success=True,
    )

    # Show stats
    stats = collector.get_stats()
    print(f"Collected {stats['dataset']['total']} examples")
    print(f"Success rate: {stats['dataset']['success_rate']:.1%}")
    print(f"\nDomain distribution:")
    for domain, count in stats['dataset']['domains'].items():
        print(f"  {domain}: {count}")

    print(f"\nTask type distribution:")
    for task_type, count in stats['dataset']['task_types'].items():
        print(f"  {task_type}: {count}")

    # Export for training
    training_data = collector.export_for_training(success_only=True)
    print(f"\n{len(training_data)} successful examples ready for training")

    # Save
    collector.save_dataset()

    return collector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_data_collector()
