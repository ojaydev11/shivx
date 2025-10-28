"""
Task Graph Executor - Multi-Agent Orchestration Framework
==========================================================

DAG-based task composition and execution with:
- Sequential, parallel, conditional, and loop task types
- Dependency resolution with topological sort
- Parallel execution using ThreadPoolExecutor
- Error handling and rollback capabilities
- Progress tracking and persistence
- Visualization export (DOT format for graphviz)

Features:
- Directed Acyclic Graph (DAG) validation
- Dynamic task graph construction
- Concurrent execution where possible
- Checkpoint/resume support
- Audit trail integration
- Resource-aware scheduling
"""

import json
import logging
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, asdict, field
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task node types"""
    SEQUENTIAL = "sequential"  # A → B → C
    PARALLEL = "parallel"  # A || B || C
    CONDITIONAL = "conditional"  # if/else branches
    LOOP = "loop"  # for each item


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"  # Dependencies satisfied
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Conditional branch not taken
    CANCELLED = "cancelled"


@dataclass
class TaskNode:
    """Task node in execution graph"""
    task_id: str
    name: str
    task_type: TaskType
    handler: Optional[Callable] = None  # Task execution function
    dependencies: List[str] = field(default_factory=list)  # Task IDs that must complete first
    params: Dict[str, Any] = field(default_factory=dict)  # Task parameters
    condition: Optional[Callable] = None  # For conditional tasks
    loop_items: Optional[List[Any]] = None  # For loop tasks
    timeout_sec: Optional[float] = None  # Execution timeout
    retry_count: int = 0  # Number of retries on failure
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_sec: Optional[float] = None


@dataclass
class ExecutionResult:
    """Task graph execution result"""
    graph_id: str
    status: str  # completed, partial, failed
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    execution_time_sec: float
    results: Dict[str, Any]  # task_id -> result
    errors: Dict[str, str]  # task_id -> error
    timestamp: str


class TaskGraph:
    """
    DAG-based task graph executor with parallel execution support.

    Features:
    - Automatic dependency resolution
    - Parallel execution where possible
    - Error handling with rollback
    - Progress tracking
    - Checkpoint/resume
    """

    def __init__(self, graph_id: Optional[str] = None, max_workers: int = 4):
        """
        Initialize task graph.

        Args:
            graph_id: Unique graph identifier
            max_workers: Maximum parallel workers
        """
        self.graph_id = graph_id or str(uuid4())
        self.max_workers = max_workers

        # Task registry
        self.tasks: Dict[str, TaskNode] = {}

        # Adjacency list (task_id -> [dependent_task_ids])
        self.graph: Dict[str, List[str]] = defaultdict(list)

        # Reverse adjacency list (for dependency tracking)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)

        # Execution state
        self.execution_order: List[str] = []
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()

        # Thread pool
        self.executor: Optional[ThreadPoolExecutor] = None
        self.futures: Dict[str, Future] = {}

        # Locks
        self.lock = threading.Lock()

        # Rollback handlers
        self.rollback_handlers: Dict[str, Callable] = {}

        logger.info(f"TaskGraph initialized: {self.graph_id} (max_workers: {max_workers})")

    def add_task(
        self,
        task_id: str,
        name: str,
        handler: Callable,
        task_type: TaskType = TaskType.SEQUENTIAL,
        dependencies: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
        condition: Optional[Callable] = None,
        loop_items: Optional[List[Any]] = None,
        timeout_sec: Optional[float] = None,
        max_retries: int = 3,
        rollback_handler: Optional[Callable] = None
    ) -> TaskNode:
        """
        Add task to graph.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            handler: Task execution function (receives params dict, returns result)
            task_type: Task type (sequential, parallel, conditional, loop)
            dependencies: List of task IDs that must complete first
            params: Task parameters
            condition: Condition function for conditional tasks (returns bool)
            loop_items: Items to loop over for loop tasks
            timeout_sec: Task timeout in seconds
            max_retries: Maximum retry attempts on failure
            rollback_handler: Function to call on rollback

        Returns:
            Created TaskNode
        """
        if task_id in self.tasks:
            raise ValueError(f"Task already exists: {task_id}")

        dependencies = dependencies or []

        # Validate dependencies exist
        for dep_id in dependencies:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency not found: {dep_id}")

        task = TaskNode(
            task_id=task_id,
            name=name,
            task_type=task_type,
            handler=handler,
            dependencies=dependencies,
            params=params or {},
            condition=condition,
            loop_items=loop_items,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            status=TaskStatus.PENDING
        )

        self.tasks[task_id] = task

        # Update graph
        for dep_id in dependencies:
            self.graph[dep_id].append(task_id)
            self.reverse_graph[task_id].append(dep_id)

        if rollback_handler:
            self.rollback_handlers[task_id] = rollback_handler

        logger.debug(f"Task added: {task_id} ({name}) with {len(dependencies)} dependencies")

        return task

    def validate_dag(self) -> bool:
        """
        Validate graph is a DAG (no cycles).

        Returns:
            True if valid DAG, False otherwise
        """
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    logger.error("Cycle detected in task graph!")
                    return False

        return True

    def topological_sort(self) -> List[str]:
        """
        Compute topological ordering of tasks using Kahn's algorithm.

        Returns:
            List of task IDs in execution order
        """
        # Calculate in-degree for each node
        in_degree = {task_id: len(self.reverse_graph.get(task_id, []))
                     for task_id in self.tasks}

        # Queue of tasks with no dependencies
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])

        sorted_order = []

        while queue:
            task_id = queue.popleft()
            sorted_order.append(task_id)

            # Reduce in-degree for dependent tasks
            for dependent in self.graph.get(task_id, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_order) != len(self.tasks):
            raise RuntimeError("Graph contains cycle - cannot compute topological order")

        return sorted_order

    def get_ready_tasks(self) -> List[str]:
        """
        Get tasks that are ready to execute (all dependencies satisfied).

        Returns:
            List of task IDs ready for execution
        """
        ready = []

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies completed
            deps_satisfied = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )

            if deps_satisfied:
                ready.append(task_id)

        return ready

    def execute_task(self, task_id: str) -> Any:
        """
        Execute a single task.

        Args:
            task_id: Task to execute

        Returns:
            Task result

        Raises:
            Exception if task fails after retries
        """
        task = self.tasks[task_id]

        logger.info(f"Executing task: {task_id} ({task.name})")

        with self.lock:
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.now().isoformat()
            self.running_tasks.add(task_id)

        try:
            # Handle conditional tasks
            if task.task_type == TaskType.CONDITIONAL and task.condition:
                condition_met = task.condition(task.params)
                if not condition_met:
                    with self.lock:
                        task.status = TaskStatus.SKIPPED
                        task.end_time = datetime.now().isoformat()
                        self.running_tasks.discard(task_id)
                    logger.info(f"Task skipped (condition not met): {task_id}")
                    return None

            # Handle loop tasks
            if task.task_type == TaskType.LOOP and task.loop_items:
                results = []
                for item in task.loop_items:
                    item_params = {**task.params, "loop_item": item}
                    result = task.handler(item_params)
                    results.append(result)

                with self.lock:
                    task.result = results
                    task.status = TaskStatus.COMPLETED
                    task.end_time = datetime.now().isoformat()
                    start_dt = datetime.fromisoformat(task.start_time)
                    end_dt = datetime.fromisoformat(task.end_time)
                    task.duration_sec = (end_dt - start_dt).total_seconds()
                    self.running_tasks.discard(task_id)
                    self.completed_tasks.add(task_id)

                logger.info(f"Loop task completed: {task_id} ({len(results)} items)")
                return results

            # Execute regular task with retries
            last_error = None
            for attempt in range(task.max_retries + 1):
                try:
                    # Execute with timeout
                    if task.timeout_sec:
                        import signal

                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Task timeout after {task.timeout_sec}s")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(task.timeout_sec))

                    result = task.handler(task.params)

                    if task.timeout_sec:
                        signal.alarm(0)  # Cancel alarm

                    with self.lock:
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.end_time = datetime.now().isoformat()
                        start_dt = datetime.fromisoformat(task.start_time)
                        end_dt = datetime.fromisoformat(task.end_time)
                        task.duration_sec = (end_dt - start_dt).total_seconds()
                        self.running_tasks.discard(task_id)
                        self.completed_tasks.add(task_id)

                    logger.info(f"Task completed: {task_id} (duration: {task.duration_sec:.2f}s)")
                    return result

                except Exception as e:
                    last_error = str(e)
                    task.retry_count = attempt + 1

                    if attempt < task.max_retries:
                        logger.warning(f"Task failed (attempt {attempt + 1}/{task.max_retries}): {task_id} - {e}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise

            raise RuntimeError(f"Task failed after {task.max_retries} retries: {last_error}")

        except Exception as e:
            with self.lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = datetime.now().isoformat()
                self.running_tasks.discard(task_id)
                self.failed_tasks.add(task_id)

            logger.error(f"Task failed: {task_id} - {e}")
            raise

    def execute(
        self,
        enable_parallel: bool = True,
        stop_on_error: bool = True
    ) -> ExecutionResult:
        """
        Execute task graph.

        Args:
            enable_parallel: Enable parallel execution where possible
            stop_on_error: Stop execution on first error

        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()

        logger.info(f"Executing task graph: {self.graph_id}")

        # Validate DAG
        if not self.validate_dag():
            raise RuntimeError("Graph contains cycles - cannot execute")

        # Compute execution order
        self.execution_order = self.topological_sort()
        logger.info(f"Execution order: {len(self.execution_order)} tasks")

        # Execute tasks
        if enable_parallel and self.max_workers > 1:
            self._execute_parallel(stop_on_error)
        else:
            self._execute_sequential(stop_on_error)

        # Collect results
        execution_time = time.time() - start_time

        results = {
            task_id: task.result
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }

        errors = {
            task_id: task.error
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.FAILED
        }

        completed_count = len(self.completed_tasks)
        failed_count = len(self.failed_tasks)
        skipped_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)

        # Determine overall status
        if failed_count == 0:
            status = "completed"
        elif completed_count > 0:
            status = "partial"
        else:
            status = "failed"

        result = ExecutionResult(
            graph_id=self.graph_id,
            status=status,
            total_tasks=len(self.tasks),
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            skipped_tasks=skipped_count,
            execution_time_sec=execution_time,
            results=results,
            errors=errors,
            timestamp=datetime.now().isoformat()
        )

        logger.info(
            f"Graph execution {status}: {completed_count}/{len(self.tasks)} tasks "
            f"({execution_time:.2f}s)"
        )

        return result

    def _execute_sequential(self, stop_on_error: bool):
        """Execute tasks sequentially in topological order"""
        for task_id in self.execution_order:
            task = self.tasks[task_id]

            # Skip if dependencies failed
            if any(self.tasks[dep_id].status == TaskStatus.FAILED
                   for dep_id in task.dependencies):
                task.status = TaskStatus.SKIPPED
                logger.warning(f"Task skipped (dependency failed): {task_id}")
                continue

            try:
                self.execute_task(task_id)
            except Exception as e:
                if stop_on_error:
                    logger.error(f"Stopping execution due to task failure: {task_id}")
                    break

    def _execute_parallel(self, stop_on_error: bool):
        """Execute tasks in parallel where possible"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        try:
            while len(self.completed_tasks) + len(self.failed_tasks) < len(self.tasks):
                # Get ready tasks
                ready_tasks = self.get_ready_tasks()

                # Skip tasks with failed dependencies
                ready_tasks = [
                    task_id for task_id in ready_tasks
                    if not any(self.tasks[dep_id].status == TaskStatus.FAILED
                              for dep_id in self.tasks[task_id].dependencies)
                ]

                # Mark skipped tasks
                for task_id in self.execution_order:
                    task = self.tasks[task_id]
                    if task.status == TaskStatus.PENDING and any(
                        self.tasks[dep_id].status == TaskStatus.FAILED
                        for dep_id in task.dependencies
                    ):
                        task.status = TaskStatus.SKIPPED

                # Submit ready tasks
                for task_id in ready_tasks:
                    if task_id not in self.futures:
                        future = self.executor.submit(self.execute_task, task_id)
                        self.futures[task_id] = future
                        logger.debug(f"Task submitted: {task_id}")

                # Wait for at least one task to complete
                if self.futures:
                    done_futures = [f for f in self.futures.values() if f.done()]

                    if not done_futures:
                        time.sleep(0.1)  # Brief sleep before checking again
                        continue

                    # Process completed tasks
                    for task_id, future in list(self.futures.items()):
                        if future.done():
                            try:
                                future.result()  # Raise exception if task failed
                            except Exception as e:
                                logger.error(f"Task execution error: {task_id} - {e}")
                                if stop_on_error:
                                    logger.error("Stopping execution due to error")
                                    # Cancel remaining tasks
                                    for f in self.futures.values():
                                        f.cancel()
                                    return

                            del self.futures[task_id]
                else:
                    # No tasks running and no ready tasks - check if we're done
                    if not ready_tasks:
                        break

        finally:
            if self.executor:
                self.executor.shutdown(wait=True)

    def rollback(self):
        """Rollback completed tasks in reverse order"""
        logger.warning(f"Rolling back graph: {self.graph_id}")

        rollback_order = list(reversed(list(self.completed_tasks)))

        for task_id in rollback_order:
            if task_id in self.rollback_handlers:
                try:
                    logger.info(f"Rolling back task: {task_id}")
                    self.rollback_handlers[task_id]()
                except Exception as e:
                    logger.error(f"Rollback failed for task {task_id}: {e}")

        logger.info("Rollback completed")

    def save(self, file_path: Path):
        """Save task graph to JSON file"""
        data = {
            "graph_id": self.graph_id,
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "name": task.name,
                    "task_type": task.task_type.value,
                    "dependencies": task.dependencies,
                    "params": task.params,
                    "status": task.status.value,
                    "result": str(task.result) if task.result else None,
                    "error": task.error,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "duration_sec": task.duration_sec,
                }
                for task_id, task in self.tasks.items()
            }
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Task graph saved: {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> 'TaskGraph':
        """Load task graph from JSON file (without handlers)"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        graph = cls(graph_id=data["graph_id"])

        # Note: Handlers cannot be serialized, would need to be re-registered
        logger.warning("Loaded graph without task handlers - handlers must be re-registered")

        logger.info(f"Task graph loaded: {file_path}")
        return graph

    def export_dot(self, file_path: Path):
        """Export task graph to DOT format for visualization with graphviz"""
        lines = ["digraph TaskGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")

        # Add nodes with status colors
        colors = {
            TaskStatus.PENDING: "lightgray",
            TaskStatus.READY: "lightblue",
            TaskStatus.RUNNING: "yellow",
            TaskStatus.COMPLETED: "lightgreen",
            TaskStatus.FAILED: "red",
            TaskStatus.SKIPPED: "orange",
            TaskStatus.CANCELLED: "darkgray",
        }

        for task_id, task in self.tasks.items():
            color = colors.get(task.status, "white")
            label = f"{task.name}\\n({task.status.value})"
            if task.duration_sec:
                label += f"\\n{task.duration_sec:.2f}s"

            lines.append(f'  "{task_id}" [label="{label}", fillcolor="{color}", style=filled];')

        # Add edges
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                lines.append(f'  "{dep_id}" -> "{task_id}";')

        lines.append("}")

        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"DOT graph exported: {file_path}")

    def get_progress(self) -> Dict[str, Any]:
        """Get current execution progress"""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        running = len(self.running_tasks)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress": completed / total if total > 0 else 0.0,
            "status_breakdown": {
                status.value: sum(1 for t in self.tasks.values() if t.status == status)
                for status in TaskStatus
            }
        }
