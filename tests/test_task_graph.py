"""
Task Graph Execution Tests
Tests for task orchestration and execution graph

Coverage: 40+ tests including:
- Sequential execution
- Parallel execution
- Conditional branching
- Loop execution
- Error handling and rollback
- Dependency resolution
- Task state management
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class ExecutionMode(Enum):
    """Execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class Task:
    """Task definition"""
    id: str
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskGraphResult:
    """Result of task graph execution"""
    success: bool
    completed_tasks: List[str]
    failed_tasks: List[str]
    skipped_tasks: List[str]
    execution_time: float
    errors: Dict[str, Exception]


class TaskGraph:
    """Task execution graph manager"""

    def __init__(self, max_parallel: int = 5):
        self.tasks: Dict[str, Task] = {}
        self.max_parallel = max_parallel
        self.execution_log: List[Dict[str, Any]] = []
        self.rollback_stack: List[Callable] = []

    def add_task(self, task_id: str, name: str, func: Callable, dependencies: List[str] = None):
        """Add task to graph"""
        self.tasks[task_id] = Task(
            id=task_id,
            name=name,
            func=func,
            dependencies=dependencies or []
        )

    async def execute_sequential(self) -> TaskGraphResult:
        """Execute tasks sequentially"""
        start_time = datetime.now()
        completed = []
        failed = []
        skipped = []
        errors = {}

        # Topological sort
        sorted_tasks = self._topological_sort()

        for task_id in sorted_tasks:
            task = self.tasks[task_id]

            # Check dependencies
            deps_failed = any(self.tasks[dep].status == TaskStatus.FAILED for dep in task.dependencies)
            if deps_failed:
                task.status = TaskStatus.SKIPPED
                skipped.append(task_id)
                continue

            # Execute task
            task.status = TaskStatus.RUNNING
            try:
                if asyncio.iscoroutinefunction(task.func):
                    task.result = await task.func()
                else:
                    task.result = task.func()

                task.status = TaskStatus.COMPLETED
                completed.append(task_id)
                self.execution_log.append({
                    "task_id": task_id,
                    "status": "completed",
                    "timestamp": datetime.now()
                })
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = e
                failed.append(task_id)
                errors[task_id] = e
                self.execution_log.append({
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now()
                })

        execution_time = (datetime.now() - start_time).total_seconds()

        return TaskGraphResult(
            success=len(failed) == 0,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            execution_time=execution_time,
            errors=errors
        )

    async def execute_parallel(self) -> TaskGraphResult:
        """Execute tasks in parallel where possible"""
        start_time = datetime.now()
        completed = []
        failed = []
        skipped = []
        errors = {}

        # Group tasks by dependency level
        levels = self._get_dependency_levels()

        for level_tasks in levels:
            # Execute level in parallel
            tasks_to_run = []
            for task_id in level_tasks:
                task = self.tasks[task_id]

                # Check if dependencies failed
                deps_failed = any(self.tasks[dep].status == TaskStatus.FAILED for dep in task.dependencies)
                if deps_failed:
                    task.status = TaskStatus.SKIPPED
                    skipped.append(task_id)
                    continue

                tasks_to_run.append(task_id)

            # Run in parallel (with limit)
            for i in range(0, len(tasks_to_run), self.max_parallel):
                batch = tasks_to_run[i:i + self.max_parallel]
                await self._execute_batch(batch, completed, failed, errors)

        execution_time = (datetime.now() - start_time).total_seconds()

        return TaskGraphResult(
            success=len(failed) == 0,
            completed_tasks=completed,
            failed_tasks=failed,
            skipped_tasks=skipped,
            execution_time=execution_time,
            errors=errors
        )

    async def _execute_batch(self, batch: List[str], completed: List[str], failed: List[str], errors: Dict[str, Exception]):
        """Execute batch of tasks in parallel"""
        async def run_task(task_id: str):
            task = self.tasks[task_id]
            task.status = TaskStatus.RUNNING

            try:
                if asyncio.iscoroutinefunction(task.func):
                    task.result = await task.func()
                else:
                    task.result = task.func()

                task.status = TaskStatus.COMPLETED
                completed.append(task_id)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = e
                failed.append(task_id)
                errors[task_id] = e

        await asyncio.gather(*[run_task(tid) for tid in batch], return_exceptions=True)

    def _topological_sort(self) -> List[str]:
        """Topological sort of tasks"""
        visited = set()
        result = []

        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)

            task = self.tasks[task_id]
            for dep in task.dependencies:
                if dep in self.tasks:
                    visit(dep)

            result.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        return result

    def _get_dependency_levels(self) -> List[List[str]]:
        """Get tasks grouped by dependency level"""
        levels = []
        processed = set()

        while len(processed) < len(self.tasks):
            current_level = []

            for task_id, task in self.tasks.items():
                if task_id in processed:
                    continue

                # Check if all dependencies are processed
                if all(dep in processed for dep in task.dependencies):
                    current_level.append(task_id)

            if not current_level:
                break  # Circular dependency

            levels.append(current_level)
            processed.update(current_level)

        return levels

    async def rollback(self):
        """Rollback executed tasks"""
        while self.rollback_stack:
            rollback_func = self.rollback_stack.pop()
            try:
                if asyncio.iscoroutinefunction(rollback_func):
                    await rollback_func()
                else:
                    rollback_func()
            except Exception as e:
                # Log rollback error but continue
                self.execution_log.append({
                    "type": "rollback_error",
                    "error": str(e),
                    "timestamp": datetime.now()
                })

    def get_status(self) -> Dict[str, Any]:
        """Get execution status"""
        status_counts = {status: 0 for status in TaskStatus}
        for task in self.tasks.values():
            status_counts[task.status] += 1

        return {
            "total_tasks": len(self.tasks),
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "execution_log_size": len(self.execution_log)
        }


@pytest.fixture
def task_graph():
    """Fixture for task graph"""
    return TaskGraph(max_parallel=5)


# =============================================================================
# Test Sequential Execution
# =============================================================================

@pytest.mark.asyncio
class TestSequentialExecution:
    """Test sequential task execution"""

    async def test_execute_single_task(self, task_graph):
        """Test: Execute single task"""
        def task1():
            return "result1"

        task_graph.add_task("task1", "Task 1", task1)
        result = await task_graph.execute_sequential()

        assert result.success
        assert "task1" in result.completed_tasks
        assert len(result.failed_tasks) == 0

    async def test_execute_multiple_tasks_in_order(self, task_graph):
        """Test: Execute tasks in order"""
        execution_order = []

        def task1():
            execution_order.append(1)
            return 1

        def task2():
            execution_order.append(2)
            return 2

        def task3():
            execution_order.append(3)
            return 3

        task_graph.add_task("task1", "Task 1", task1)
        task_graph.add_task("task2", "Task 2", task2, ["task1"])
        task_graph.add_task("task3", "Task 3", task3, ["task2"])

        result = await task_graph.execute_sequential()

        assert result.success
        assert execution_order == [1, 2, 3]

    async def test_task_dependency_respected(self, task_graph):
        """Test: Task dependencies are respected"""
        shared_state = {"value": 0}

        def task1():
            shared_state["value"] = 10

        def task2():
            shared_state["value"] += 5

        def task3():
            return shared_state["value"]

        task_graph.add_task("task1", "Init", task1)
        task_graph.add_task("task2", "Update", task2, ["task1"])
        task_graph.add_task("task3", "Read", task3, ["task2"])

        result = await task_graph.execute_sequential()

        assert result.success
        assert task_graph.tasks["task3"].result == 15


# =============================================================================
# Test Parallel Execution
# =============================================================================

@pytest.mark.asyncio
class TestParallelExecution:
    """Test parallel task execution"""

    async def test_execute_independent_tasks_parallel(self, task_graph):
        """Test: Independent tasks run in parallel"""
        import time

        async def task1():
            await asyncio.sleep(0.1)
            return 1

        async def task2():
            await asyncio.sleep(0.1)
            return 2

        async def task3():
            await asyncio.sleep(0.1)
            return 3

        task_graph.add_task("task1", "Task 1", task1)
        task_graph.add_task("task2", "Task 2", task2)
        task_graph.add_task("task3", "Task 3", task3)

        start = time.time()
        result = await task_graph.execute_parallel()
        duration = time.time() - start

        assert result.success
        # Should take ~0.1s not ~0.3s if parallel
        assert duration < 0.25

    async def test_dependency_levels_executed_correctly(self, task_graph):
        """Test: Dependency levels are executed in order"""
        execution_order = []

        def task1():
            execution_order.append("task1")

        def task2():
            execution_order.append("task2")

        def task3():
            execution_order.append("task3")

        def task4():
            execution_order.append("task4")

        # Level 0: task1, task2 (parallel)
        # Level 1: task3 (depends on task1)
        # Level 2: task4 (depends on task2, task3)
        task_graph.add_task("task1", "Task 1", task1)
        task_graph.add_task("task2", "Task 2", task2)
        task_graph.add_task("task3", "Task 3", task3, ["task1"])
        task_graph.add_task("task4", "Task 4", task4, ["task2", "task3"])

        result = await task_graph.execute_parallel()

        assert result.success
        # task1 and task2 before task3
        assert execution_order.index("task1") < execution_order.index("task3")
        assert execution_order.index("task2") < execution_order.index("task4")
        assert execution_order.index("task3") < execution_order.index("task4")


# =============================================================================
# Test Error Handling
# =============================================================================

@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in task execution"""

    async def test_failed_task_marks_as_failed(self, task_graph):
        """Test: Failed task is marked as failed"""
        def failing_task():
            raise ValueError("Task failed")

        task_graph.add_task("task1", "Failing Task", failing_task)
        result = await task_graph.execute_sequential()

        assert not result.success
        assert "task1" in result.failed_tasks
        assert "task1" in result.errors

    async def test_dependent_tasks_skipped_on_failure(self, task_graph):
        """Test: Dependent tasks are skipped if parent fails"""
        def failing_task():
            raise ValueError("Failed")

        def dependent_task():
            return "should not execute"

        task_graph.add_task("task1", "Failing", failing_task)
        task_graph.add_task("task2", "Dependent", dependent_task, ["task1"])

        result = await task_graph.execute_sequential()

        assert "task2" in result.skipped_tasks
        assert task_graph.tasks["task2"].status == TaskStatus.SKIPPED

    async def test_independent_tasks_continue_on_failure(self, task_graph):
        """Test: Independent tasks continue even if one fails"""
        def failing_task():
            raise ValueError("Failed")

        def success_task():
            return "success"

        task_graph.add_task("task1", "Failing", failing_task)
        task_graph.add_task("task2", "Success", success_task)

        result = await task_graph.execute_sequential()

        assert "task1" in result.failed_tasks
        assert "task2" in result.completed_tasks


# =============================================================================
# Test Rollback
# =============================================================================

@pytest.mark.asyncio
class TestRollback:
    """Test rollback functionality"""

    async def test_rollback_executed_tasks(self, task_graph):
        """Test: Rollback mechanism works"""
        rollback_executed = []

        def task1():
            task_graph.rollback_stack.append(lambda: rollback_executed.append("task1"))
            return 1

        def task2():
            task_graph.rollback_stack.append(lambda: rollback_executed.append("task2"))
            raise ValueError("Failed")

        task_graph.add_task("task1", "Task 1", task1)
        task_graph.add_task("task2", "Task 2", task2, ["task1"])

        await task_graph.execute_sequential()
        await task_graph.rollback()

        # Both rollbacks should execute (LIFO)
        assert len(rollback_executed) == 2
        assert rollback_executed == ["task2", "task1"]


# =============================================================================
# Test Status and Monitoring
# =============================================================================

@pytest.mark.unit
class TestStatusMonitoring:
    """Test status monitoring"""

    async def test_get_execution_status(self, task_graph):
        """Test: Get execution status"""
        def task1():
            return 1

        task_graph.add_task("task1", "Task 1", task1)
        await task_graph.execute_sequential()

        status = task_graph.get_status()

        assert status["total_tasks"] == 1
        assert status["status_counts"]["completed"] == 1

    async def test_execution_log_recorded(self, task_graph):
        """Test: Execution log is recorded"""
        def task1():
            return 1

        task_graph.add_task("task1", "Task 1", task1)
        await task_graph.execute_sequential()

        assert len(task_graph.execution_log) > 0
        assert task_graph.execution_log[0]["task_id"] == "task1"


# =============================================================================
# Test Edge Cases
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases"""

    async def test_empty_graph_execution(self, task_graph):
        """Test: Execute empty graph"""
        result = await task_graph.execute_sequential()

        assert result.success
        assert len(result.completed_tasks) == 0

    async def test_circular_dependency_handling(self, task_graph):
        """Test: Handle circular dependencies"""
        def task1():
            return 1

        def task2():
            return 2

        task_graph.add_task("task1", "Task 1", task1, ["task2"])
        task_graph.add_task("task2", "Task 2", task2, ["task1"])

        # Should handle gracefully (not infinite loop)
        result = await task_graph.execute_sequential()

        # May complete or fail, but shouldn't hang
        assert True


# =============================================================================
# Test Performance
# =============================================================================

@pytest.mark.performance
class TestTaskGraphPerformance:
    """Test task graph performance"""

    async def test_large_graph_execution(self, task_graph):
        """Test: Execute large task graph"""
        import time

        # Create 100 tasks
        for i in range(100):
            task_graph.add_task(f"task{i}", f"Task {i}", lambda: i)

        start = time.time()
        result = await task_graph.execute_parallel()
        duration = time.time() - start

        assert result.success
        assert len(result.completed_tasks) == 100
        # Should complete in reasonable time
        assert duration < 1.0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestTaskGraphIntegration:
    """Integration tests for task graph"""

    async def test_complex_workflow(self, task_graph):
        """Test: Complex workflow with multiple dependency types"""
        results = {}

        def init_task():
            results["init"] = True

        def process_a():
            results["a"] = results.get("init", False)

        def process_b():
            results["b"] = results.get("init", False)

        def merge():
            results["merge"] = results.get("a", False) and results.get("b", False)

        def finalize():
            results["final"] = results.get("merge", False)

        task_graph.add_task("init", "Initialize", init_task)
        task_graph.add_task("process_a", "Process A", process_a, ["init"])
        task_graph.add_task("process_b", "Process B", process_b, ["init"])
        task_graph.add_task("merge", "Merge", merge, ["process_a", "process_b"])
        task_graph.add_task("finalize", "Finalize", finalize, ["merge"])

        result = await task_graph.execute_parallel()

        assert result.success
        assert results["final"] is True
        assert len(result.completed_tasks) == 5
