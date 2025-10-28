"""
Planner Agent - Multi-Agent Framework
======================================

Breaks down high-level goals into actionable task graphs.

Capabilities:
- Goal decomposition
- Task sequencing
- Dependency identification
- Resource estimation
- Sub-goal generation

Features:
- Hierarchical planning
- Constraint-aware decomposition
- Task graph generation
- Adaptive planning based on feedback
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from core.agents.base_agent import BaseAgent, AgentCapability, TaskResult, AgentStatus

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Plans and decomposes complex goals into executable tasks.
    """

    def __init__(self, agent_id: str = "planner"):
        super().__init__(
            agent_id=agent_id,
            role="planner",
            capabilities=[
                AgentCapability.PLANNING,
            ]
        )

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if planner can handle task"""
        task_type = task.get("type", "")
        return task_type in [
            "plan",
            "decompose",
            "generate_tasks",
            "create_workflow"
        ]

    def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute planning task"""
        task_id = task.get("task_id", str(datetime.now().timestamp()))
        start_time = datetime.now()

        # Update status
        self.status = AgentStatus.BUSY
        self.current_task = task_id
        self.total_tasks += 1

        try:
            # Validate safety
            if not self._validate_safety(task):
                raise ValueError("Task failed safety validation")

            # Track resource usage
            if not self._track_resource_usage("api_calls", 1.0):
                raise RuntimeError("API call quota exceeded")

            task_type = task.get("type")
            params = task.get("params", {})

            if task_type == "plan":
                result = self._plan_goal(params)
            elif task_type == "decompose":
                result = self._decompose_task(params)
            elif task_type == "generate_tasks":
                result = self._generate_task_graph(params)
            elif task_type == "create_workflow":
                result = self._create_workflow(params)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()

            # Success
            self.status = AgentStatus.IDLE
            self.current_task = None
            self.successful_tasks += 1
            self.completed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=True,
                result=result,
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)

            return task_result

        except Exception as e:
            logger.error(f"Planner task failed: {e}", exc_info=True)

            execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.IDLE
            self.current_task = None
            self.failed_tasks_count += 1
            self.failed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)

            return task_result

    def _plan_goal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Plan high-level goal"""
        goal = params.get("goal", "")
        constraints = params.get("constraints", {})

        logger.info(f"Planning goal: {goal}")

        # Simple rule-based decomposition
        # In production, would use LLM or symbolic planner
        tasks = []

        if "code" in goal.lower() or "implement" in goal.lower():
            tasks = [
                {"type": "research", "description": "Research requirements and best practices"},
                {"type": "design", "description": "Design solution architecture"},
                {"type": "implement", "description": "Implement code"},
                {"type": "test", "description": "Write and run tests"},
                {"type": "review", "description": "Code review and quality check"},
            ]
        elif "research" in goal.lower() or "analyze" in goal.lower():
            tasks = [
                {"type": "gather", "description": "Gather information from sources"},
                {"type": "analyze", "description": "Analyze collected data"},
                {"type": "summarize", "description": "Summarize findings"},
                {"type": "report", "description": "Generate report"},
            ]
        elif "trade" in goal.lower() or "buy" in goal.lower() or "sell" in goal.lower():
            tasks = [
                {"type": "market_analysis", "description": "Analyze market conditions"},
                {"type": "risk_assessment", "description": "Assess trade risks"},
                {"type": "strategy_selection", "description": "Select trading strategy"},
                {"type": "execute_trade", "description": "Execute trade"},
                {"type": "monitor", "description": "Monitor position"},
            ]
        else:
            # Generic plan
            tasks = [
                {"type": "analyze", "description": f"Analyze goal: {goal}"},
                {"type": "execute", "description": "Execute main task"},
                {"type": "verify", "description": "Verify completion"},
            ]

        return {
            "goal": goal,
            "tasks": tasks,
            "estimated_duration": len(tasks) * 60,  # 60 seconds per task
            "constraints": constraints
        }

    def _decompose_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex task into subtasks"""
        task_description = params.get("task", "")

        logger.info(f"Decomposing task: {task_description}")

        # Simple decomposition
        subtasks = [
            f"Step 1: Prepare for {task_description}",
            f"Step 2: Execute {task_description}",
            f"Step 3: Verify {task_description} completion"
        ]

        return {
            "task": task_description,
            "subtasks": subtasks,
            "parallel_eligible": False
        }

    def _generate_task_graph(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task graph (DAG)"""
        goal = params.get("goal", "")

        logger.info(f"Generating task graph for: {goal}")

        # Generate simple sequential task graph
        # In production, would use more sophisticated planning
        nodes = [
            {"id": "task_1", "name": "Initialize", "dependencies": []},
            {"id": "task_2", "name": "Process", "dependencies": ["task_1"]},
            {"id": "task_3", "name": "Finalize", "dependencies": ["task_2"]},
        ]

        return {
            "goal": goal,
            "nodes": nodes,
            "graph_type": "sequential"
        }

    def _create_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow specification"""
        workflow_type = params.get("workflow_type", "generic")

        logger.info(f"Creating workflow: {workflow_type}")

        workflow = {
            "workflow_id": f"workflow_{datetime.now().timestamp()}",
            "type": workflow_type,
            "steps": [
                {"step": 1, "action": "start", "agent": "planner"},
                {"step": 2, "action": "execute", "agent": "operator"},
                {"step": 3, "action": "verify", "agent": "safety"},
            ]
        }

        return workflow
