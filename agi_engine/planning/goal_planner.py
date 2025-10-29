"""
Goal-Directed Planning System

Enables AGI to set goals, decompose them, create executable plans,
and pursue objectives autonomously.

Key capabilities:
- Goal representation and decomposition
- Multi-step plan generation
- Constraint satisfaction
- Resource allocation
- Progress tracking
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque


class GoalStatus(str, Enum):
    """Status of a goal"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class PlanStepStatus(str, Enum):
    """Status of a plan step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Goal:
    """Represents a goal to be achieved"""
    description: str
    goal_id: str
    priority: float = 1.0  # 0-1, higher = more important
    deadline: Optional[float] = None  # Unix timestamp
    parent_goal: Optional[str] = None  # For hierarchical goals
    subgoals: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    resources_needed: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def is_terminal(self) -> bool:
        """Check if this is a terminal (leaf) goal"""
        return len(self.subgoals) == 0

    def is_overdue(self) -> bool:
        """Check if goal is past deadline"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass
class PlanStep:
    """A single step in a plan"""
    step_id: str
    action: str
    description: str
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    resources_required: Dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 60.0  # seconds
    status: PlanStepStatus = PlanStepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # Other step IDs
    retry_count: int = 0
    max_retries: int = 3

    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_steps for dep in self.dependencies)


@dataclass
class Plan:
    """A plan to achieve a goal"""
    plan_id: str
    goal_id: str
    steps: List[PlanStep]
    estimated_total_duration: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    success: bool = False

    def get_next_executable_step(self, completed_steps: Set[str]) -> Optional[PlanStep]:
        """Get the next step that can be executed"""
        for step in self.steps:
            if step.status == PlanStepStatus.PENDING and step.can_execute(completed_steps):
                return step
        return None

    def progress(self) -> float:
        """Calculate plan completion progress (0-1)"""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == PlanStepStatus.COMPLETED)
        return completed / len(self.steps)


class GoalPlanner:
    """
    Goal-directed planning system for AGI

    Features:
    - Hierarchical goal decomposition
    - Multi-step plan generation
    - Constraint satisfaction
    - Resource management
    - Progress tracking
    """

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.plans: Dict[str, Plan] = {}
        self.goal_counter = 0
        self.plan_counter = 0
        self.step_counter = 0
        self.available_resources: Dict[str, float] = {
            "compute": 100.0,
            "memory": 100.0,
            "time": float('inf'),
        }

    def create_goal(
        self,
        description: str,
        priority: float = 1.0,
        deadline: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
        success_criteria: Optional[List[str]] = None,
        resources_needed: Optional[Dict[str, float]] = None
    ) -> Goal:
        """Create a new goal"""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"

        goal = Goal(
            description=description,
            goal_id=goal_id,
            priority=priority,
            deadline=deadline,
            constraints=constraints or {},
            success_criteria=success_criteria or [],
            resources_needed=resources_needed or {}
        )

        self.goals[goal_id] = goal
        return goal

    def decompose_goal(self, goal: Goal) -> List[Goal]:
        """
        Decompose a high-level goal into subgoals

        Uses heuristics and domain knowledge to break down complex goals
        """
        subgoals = []

        # Simple heuristic-based decomposition
        # In a real system, this would use the reasoning engine

        desc_lower = goal.description.lower()

        # Pattern: "Build X"
        if "build" in desc_lower or "create" in desc_lower:
            subgoals.extend(self._decompose_build_goal(goal))

        # Pattern: "Learn X"
        elif "learn" in desc_lower or "understand" in desc_lower:
            subgoals.extend(self._decompose_learn_goal(goal))

        # Pattern: "Optimize X"
        elif "optimize" in desc_lower or "improve" in desc_lower:
            subgoals.extend(self._decompose_optimize_goal(goal))

        # Pattern: "Solve X"
        elif "solve" in desc_lower or "fix" in desc_lower:
            subgoals.extend(self._decompose_solve_goal(goal))

        # Default: break into analysis, planning, execution, validation
        else:
            subgoals.extend(self._decompose_generic_goal(goal))

        # Link subgoals to parent
        for subgoal in subgoals:
            subgoal.parent_goal = goal.goal_id
            goal.subgoals.append(subgoal.goal_id)
            self.goals[subgoal.goal_id] = subgoal

        return subgoals

    def _decompose_build_goal(self, goal: Goal) -> List[Goal]:
        """Decompose a 'build' goal"""
        subgoals = []

        # 1. Research/Design
        subgoals.append(self.create_goal(
            f"Research and design {goal.description.split('build')[-1].strip()}",
            priority=goal.priority * 0.9,
            deadline=goal.deadline
        ))

        # 2. Implement core functionality
        subgoals.append(self.create_goal(
            f"Implement core functionality for {goal.description}",
            priority=goal.priority,
            deadline=goal.deadline
        ))

        # 3. Test and validate
        subgoals.append(self.create_goal(
            f"Test and validate {goal.description}",
            priority=goal.priority * 0.8,
            deadline=goal.deadline
        ))

        # 4. Integrate and deploy
        subgoals.append(self.create_goal(
            f"Integrate and deploy {goal.description}",
            priority=goal.priority * 0.7,
            deadline=goal.deadline
        ))

        return subgoals

    def _decompose_learn_goal(self, goal: Goal) -> List[Goal]:
        """Decompose a 'learn' goal"""
        subgoals = []

        # 1. Gather resources/data
        subgoals.append(self.create_goal(
            f"Gather learning resources for {goal.description}",
            priority=goal.priority * 0.9
        ))

        # 2. Study/practice
        subgoals.append(self.create_goal(
            f"Study and practice {goal.description}",
            priority=goal.priority
        ))

        # 3. Apply knowledge
        subgoals.append(self.create_goal(
            f"Apply learned knowledge from {goal.description}",
            priority=goal.priority * 0.8
        ))

        # 4. Validate understanding
        subgoals.append(self.create_goal(
            f"Validate understanding of {goal.description}",
            priority=goal.priority * 0.7
        ))

        return subgoals

    def _decompose_optimize_goal(self, goal: Goal) -> List[Goal]:
        """Decompose an 'optimize' goal"""
        subgoals = []

        # 1. Profile/analyze current state
        subgoals.append(self.create_goal(
            f"Analyze current state for {goal.description}",
            priority=goal.priority * 0.9
        ))

        # 2. Identify bottlenecks
        subgoals.append(self.create_goal(
            f"Identify optimization opportunities in {goal.description}",
            priority=goal.priority
        ))

        # 3. Implement improvements
        subgoals.append(self.create_goal(
            f"Implement optimizations for {goal.description}",
            priority=goal.priority * 0.95
        ))

        # 4. Measure improvements
        subgoals.append(self.create_goal(
            f"Measure and validate improvements in {goal.description}",
            priority=goal.priority * 0.8
        ))

        return subgoals

    def _decompose_solve_goal(self, goal: Goal) -> List[Goal]:
        """Decompose a 'solve' goal"""
        subgoals = []

        # 1. Understand problem
        subgoals.append(self.create_goal(
            f"Understand problem in {goal.description}",
            priority=goal.priority
        ))

        # 2. Generate solutions
        subgoals.append(self.create_goal(
            f"Generate solutions for {goal.description}",
            priority=goal.priority * 0.9
        ))

        # 3. Implement solution
        subgoals.append(self.create_goal(
            f"Implement solution for {goal.description}",
            priority=goal.priority * 0.95
        ))

        # 4. Verify solution
        subgoals.append(self.create_goal(
            f"Verify solution works for {goal.description}",
            priority=goal.priority * 0.8
        ))

        return subgoals

    def _decompose_generic_goal(self, goal: Goal) -> List[Goal]:
        """Generic goal decomposition"""
        subgoals = []

        # Standard 4-phase approach
        phases = [
            ("Analyze", 0.9),
            ("Plan", 0.85),
            ("Execute", 1.0),
            ("Validate", 0.8)
        ]

        for phase, priority_mult in phases:
            subgoals.append(self.create_goal(
                f"{phase}: {goal.description}",
                priority=goal.priority * priority_mult,
                deadline=goal.deadline
            ))

        return subgoals

    def generate_plan(self, goal: Goal) -> Plan:
        """
        Generate an executable plan to achieve a goal

        Creates a sequence of steps with dependencies, resources, and timing
        """
        self.plan_counter += 1
        plan_id = f"plan_{self.plan_counter}"

        steps = []

        # If goal has subgoals, create steps for each
        if goal.subgoals:
            for i, subgoal_id in enumerate(goal.subgoals):
                subgoal = self.goals.get(subgoal_id)
                if subgoal:
                    step = self._create_step_for_goal(subgoal, previous_step_id=steps[-1].step_id if steps else None)
                    steps.append(step)
        else:
            # Terminal goal - create concrete action steps
            steps = self._generate_concrete_steps(goal)

        # Calculate total estimated duration
        total_duration = sum(step.estimated_duration for step in steps)

        plan = Plan(
            plan_id=plan_id,
            goal_id=goal.goal_id,
            steps=steps,
            estimated_total_duration=total_duration
        )

        self.plans[plan_id] = plan
        return plan

    def _create_step_for_goal(self, goal: Goal, previous_step_id: Optional[str] = None) -> PlanStep:
        """Create a plan step for achieving a goal"""
        self.step_counter += 1
        step_id = f"step_{self.step_counter}"

        dependencies = [previous_step_id] if previous_step_id else []

        return PlanStep(
            step_id=step_id,
            action=f"achieve_goal_{goal.goal_id}",
            description=f"Achieve: {goal.description}",
            preconditions=[f"goal_{goal.goal_id}_ready"],
            effects=[f"goal_{goal.goal_id}_achieved"],
            resources_required=goal.resources_needed,
            estimated_duration=300.0,  # 5 minutes default
            dependencies=dependencies
        )

    def _generate_concrete_steps(self, goal: Goal) -> List[PlanStep]:
        """Generate concrete action steps for a terminal goal"""
        steps = []

        # Standard execution pattern
        step_templates = [
            ("prepare", "Prepare resources and environment", 60.0),
            ("execute", "Execute main action", 180.0),
            ("verify", "Verify results", 60.0),
            ("cleanup", "Clean up and finalize", 30.0),
        ]

        for i, (action_type, desc_template, duration) in enumerate(step_templates):
            self.step_counter += 1
            step_id = f"step_{self.step_counter}"

            dependencies = [f"step_{self.step_counter - 1}"] if i > 0 else []

            step = PlanStep(
                step_id=step_id,
                action=f"{action_type}_{goal.goal_id}",
                description=f"{desc_template} for: {goal.description}",
                dependencies=dependencies,
                estimated_duration=duration
            )
            steps.append(step)

        return steps

    def prioritize_goals(self) -> List[Goal]:
        """
        Prioritize goals based on multiple factors

        Considers:
        - Goal priority
        - Deadlines
        - Resource availability
        - Dependencies
        """
        pending_goals = [g for g in self.goals.values() if g.status == GoalStatus.PENDING]

        def priority_score(goal: Goal) -> float:
            score = goal.priority

            # Boost score if deadline is approaching
            if goal.deadline:
                time_until_deadline = goal.deadline - time.time()
                if time_until_deadline < 3600:  # Less than 1 hour
                    score *= 2.0
                elif time_until_deadline < 86400:  # Less than 1 day
                    score *= 1.5

            # Reduce score if resources not available
            for resource, amount in goal.resources_needed.items():
                if resource in self.available_resources:
                    if self.available_resources[resource] < amount:
                        score *= 0.5

            return score

        return sorted(pending_goals, key=priority_score, reverse=True)

    def execute_step(self, step: PlanStep) -> bool:
        """
        Execute a single plan step

        Returns True if successful, False otherwise

        In a real implementation, this would actually perform the action.
        For now, it's a simulation.
        """
        step.status = PlanStepStatus.IN_PROGRESS

        # Check resources
        for resource, amount in step.resources_required.items():
            if resource in self.available_resources:
                if self.available_resources[resource] < amount:
                    step.status = PlanStepStatus.FAILED
                    return False
                self.available_resources[resource] -= amount

        # Simulate execution (in real system, this would call actual actions)
        # For now, just mark as completed
        time.sleep(0.01)  # Tiny delay to simulate work

        # Success rate depends on step type
        # More complex actions have higher failure rates
        import random
        success_rate = 0.9  # 90% success rate

        if random.random() < success_rate:
            step.status = PlanStepStatus.COMPLETED
            return True
        else:
            step.retry_count += 1
            if step.retry_count >= step.max_retries:
                step.status = PlanStepStatus.FAILED
                return False
            else:
                step.status = PlanStepStatus.PENDING  # Retry
                return False

    def execute_plan(self, plan: Plan) -> bool:
        """
        Execute a complete plan

        Returns True if plan completes successfully
        """
        plan.started_at = time.time()
        completed_steps = set()

        while plan.progress() < 1.0:
            # Get next executable step
            step = plan.get_next_executable_step(completed_steps)

            if step is None:
                # No executable steps - check if we're done or blocked
                if all(s.status == PlanStepStatus.COMPLETED for s in plan.steps):
                    break
                else:
                    # Blocked - no steps can execute
                    plan.success = False
                    return False

            # Execute step
            success = self.execute_step(step)

            if success:
                completed_steps.add(step.step_id)
            elif step.status == PlanStepStatus.FAILED:
                # Step failed permanently
                plan.success = False
                plan.completed_at = time.time()
                return False

        # All steps completed
        plan.success = True
        plan.completed_at = time.time()
        return True

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of planner status"""
        return {
            "total_goals": len(self.goals),
            "pending_goals": sum(1 for g in self.goals.values() if g.status == GoalStatus.PENDING),
            "in_progress_goals": sum(1 for g in self.goals.values() if g.status == GoalStatus.IN_PROGRESS),
            "completed_goals": sum(1 for g in self.goals.values() if g.status == GoalStatus.COMPLETED),
            "failed_goals": sum(1 for g in self.goals.values() if g.status == GoalStatus.FAILED),
            "total_plans": len(self.plans),
            "available_resources": self.available_resources.copy(),
        }
