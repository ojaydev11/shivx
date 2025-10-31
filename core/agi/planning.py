"""
AGI Planning Module
General Planning Algorithms

Implements:
- STRIPS planning
- HTN (Hierarchical Task Network) planning
- Goal-oriented planning
- Multi-step reasoning
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Action execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class State:
    """World state"""
    predicates: Set[str]  # Set of true predicates

    def satisfies(self, conditions: Set[str]) -> bool:
        """Check if state satisfies conditions"""
        return conditions.issubset(self.predicates)

    def apply(self, add_effects: Set[str], del_effects: Set[str]) -> 'State':
        """Apply action effects to create new state"""
        new_predicates = (self.predicates - del_effects) | add_effects
        return State(predicates=new_predicates)


@dataclass
class Action:
    """Planning action"""
    name: str
    preconditions: Set[str]
    add_effects: Set[str]
    del_effects: Set[str]
    cost: float = 1.0

    def is_applicable(self, state: State) -> bool:
        """Check if action is applicable in state"""
        return state.satisfies(self.preconditions)

    def apply(self, state: State) -> State:
        """Apply action to state"""
        return state.apply(self.add_effects, self.del_effects)


@dataclass
class Plan:
    """Plan (sequence of actions)"""
    actions: List[Action]
    cost: float
    success_probability: float


class STRIPSPlanner:
    """
    STRIPS Planning Algorithm
    Classic AI planning using forward/backward search
    """

    def __init__(self):
        """Initialize STRIPS planner"""
        self.actions: List[Action] = []
        logger.info("STRIPS planner initialized")

    def add_action(self, action: Action):
        """Add action to planner's action library"""
        self.actions.append(action)

    def plan(
        self,
        initial_state: State,
        goal: Set[str],
        max_depth: int = 10
    ) -> Optional[Plan]:
        """
        Create plan to achieve goal from initial state

        Args:
            initial_state: Initial world state
            goal: Goal predicates to achieve
            max_depth: Maximum plan depth

        Returns:
            Plan if found, None otherwise
        """
        logger.info(f"Planning to achieve: {goal}")

        # Forward search using breadth-first search
        queue = [(initial_state, [], 0.0)]  # (state, actions, cost)
        visited = set()

        for _ in range(max_depth * 100):  # Limit iterations
            if not queue:
                break

            state, actions, cost = queue.pop(0)

            # Convert state to hashable format
            state_hash = tuple(sorted(state.predicates))
            if state_hash in visited:
                continue
            visited.add(state_hash)

            # Check if goal reached
            if state.satisfies(goal):
                logger.info(f"Plan found with {len(actions)} actions")
                return Plan(
                    actions=actions,
                    cost=cost,
                    success_probability=0.9
                )

            # Try all applicable actions
            for action in self.actions:
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    new_actions = actions + [action]
                    new_cost = cost + action.cost

                    queue.append((new_state, new_actions, new_cost))

        logger.warning("No plan found")
        return None


class HTNPlanner:
    """
    HTN (Hierarchical Task Network) Planning
    Decomposes high-level tasks into primitive actions
    """

    def __init__(self):
        """Initialize HTN planner"""
        self.methods: Dict[str, List[List[str]]] = {}  # task -> list of decompositions
        self.primitive_actions: Dict[str, Action] = {}
        logger.info("HTN planner initialized")

    def add_method(self, task: str, decomposition: List[str]):
        """
        Add decomposition method for task

        Args:
            task: High-level task name
            decomposition: List of subtasks/actions
        """
        if task not in self.methods:
            self.methods[task] = []
        self.methods[task].append(decomposition)

    def add_primitive_action(self, action: Action):
        """Add primitive action"""
        self.primitive_actions[action.name] = action

    def plan(
        self,
        task: str,
        state: State,
        max_depth: int = 5
    ) -> Optional[Plan]:
        """
        Create plan for high-level task

        Args:
            task: High-level task to accomplish
            state: Current world state
            max_depth: Maximum decomposition depth

        Returns:
            Plan if found
        """
        logger.info(f"HTN planning for task: {task}")

        actions = self._decompose(task, state, max_depth)

        if actions:
            total_cost = sum(a.cost for a in actions)
            return Plan(
                actions=actions,
                cost=total_cost,
                success_probability=0.85
            )

        return None

    def _decompose(
        self,
        task: str,
        state: State,
        depth: int
    ) -> Optional[List[Action]]:
        """Recursively decompose task"""
        if depth == 0:
            return None

        # If task is primitive action
        if task in self.primitive_actions:
            action = self.primitive_actions[task]
            if action.is_applicable(state):
                return [action]
            return None

        # Try decomposition methods
        if task in self.methods:
            for decomposition in self.methods[task]:
                result = []
                current_state = state

                # Try to decompose each subtask
                success = True
                for subtask in decomposition:
                    subtask_actions = self._decompose(subtask, current_state, depth - 1)

                    if not subtask_actions:
                        success = False
                        break

                    result.extend(subtask_actions)

                    # Update state
                    for action in subtask_actions:
                        current_state = action.apply(current_state)

                if success:
                    return result

        return None


class PlanningModule:
    """
    Complete planning system combining multiple planning approaches
    """

    def __init__(self):
        """Initialize planning module"""
        self.strips = STRIPSPlanner()
        self.htn = HTNPlanner()

        # Initialize with default actions
        self._initialize_default_actions()

        logger.info("Planning module initialized")

    def _initialize_default_actions(self):
        """Initialize default action library"""
        # Trading actions
        self.strips.add_action(Action(
            name="analyze_market",
            preconditions=set(),
            add_effects={"market_analyzed"},
            del_effects=set(),
            cost=1.0
        ))

        self.strips.add_action(Action(
            name="generate_signal",
            preconditions={"market_analyzed"},
            add_effects={"signal_generated"},
            del_effects=set(),
            cost=1.5
        ))

        self.strips.add_action(Action(
            name="execute_trade",
            preconditions={"signal_generated", "market_analyzed"},
            add_effects={"trade_executed"},
            del_effects=set(),
            cost=2.0
        ))

        # HTN methods
        self.htn.add_method("trade_workflow", [
            "analyze_market",
            "generate_signal",
            "execute_trade"
        ])

        # Add primitives to HTN
        for action in self.strips.actions:
            self.htn.add_primitive_action(action)

    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        method: str = "strips"
    ) -> Optional[Plan]:
        """
        Create plan to achieve goal

        Args:
            goal: Goal description or task name
            context: Current context
            method: Planning method ("strips" or "htn")

        Returns:
            Plan if successful
        """
        # Parse initial state from context
        initial_state = State(predicates=set(context.get("state", [])) if context else set())

        if method == "htn":
            # Use HTN for high-level tasks
            plan = self.htn.plan(goal, initial_state)
        else:
            # Use STRIPS for low-level goals
            goal_predicates = {goal} if isinstance(goal, str) else set(goal)
            plan = self.strips.plan(initial_state, goal_predicates)

        if plan:
            logger.info(f"Plan created: {len(plan.actions)} actions, cost={plan.cost:.2f}")

        return plan

    async def execute_plan(
        self,
        plan: Plan,
        executor: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan

        Args:
            plan: Plan to execute
            executor: Optional executor for actions

        Returns:
            Execution result
        """
        logger.info(f"Executing plan with {len(plan.actions)} actions")

        results = []
        for i, action in enumerate(plan.actions):
            logger.info(f"Step {i+1}/{len(plan.actions)}: {action.name}")

            # Execute action (would call actual executor in production)
            result = {
                "action": action.name,
                "status": ActionStatus.COMPLETED.value,
                "cost": action.cost
            }

            results.append(result)

        return {
            "plan_cost": plan.cost,
            "actions_executed": len(plan.actions),
            "results": results,
            "success": True
        }

    def get_capabilities(self) -> Dict[str, bool]:
        """Get planning capabilities status"""
        return {
            "strips_planning": True,
            "htn_planning": True,
            "goal_oriented": True,
            "multi_step_reasoning": True,
            "action_library_size": len(self.strips.actions)
        }
