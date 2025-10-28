"""
Orchestration API Router
=========================

REST API endpoints for multi-agent orchestration:
- Intent routing
- Task graph execution
- Agent status and management
- Handoff monitoring
- Resource governance

Endpoints:
- POST /api/orchestration/route - Route user intent to agents
- POST /api/orchestration/execute - Execute task graph
- GET /api/orchestration/agents - List all agents
- GET /api/orchestration/agents/{agent_id}/status - Get agent status
- POST /api/orchestration/handoff - Initiate agent handoff
- GET /api/orchestration/resources - Get resource usage
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

# Import orchestration components
from core.orchestration import (
    get_intent_router,
    IntentRouter,
    IntentCategory,
    TaskGraph,
    TaskType,
    get_handoff_manager,
    HandoffManager,
    HandoffTrigger,
    get_resource_governor,
    ResourceGovernor,
    ResourceType,
)

# Import agents
from core.agents import (
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    OperatorAgent,
    FinanceAgent,
    SafetyAgent,
    AgentStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestration", tags=["orchestration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class IntentRouteRequest(BaseModel):
    """Request to route user intent"""
    user_input: str = Field(..., description="User's natural language input")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    user_id: Optional[str] = Field(default=None, description="User identifier")


class IntentRouteResponse(BaseModel):
    """Response from intent routing"""
    request_id: str
    intent: str
    confidence: float
    agent_role: str
    context: Dict[str, Any]
    is_safe: bool
    timestamp: str


class TaskGraphRequest(BaseModel):
    """Request to execute task graph"""
    goal: str = Field(..., description="High-level goal to accomplish")
    parallel: bool = Field(default=True, description="Enable parallel execution")
    stop_on_error: bool = Field(default=True, description="Stop on first error")


class TaskGraphResponse(BaseModel):
    """Response from task graph execution"""
    graph_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time_sec: float
    results: Dict[str, Any]
    errors: Dict[str, str]
    timestamp: str


class AgentStatusResponse(BaseModel):
    """Agent status response"""
    agent_id: str
    role: str
    status: str
    capabilities: List[str]
    uptime_sec: float
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float


class HandoffRequest(BaseModel):
    """Request to initiate agent handoff"""
    from_agent: str
    to_agent: str
    trigger: str
    task_state: Dict[str, Any]
    user_id: Optional[str] = None


class ResourceStatusResponse(BaseModel):
    """Resource usage status"""
    agent_id: str
    resources: Dict[str, Any]
    timestamp: str


# =============================================================================
# Agent Registry (In-memory for demo)
# =============================================================================

class AgentRegistry:
    """Registry of active agents"""

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize default agents"""
        agents_to_create = [
            PlannerAgent(agent_id="planner"),
            ResearcherAgent(agent_id="researcher"),
            CoderAgent(agent_id="coder"),
            OperatorAgent(agent_id="operator"),
            FinanceAgent(agent_id="finance"),
            SafetyAgent(agent_id="safety"),
        ]

        for agent in agents_to_create:
            agent.start()
            self.agents[agent.agent_id] = agent
            logger.info(f"Registered agent: {agent.agent_id} ({agent.role})")

    def get_agent(self, agent_id: str):
        """Get agent by ID"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent not found: {agent_id}")
        return self.agents[agent_id]

    def get_agent_by_role(self, role: str):
        """Get agent by role"""
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        raise ValueError(f"No agent with role: {role}")

    def list_agents(self) -> List[Any]:
        """List all agents"""
        return list(self.agents.values())


# Global agent registry
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """Get singleton agent registry"""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/route", response_model=IntentRouteResponse)
async def route_intent(
    request: IntentRouteRequest,
    intent_router: IntentRouter = Depends(get_intent_router),
    agent_registry: AgentRegistry = Depends(get_agent_registry)
):
    """
    Route user intent to appropriate agent.

    This endpoint:
    1. Classifies user intent
    2. Validates safety
    3. Routes to appropriate agent
    4. Returns routing decision

    Example:
    ```
    POST /api/orchestration/route
    {
        "user_input": "write a function to sort an array",
        "context": {"language": "python"}
    }
    ```
    """
    try:
        logger.info(f"Routing intent: {request.user_input[:100]}")

        # Classify intent
        intent_result = intent_router.classify(
            user_input=request.user_input,
            context=request.context or {}
        )

        # Validate safety
        is_safe = intent_router.validate_intent(intent_result)

        # Get target agent
        try:
            target_agent = agent_registry.get_agent_by_role(intent_result.agent_role)
        except ValueError:
            # Fallback to planner if agent not found
            target_agent = agent_registry.get_agent("planner")

        response = IntentRouteResponse(
            request_id=intent_result.request_id,
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            agent_role=target_agent.role,
            context=intent_result.context,
            is_safe=is_safe,
            timestamp=intent_result.timestamp
        )

        logger.info(
            f"Intent routed: {intent_result.intent.value} -> {target_agent.role} "
            f"(confidence: {intent_result.confidence:.2f})"
        )

        return response

    except Exception as e:
        logger.error(f"Intent routing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent routing failed: {str(e)}"
        )


@router.post("/execute", response_model=TaskGraphResponse)
async def execute_task_graph(
    request: TaskGraphRequest,
    agent_registry: AgentRegistry = Depends(get_agent_registry)
):
    """
    Execute task graph for goal.

    This endpoint:
    1. Uses planner to decompose goal
    2. Creates task graph
    3. Executes tasks using appropriate agents
    4. Returns execution results

    Example:
    ```
    POST /api/orchestration/execute
    {
        "goal": "research and implement feature X",
        "parallel": true
    }
    ```
    """
    try:
        logger.info(f"Executing task graph for goal: {request.goal}")

        # Get planner agent
        planner = agent_registry.get_agent("planner")

        # Create plan
        plan_result = planner.execute_task({
            "type": "plan",
            "params": {"goal": request.goal}
        })

        if not plan_result.success:
            raise ValueError(f"Planning failed: {plan_result.error}")

        # Create task graph
        graph = TaskGraph()

        # Add tasks from plan
        tasks = plan_result.result.get("tasks", [])

        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            task_type = task.get("type", "generic")

            # Determine which agent should handle this task
            if task_type in ["research", "gather", "analyze"]:
                agent_role = "researcher"
            elif task_type in ["design", "implement", "test", "review"]:
                agent_role = "coder"
            elif task_type in ["execute", "deploy", "monitor"]:
                agent_role = "operator"
            else:
                agent_role = "planner"

            # Create task handler
            def create_handler(agent_role, task):
                def handler(params):
                    agent = agent_registry.get_agent_by_role(agent_role)
                    result = agent.execute_task({
                        "type": task.get("type"),
                        "params": params
                    })
                    return result.result if result.success else None
                return handler

            # Add task to graph
            dependencies = [f"task_{i-1}"] if i > 0 else []

            graph.add_task(
                task_id=task_id,
                name=task.get("description", f"Task {i}"),
                handler=create_handler(agent_role, task),
                dependencies=dependencies,
                params={}
            )

        # Execute graph
        execution_result = graph.execute(
            enable_parallel=request.parallel,
            stop_on_error=request.stop_on_error
        )

        response = TaskGraphResponse(
            graph_id=execution_result.graph_id,
            status=execution_result.status,
            total_tasks=execution_result.total_tasks,
            completed_tasks=execution_result.completed_tasks,
            failed_tasks=execution_result.failed_tasks,
            execution_time_sec=execution_result.execution_time_sec,
            results=execution_result.results,
            errors=execution_result.errors,
            timestamp=execution_result.timestamp
        )

        logger.info(
            f"Task graph executed: {execution_result.status} "
            f"({execution_result.completed_tasks}/{execution_result.total_tasks} tasks)"
        )

        return response

    except Exception as e:
        logger.error(f"Task graph execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task graph execution failed: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentStatusResponse])
async def list_agents(agent_registry: AgentRegistry = Depends(get_agent_registry)):
    """
    List all registered agents.

    Returns status information for all agents in the system.
    """
    try:
        agents = agent_registry.list_agents()

        response = []
        for agent in agents:
            status = agent.get_status()
            response.append(AgentStatusResponse(
                agent_id=status["agent_id"],
                role=status["role"],
                status=status["status"],
                capabilities=status["capabilities"],
                uptime_sec=status["uptime_sec"],
                total_tasks=status["total_tasks"],
                successful_tasks=status["successful_tasks"],
                failed_tasks=status["failed_tasks"],
                success_rate=status["success_rate"]
            ))

        return response

    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@router.get("/agents/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str,
    agent_registry: AgentRegistry = Depends(get_agent_registry)
):
    """
    Get status for specific agent.

    Returns detailed status information including:
    - Current status (idle, busy, etc.)
    - Task statistics
    - Resource usage
    - Uptime
    """
    try:
        agent = agent_registry.get_agent(agent_id)
        status = agent.get_status()

        return AgentStatusResponse(
            agent_id=status["agent_id"],
            role=status["role"],
            status=status["status"],
            capabilities=status["capabilities"],
            uptime_sec=status["uptime_sec"],
            total_tasks=status["total_tasks"],
            successful_tasks=status["successful_tasks"],
            failed_tasks=status["failed_tasks"],
            success_rate=status["success_rate"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent status: {str(e)}"
        )


@router.post("/handoff")
async def initiate_handoff(
    request: HandoffRequest,
    handoff_manager: HandoffManager = Depends(get_handoff_manager)
):
    """
    Initiate handoff between agents.

    Transfers task state and context from one agent to another.
    """
    try:
        # Parse trigger
        try:
            trigger = HandoffTrigger(request.trigger)
        except ValueError:
            trigger = HandoffTrigger.AGENT_REQUEST

        # Initiate handoff
        context = handoff_manager.initiate_handoff(
            from_agent=request.from_agent,
            to_agent=request.to_agent,
            trigger=trigger,
            task_state=request.task_state,
            user_id=request.user_id
        )

        # Complete handoff immediately for demo
        result = handoff_manager.complete_handoff(
            handoff_id=context.handoff_id,
            success=True
        )

        return {
            "handoff_id": result.handoff_id,
            "status": result.status.value,
            "from_agent": result.from_agent,
            "to_agent": result.to_agent,
            "success": result.success,
            "timestamp": result.initiated_at
        }

    except Exception as e:
        logger.error(f"Handoff failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Handoff failed: {str(e)}"
        )


@router.get("/resources", response_model=List[ResourceStatusResponse])
async def get_resource_status(
    agent_id: Optional[str] = None,
    resource_governor: ResourceGovernor = Depends(get_resource_governor)
):
    """
    Get resource usage status.

    Returns resource usage for all agents or specific agent.
    """
    try:
        if agent_id:
            # Get specific agent status
            status = resource_governor.get_agent_status(agent_id)
            return [ResourceStatusResponse(
                agent_id=status["agent_id"],
                resources=status["resources"],
                timestamp=status["timestamp"]
            )]
        else:
            # Get all agents status
            all_status = resource_governor.get_all_agent_status()
            return [
                ResourceStatusResponse(
                    agent_id=status["agent_id"],
                    resources=status["resources"],
                    timestamp=status["timestamp"]
                )
                for status in all_status
            ]

    except Exception as e:
        logger.error(f"Failed to get resource status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource status: {str(e)}"
        )


@router.get("/stats")
async def get_orchestration_stats(
    intent_router: IntentRouter = Depends(get_intent_router),
    handoff_manager: HandoffManager = Depends(get_handoff_manager),
    resource_governor: ResourceGovernor = Depends(get_resource_governor),
    agent_registry: AgentRegistry = Depends(get_agent_registry)
):
    """
    Get orchestration system statistics.

    Returns overall system statistics including:
    - Intent routing stats
    - Handoff stats
    - Resource usage stats
    - Agent stats
    """
    try:
        return {
            "intent_router": intent_router.get_stats(),
            "handoffs": handoff_manager.get_handoff_stats(),
            "resources": resource_governor.get_stats(),
            "agents": {
                "total_agents": len(agent_registry.list_agents()),
                "agents_by_status": {
                    status.value: sum(
                        1 for a in agent_registry.list_agents()
                        if a.status == status
                    )
                    for status in AgentStatus
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
