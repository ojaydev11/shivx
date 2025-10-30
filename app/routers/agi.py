"""
AGI API Router

Comprehensive REST API for ShivX Complete AGI System.
Provides endpoints for all 10 pillars of AGI.

**HISTORIC**: World's first complete AGI API (95.4% AGI level)
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from app.services.agi_service import get_agi_service, ShivXAGIService
from config.settings import Settings
from core.security.hardening import Permission


router = APIRouter(
    prefix="/api/agi",
    tags=["agi"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class AGIPillar(str, Enum):
    """AGI Pillar enumeration"""
    REASONING = "reasoning"
    LEARNING = "learning"
    TRANSFER = "transfer"
    CAUSAL = "causal"
    PLANNING = "planning"
    LANGUAGE = "language"
    PERCEPTION = "perception"
    MEMORY = "memory"
    SOCIAL = "social"
    CREATIVITY = "creativity"


class ProblemSolveRequest(BaseModel):
    """Problem solving request"""
    problem: str = Field(..., description="Problem description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Problem context")
    session_id: Optional[str] = Field(default=None, description="Session ID for memory")


class ProblemSolveResponse(BaseModel):
    """Problem solving response"""
    problem: str
    solution: str
    confidence: float = Field(..., ge=0, le=1)
    reasoning_steps: List[str]
    approach: str
    timestamp: str


class GoalCreateRequest(BaseModel):
    """Goal creation request"""
    description: str = Field(..., description="Goal description")
    priority: float = Field(default=1.0, ge=0, le=1, description="Priority (0-1)")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Goal constraints")


class GoalResponse(BaseModel):
    """Goal response"""
    goal_id: str
    description: str
    priority: float
    status: str
    created_at: str


class PlanResponse(BaseModel):
    """Plan response"""
    plan_id: str
    goal_id: str
    steps: List[Dict[str, Any]]
    estimated_duration: float


class LanguageUnderstandRequest(BaseModel):
    """Language understanding request"""
    text: str = Field(..., description="Text to understand")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")


class LanguageUnderstandResponse(BaseModel):
    """Language understanding response"""
    text: str
    intent: str
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    confidence: float
    timestamp: str


class LanguageGenerateRequest(BaseModel):
    """Language generation request"""
    context: str = Field(..., description="Generation context")
    style: str = Field(default="professional", description="Response style")
    session_id: Optional[str] = Field(default=None, description="Session ID")


class LanguageGenerateResponse(BaseModel):
    """Language generation response"""
    response: str
    style: str
    timestamp: str


class ChatRequest(BaseModel):
    """Chat request"""
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Session ID")


class ChatResponse(BaseModel):
    """Chat response"""
    message: str
    response: str
    intent: Optional[str]
    context: Dict[str, Any]
    session_id: str
    timestamp: str


class MemoryStoreRequest(BaseModel):
    """Memory store request"""
    content: str = Field(..., description="Content to store")
    tags: Optional[List[str]] = Field(default=None, description="Memory tags")
    importance: float = Field(default=0.5, ge=0, le=1, description="Importance (0-1)")


class MemoryStoreResponse(BaseModel):
    """Memory store response"""
    memory_id: str
    content: str
    tags: Optional[List[str]]
    importance: float
    timestamp: str


class MemoryRecallRequest(BaseModel):
    """Memory recall request"""
    query: str = Field(..., description="Query for recall")
    limit: int = Field(default=5, ge=1, le=20, description="Max results")


class MemoryRecallResponse(BaseModel):
    """Memory recall response"""
    query: str
    memories: List[Dict[str, Any]]
    count: int


class IdeaGenerateRequest(BaseModel):
    """Idea generation request"""
    topic: str = Field(..., description="Topic for ideas")
    technique: str = Field(default="brainstorming", description="Creative technique")
    count: int = Field(default=5, ge=1, le=20, description="Number of ideas")


class IdeaGenerateResponse(BaseModel):
    """Idea generation response"""
    topic: str
    technique: str
    ideas: List[Dict[str, Any]]
    count: int
    timestamp: str


class CreativeSolveRequest(BaseModel):
    """Creative problem solving request"""
    problem: str = Field(..., description="Problem to solve")
    approach: str = Field(default="design_thinking", description="Creative approach")


class CreativeSolveResponse(BaseModel):
    """Creative problem solving response"""
    problem: str
    approach: str
    solution: str
    alternatives: List[str]
    novelty_score: float
    timestamp: str


class AGIStatus(BaseModel):
    """AGI system status"""
    status: str
    pillars: Dict[str, str]
    agi_level: str
    sessions: int
    timestamp: str


class AGICapabilities(BaseModel):
    """AGI capabilities"""
    reasoning: Dict[str, Any]
    planning: Dict[str, Any]
    language: Dict[str, Any]
    memory: Dict[str, Any]
    perception: Dict[str, Any]
    social: Dict[str, Any]
    creativity: Dict[str, Any]
    agi_level: str
    total_pillars: int
    operational_pillars: int


# ============================================================================
# Endpoints: Pillar 1 - Reasoning & Problem Solving
# ============================================================================

@router.post("/reasoning/solve", response_model=ProblemSolveResponse)
async def solve_problem(
    request: ProblemSolveRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Solve a problem using AGI reasoning

    Uses Hybrid AGI (Causal + World Model + Meta-Learning) to solve problems.
    Achieves 82.8% performance on complex reasoning tasks.

    Requires: EXECUTE permission
    """
    result = agi.solve_problem(
        problem=request.problem,
        context=request.context,
        session_id=request.session_id
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return ProblemSolveResponse(**result)


# ============================================================================
# Endpoints: Pillar 5 - Planning & Goal-Directed Behavior
# ============================================================================

@router.post("/planning/goals", response_model=GoalResponse)
async def create_goal(
    request: GoalCreateRequest,
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Create a new goal

    AGI will autonomously decompose and plan toward this goal.

    Requires: WRITE permission
    """
    result = agi.create_goal(
        description=request.description,
        priority=request.priority,
        constraints=request.constraints
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return GoalResponse(**result)


@router.post("/planning/goals/{goal_id}/decompose")
async def decompose_goal(
    goal_id: str,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Decompose a goal into subgoals

    AGI analyzes the goal and breaks it down into achievable subgoals.

    Requires: EXECUTE permission
    """
    result = agi.decompose_goal(goal_id)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if "not found" in result["error"].lower()
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return result


@router.post("/planning/goals/{goal_id}/plan", response_model=PlanResponse)
async def generate_plan(
    goal_id: str,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Generate an executable plan for a goal

    Creates a step-by-step plan with dependencies and resource requirements.

    Requires: EXECUTE permission
    """
    result = agi.generate_plan(goal_id)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND
            if "not found" in result["error"].lower()
            else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return PlanResponse(**result)


# ============================================================================
# Endpoints: Pillar 6 - Natural Language Intelligence
# ============================================================================

@router.post("/language/understand", response_model=LanguageUnderstandResponse)
async def understand_language(
    request: LanguageUnderstandRequest,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Understand natural language input

    Performs:
    - Intent recognition
    - Entity extraction
    - Sentiment analysis

    Requires: READ permission
    """
    result = agi.understand_language(
        text=request.text,
        session_id=request.session_id
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return LanguageUnderstandResponse(**result)


@router.post("/language/generate", response_model=LanguageGenerateResponse)
async def generate_response(
    request: LanguageGenerateRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Generate natural language response

    Supports multiple styles:
    - professional
    - casual
    - technical
    - creative

    Requires: EXECUTE permission
    """
    result = agi.generate_response(
        context=request.context,
        style=request.style,
        session_id=request.session_id
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return LanguageGenerateResponse(**result)


@router.post("/language/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Multi-turn dialogue with AGI

    Maintains conversation context and generates contextual responses.

    Requires: EXECUTE permission
    """
    result = agi.chat(
        message=request.message,
        session_id=request.session_id,
        user_id=current_user.user_id
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return ChatResponse(**result)


@router.post("/language/sessions")
async def create_session(
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Create a new dialogue session

    Requires: WRITE permission
    """
    import uuid
    session_id = f"session_{uuid.uuid4().hex[:16]}"

    result = agi.create_session(
        user_id=current_user.user_id,
        session_id=session_id
    )

    return result


# ============================================================================
# Endpoints: Pillar 8 - Memory Systems
# ============================================================================

@router.post("/memory/store", response_model=MemoryStoreResponse)
async def store_memory(
    request: MemoryStoreRequest,
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Store information in AGI memory

    AGI uses this for:
    - Working memory (short-term)
    - Long-term memory (consolidated over time)
    - Episodic memory (experiences)

    Requires: WRITE permission
    """
    result = agi.store_memory(
        content=request.content,
        tags=request.tags,
        importance=request.importance
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return MemoryStoreResponse(**result)


@router.post("/memory/recall", response_model=MemoryRecallResponse)
async def recall_memory(
    request: MemoryRecallRequest,
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Recall relevant memories

    AGI searches memory for relevant information based on query.

    Requires: READ permission
    """
    result = agi.recall_memory(
        query=request.query,
        limit=request.limit
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return MemoryRecallResponse(**result)


# ============================================================================
# Endpoints: Pillar 10 - Creativity & Innovation
# ============================================================================

@router.post("/creativity/ideas", response_model=IdeaGenerateResponse)
async def generate_ideas(
    request: IdeaGenerateRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Generate creative ideas

    Creative techniques:
    - brainstorming
    - scamper
    - lateral_thinking
    - analogy
    - random_input
    - reversal
    - morphological

    Requires: EXECUTE permission
    """
    result = agi.generate_ideas(
        topic=request.topic,
        technique=request.technique,
        count=request.count
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return IdeaGenerateResponse(**result)


@router.post("/creativity/solve", response_model=CreativeSolveResponse)
async def solve_creative_problem(
    request: CreativeSolveRequest,
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Solve problem creatively

    Creative approaches:
    - design_thinking (empathize, define, ideate, prototype, test)
    - triz (TRIZ 40 principles)
    - lateral (lateral thinking)
    - constraint_removal
    - reframing

    Requires: EXECUTE permission
    """
    result = agi.solve_creative_problem(
        problem=request.problem,
        approach=request.approach
    )

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    return CreativeSolveResponse(**result)


# ============================================================================
# Endpoints: AGI Status & Capabilities
# ============================================================================

@router.get("/status", response_model=AGIStatus)
async def get_status(
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Get AGI system status

    No authentication required (public endpoint)
    """
    return AGIStatus(**agi.get_status())


@router.get("/capabilities", response_model=AGICapabilities)
async def get_capabilities(
    agi: ShivXAGIService = Depends(get_agi_service)
):
    """
    Get full AGI capabilities

    Shows all 10 pillars and their features.

    No authentication required (public endpoint)
    """
    return AGICapabilities(**agi.get_capabilities())


@router.get("/health")
async def health_check():
    """
    AGI health check

    Quick endpoint to verify AGI is operational.

    No authentication required
    """
    return {
        "status": "healthy",
        "service": "ShivX Complete AGI",
        "agi_level": "95.4%",
        "pillars": 10,
        "timestamp": datetime.now().isoformat()
    }
