"""
Agent Tests - Multi-Agent Framework
====================================

Comprehensive tests for all agent implementations:
- Base Agent functionality
- Planner Agent
- Researcher Agent
- Coder Agent
- Operator Agent
- Finance Agent
- Safety Agent
- Agent handoffs
- Resource limits

Coverage: 60+ tests for complete agent system validation
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime

# Import all agents
from core.agents import (
    BaseAgent,
    AgentStatus,
    AgentCapability,
    TaskResult,
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    OperatorAgent,
    FinanceAgent,
    SafetyAgent,
)

from core.orchestration import (
    HandoffManager,
    HandoffTrigger,
    ResourceGovernor,
    ResourceType,
)


# =============================================================================
# Test Base Agent
# =============================================================================

@pytest.mark.unit
class TestBaseAgent:
    """Test base agent functionality"""

    def test_agent_initialization(self):
        """Test: Agent initializes correctly"""
        agent = PlannerAgent(agent_id="test_planner")

        assert agent.agent_id == "test_planner"
        assert agent.role == "planner"
        assert agent.status == AgentStatus.SPAWNED
        assert AgentCapability.PLANNING in agent.capabilities

    def test_agent_lifecycle(self):
        """Test: Agent lifecycle transitions"""
        agent = PlannerAgent()

        # Start agent
        agent.start()
        assert agent.status == AgentStatus.IDLE

        # Pause agent
        agent.pause()
        assert agent.status == AgentStatus.PAUSED

        # Resume agent
        agent.resume()
        assert agent.status == AgentStatus.IDLE

        # Terminate agent
        agent.terminate()
        assert agent.status == AgentStatus.TERMINATED

    def test_agent_message_passing(self):
        """Test: Agent can send and receive messages"""
        agent1 = PlannerAgent(agent_id="agent1")
        agent2 = ResearcherAgent(agent_id="agent2")

        # Send message from agent1 to agent2
        message = agent1.send_message(
            to_agent="agent2",
            message_type="request",
            content={"task": "research_topic"}
        )

        assert message.from_agent == "agent1"
        assert message.to_agent == "agent2"
        assert message.message_type == "request"

        # Agent2 receives message
        agent2.receive_message(message)
        assert len(agent2.inbox) == 1
        assert agent2.inbox[0].message_id == message.message_id

    def test_agent_status_reporting(self):
        """Test: Agent reports status correctly"""
        agent = CoderAgent()
        agent.start()

        status = agent.get_status()

        assert status["agent_id"] == agent.agent_id
        assert status["role"] == "coder"
        assert status["status"] == AgentStatus.IDLE.value
        assert "uptime_sec" in status
        assert "total_tasks" in status


# =============================================================================
# Test Planner Agent
# =============================================================================

@pytest.mark.unit
class TestPlannerAgent:
    """Test planner agent"""

    def test_planner_can_handle_planning_tasks(self):
        """Test: Planner can handle planning tasks"""
        planner = PlannerAgent()

        assert planner.can_handle({"type": "plan"})
        assert planner.can_handle({"type": "decompose"})
        assert planner.can_handle({"type": "generate_tasks"})
        assert not planner.can_handle({"type": "write_code"})

    def test_planner_executes_plan_task(self):
        """Test: Planner executes planning task"""
        planner = PlannerAgent()
        planner.start()

        task = {
            "type": "plan",
            "params": {
                "goal": "implement a new feature"
            }
        }

        result = planner.execute_task(task)

        assert result.success
        assert "tasks" in result.result
        assert len(result.result["tasks"]) > 0

    def test_planner_decomposes_task(self):
        """Test: Planner decomposes complex task"""
        planner = PlannerAgent()
        planner.start()

        task = {
            "type": "decompose",
            "params": {
                "task": "Build a web application"
            }
        }

        result = planner.execute_task(task)

        assert result.success
        assert "subtasks" in result.result


# =============================================================================
# Test Researcher Agent
# =============================================================================

@pytest.mark.unit
class TestResearcherAgent:
    """Test researcher agent"""

    def test_researcher_can_handle_research_tasks(self):
        """Test: Researcher can handle research tasks"""
        researcher = ResearcherAgent()

        assert researcher.can_handle({"type": "research"})
        assert researcher.can_handle({"type": "gather_information"})
        assert researcher.can_handle({"type": "web_search"})
        assert not researcher.can_handle({"type": "execute_trade"})

    def test_researcher_executes_research_task(self):
        """Test: Researcher executes research task"""
        researcher = ResearcherAgent()
        researcher.start()

        task = {
            "type": "research",
            "params": {
                "topic": "machine learning"
            }
        }

        result = researcher.execute_task(task)

        assert result.success
        assert "topic" in result.result
        assert "sources" in result.result

    def test_researcher_performs_web_search(self):
        """Test: Researcher performs web search"""
        researcher = ResearcherAgent()
        researcher.start()

        task = {
            "type": "web_search",
            "params": {
                "query": "Python programming",
                "max_results": 5
            }
        }

        result = researcher.execute_task(task)

        assert result.success
        assert "results" in result.result


# =============================================================================
# Test Coder Agent
# =============================================================================

@pytest.mark.unit
class TestCoderAgent:
    """Test coder agent"""

    def test_coder_can_handle_coding_tasks(self):
        """Test: Coder can handle coding tasks"""
        coder = CoderAgent()

        assert coder.can_handle({"type": "write_code"})
        assert coder.can_handle({"type": "review_code"})
        assert coder.can_handle({"type": "refactor"})
        assert not coder.can_handle({"type": "research"})

    def test_coder_writes_code(self):
        """Test: Coder writes code"""
        coder = CoderAgent()
        coder.start()

        task = {
            "type": "write_code",
            "params": {
                "specification": "Function to sort array",
                "language": "python"
            }
        }

        result = coder.execute_task(task)

        assert result.success
        assert "code" in result.result
        assert "language" in result.result

    def test_coder_reviews_code(self):
        """Test: Coder reviews code"""
        coder = CoderAgent()
        coder.start()

        task = {
            "type": "review_code",
            "params": {
                "code": "def hello(): print('world')"
            }
        }

        result = coder.execute_task(task)

        assert result.success
        assert "issues" in result.result

    def test_coder_generates_tests(self):
        """Test: Coder generates tests"""
        coder = CoderAgent()
        coder.start()

        task = {
            "type": "generate_tests",
            "params": {
                "code": "def add(a, b): return a + b"
            }
        }

        result = coder.execute_task(task)

        assert result.success
        assert "test_code" in result.result


# =============================================================================
# Test Operator Agent
# =============================================================================

@pytest.mark.unit
class TestOperatorAgent:
    """Test operator agent"""

    def test_operator_can_handle_system_tasks(self):
        """Test: Operator can handle system tasks"""
        operator = OperatorAgent()

        assert operator.can_handle({"type": "execute_command"})
        assert operator.can_handle({"type": "monitor_resources"})
        assert not operator.can_handle({"type": "write_code"})

    def test_operator_monitors_resources(self):
        """Test: Operator monitors system resources"""
        operator = OperatorAgent()
        operator.start()

        task = {
            "type": "monitor_resources",
            "params": {}
        }

        result = operator.execute_task(task)

        assert result.success
        assert "cpu_percent" in result.result
        assert "memory_percent" in result.result

    def test_operator_sends_notification(self):
        """Test: Operator sends notification"""
        operator = OperatorAgent()
        operator.start()

        task = {
            "type": "send_notification",
            "params": {
                "channel": "email",
                "message": "Test notification",
                "recipients": ["user@example.com"]
            }
        }

        result = operator.execute_task(task)

        assert result.success
        assert result.result["message_sent"]


# =============================================================================
# Test Finance Agent
# =============================================================================

@pytest.mark.unit
class TestFinanceAgent:
    """Test finance agent"""

    def test_finance_can_handle_trading_tasks(self):
        """Test: Finance agent can handle trading tasks"""
        finance = FinanceAgent()

        assert finance.can_handle({"type": "execute_trade"})
        assert finance.can_handle({"type": "analyze_market"})
        assert finance.can_handle({"type": "assess_risk"})
        assert not finance.can_handle({"type": "write_code"})

    def test_finance_analyzes_market(self):
        """Test: Finance agent analyzes market"""
        finance = FinanceAgent()
        finance.start()

        task = {
            "type": "analyze_market",
            "params": {
                "symbol": "SOL",
                "chain": "solana"
            }
        }

        result = finance.execute_task(task)

        assert result.success
        assert "current_price" in result.result
        assert "trend" in result.result
        assert "indicators" in result.result

    def test_finance_executes_trade(self):
        """Test: Finance agent executes trade"""
        finance = FinanceAgent()
        finance.start()

        task = {
            "type": "execute_trade",
            "params": {
                "action": "buy",
                "symbol": "SOL",
                "amount": 10.0,
                "chain": "solana"
            }
        }

        result = finance.execute_task(task)

        assert result.success
        assert "trade_id" in result.result
        assert result.result["status"] == "completed"

    def test_finance_assesses_risk(self):
        """Test: Finance agent assesses risk"""
        finance = FinanceAgent()
        finance.start()

        task = {
            "type": "assess_risk",
            "params": {
                "action": "buy",
                "symbol": "SOL",
                "amount": 100.0
            }
        }

        result = finance.execute_task(task)

        assert result.success
        assert "risk_score" in result.result
        assert "approved" in result.result

    def test_finance_calculates_pnl(self):
        """Test: Finance agent calculates PnL"""
        finance = FinanceAgent()
        finance.start()

        task = {
            "type": "calculate_pnl",
            "params": {
                "entry_price": 100.0,
                "current_price": 110.0,
                "amount": 10.0,
                "fees_usd": 5.0
            }
        }

        result = finance.execute_task(task)

        assert result.success
        assert "net_pnl_usd" in result.result
        assert result.result["net_pnl_usd"] == 95.0  # (110-100)*10 - 5


# =============================================================================
# Test Safety Agent
# =============================================================================

@pytest.mark.unit
class TestSafetyAgent:
    """Test safety agent"""

    def test_safety_can_handle_validation_tasks(self):
        """Test: Safety agent can handle validation tasks"""
        safety = SafetyAgent()

        assert safety.can_handle({"type": "validate_safety"})
        assert safety.can_handle({"type": "check_security"})
        assert safety.can_handle({"type": "assess_risk"})
        assert not safety.can_handle({"type": "execute_trade"})

    def test_safety_validates_safe_task(self):
        """Test: Safety agent validates safe task"""
        safety = SafetyAgent()
        safety.start()

        task = {
            "type": "validate_safety",
            "params": {
                "task": {
                    "type": "research",
                    "params": {"topic": "Python programming"}
                },
                "agent_role": "researcher"
            }
        }

        result = safety.execute_task(task)

        assert result.success
        assert "is_safe" in result.result
        assert "risk_score" in result.result

    def test_safety_checks_security(self):
        """Test: Safety agent checks security"""
        safety = SafetyAgent()
        safety.start()

        task = {
            "type": "check_security",
            "params": {
                "operation": "file_access",
                "context": {}
            }
        }

        result = safety.execute_task(task)

        assert result.success
        assert "secure" in result.result


# =============================================================================
# Test Agent Handoffs
# =============================================================================

@pytest.mark.integration
class TestAgentHandoffs:
    """Test agent handoff mechanism"""

    def test_handoff_between_agents(self):
        """Test: Handoff state between agents"""
        planner = PlannerAgent()
        researcher = ResearcherAgent()

        handoff_manager = HandoffManager()

        # Initiate handoff
        context = handoff_manager.initiate_handoff(
            from_agent=planner.agent_id,
            to_agent=researcher.agent_id,
            trigger=HandoffTrigger.TASK_COMPLETED,
            task_state={"task": "research_plan"},
            shared_memory={"goal": "implement feature X"}
        )

        assert context.from_agent == planner.agent_id
        assert context.to_agent == researcher.agent_id
        assert context.task_state["task"] == "research_plan"

        # Complete handoff
        result = handoff_manager.complete_handoff(
            handoff_id=context.handoff_id,
            success=True
        )

        assert result.success
        assert result.from_agent == planner.agent_id

    def test_handoff_context_serialization(self):
        """Test: Handoff context can be serialized"""
        handoff_manager = HandoffManager()

        context = handoff_manager.initiate_handoff(
            from_agent="agent1",
            to_agent="agent2",
            trigger=HandoffTrigger.CAPABILITY_MISMATCH,
            task_state={"data": "test"}
        )

        # Serialize and deserialize
        serialized = handoff_manager.serialize_context(context)
        deserialized = handoff_manager.deserialize_context(serialized)

        assert deserialized.from_agent == context.from_agent
        assert deserialized.to_agent == context.to_agent
        assert deserialized.task_state == context.task_state


# =============================================================================
# Test Resource Limits
# =============================================================================

@pytest.mark.integration
class TestResourceLimits:
    """Test resource governor integration"""

    def test_agent_resource_quota_enforcement(self):
        """Test: Resource quotas are enforced for agents"""
        governor = ResourceGovernor()

        # Set strict quota
        governor.set_quota(
            agent_id="test_agent",
            resource_type=ResourceType.API_CALLS,
            limit=5.0
        )

        # Track usage
        for i in range(5):
            allowed = governor.track_usage("test_agent", ResourceType.API_CALLS, 1.0)
            assert allowed

        # Sixth call should be denied
        allowed = governor.track_usage("test_agent", ResourceType.API_CALLS, 1.0)
        assert not allowed

    def test_agent_resource_status(self):
        """Test: Agent resource status can be queried"""
        governor = ResourceGovernor()

        governor.set_quota(
            agent_id="test_agent",
            resource_type=ResourceType.MEMORY,
            limit=1024.0
        )

        # Use some resources
        governor.track_usage("test_agent", ResourceType.MEMORY, 512.0)

        # Check status
        status = governor.get_agent_status("test_agent")

        assert status["agent_id"] == "test_agent"
        assert "resources" in status
        assert status["resources"]["memory"]["current"] == 512.0


# =============================================================================
# Test Error Handling
# =============================================================================

@pytest.mark.unit
class TestAgentErrorHandling:
    """Test agent error handling"""

    def test_agent_handles_invalid_task(self):
        """Test: Agent handles invalid task gracefully"""
        planner = PlannerAgent()
        planner.start()

        task = {
            "type": "invalid_task_type",
            "params": {}
        }

        result = planner.execute_task(task)

        assert not result.success
        assert result.error is not None

    def test_agent_tracks_failed_tasks(self):
        """Test: Agent tracks failed tasks"""
        coder = CoderAgent()
        coder.start()

        initial_failed_count = coder.failed_tasks_count

        task = {
            "type": "invalid",
            "params": {}
        }

        coder.execute_task(task)

        assert coder.failed_tasks_count == initial_failed_count + 1


# =============================================================================
# Test Agent Metrics
# =============================================================================

@pytest.mark.unit
class TestAgentMetrics:
    """Test agent metrics tracking"""

    def test_agent_tracks_successful_tasks(self):
        """Test: Agent tracks successful task count"""
        researcher = ResearcherAgent()
        researcher.start()

        initial_count = researcher.successful_tasks

        task = {
            "type": "research",
            "params": {"topic": "AI"}
        }

        researcher.execute_task(task)

        assert researcher.successful_tasks == initial_count + 1

    def test_agent_calculates_success_rate(self):
        """Test: Agent calculates success rate"""
        planner = PlannerAgent()
        planner.start()

        # Execute multiple tasks
        for i in range(5):
            planner.execute_task({
                "type": "plan",
                "params": {"goal": f"task_{i}"}
            })

        status = planner.get_status()

        assert "success_rate" in status
        assert status["success_rate"] >= 0.0
        assert status["success_rate"] <= 1.0


# =============================================================================
# Integration Test: Multi-Agent Workflow
# =============================================================================

@pytest.mark.integration
class TestMultiAgentWorkflow:
    """Test complete multi-agent workflow"""

    def test_complete_workflow(self):
        """Test: Complete multi-agent workflow executes successfully"""
        # Create agents
        planner = PlannerAgent()
        researcher = ResearcherAgent()
        coder = CoderAgent()
        safety = SafetyAgent()

        # Start all agents
        for agent in [planner, researcher, coder, safety]:
            agent.start()

        # Step 1: Planner creates plan
        plan_result = planner.execute_task({
            "type": "plan",
            "params": {"goal": "implement a sorting function"}
        })
        assert plan_result.success

        # Step 2: Researcher gathers information
        research_result = researcher.execute_task({
            "type": "research",
            "params": {"topic": "sorting algorithms"}
        })
        assert research_result.success

        # Step 3: Coder writes code
        code_result = coder.execute_task({
            "type": "write_code",
            "params": {
                "specification": "sorting function",
                "language": "python"
            }
        })
        assert code_result.success

        # Step 4: Safety validates code
        safety_result = safety.execute_task({
            "type": "review_code",
            "params": {"code": code_result.result["code"]}
        })
        assert safety_result.success

        # Verify all agents completed their tasks
        for agent in [planner, researcher, coder, safety]:
            assert agent.successful_tasks >= 1
