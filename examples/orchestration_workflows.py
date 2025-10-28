"""
Multi-Agent Orchestration Workflows
====================================

Example workflows demonstrating the complete multi-agent orchestration system.

Workflows:
1. Code Implementation Workflow
2. Research and Analysis Workflow
3. Trading Workflow
4. System Management Workflow

Each workflow demonstrates:
- Intent routing
- Multi-agent coordination
- Task graph execution
- Agent handoffs
- Resource management
- Safety validation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

# Import orchestration components
from core.orchestration import (
    get_intent_router,
    TaskGraph,
    TaskType,
    get_handoff_manager,
    HandoffTrigger,
    get_resource_governor,
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
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationWorkflows:
    """Example workflows for multi-agent orchestration"""

    def __init__(self):
        """Initialize workflows with agents and orchestration components"""
        # Initialize agents
        self.planner = PlannerAgent(agent_id="planner")
        self.researcher = ResearcherAgent(agent_id="researcher")
        self.coder = CoderAgent(agent_id="coder")
        self.operator = OperatorAgent(agent_id="operator")
        self.finance = FinanceAgent(agent_id="finance")
        self.safety = SafetyAgent(agent_id="safety")

        # Start all agents
        for agent in [self.planner, self.researcher, self.coder, self.operator, self.finance, self.safety]:
            agent.start()

        # Initialize orchestration components
        self.intent_router = get_intent_router()
        self.handoff_manager = get_handoff_manager()
        self.resource_governor = get_resource_governor()

        logger.info("OrchestrationWorkflows initialized with 6 agents")

    def workflow_1_code_implementation(self, goal: str = "Implement a binary search function in Python"):
        """
        Workflow 1: Code Implementation

        Steps:
        1. Route intent -> Coder
        2. Planner breaks down into subtasks
        3. Researcher gathers information on best practices
        4. Coder implements the code
        5. Safety validates the code
        6. Return results
        """
        logger.info("\n" + "="*80)
        logger.info("WORKFLOW 1: Code Implementation")
        logger.info("="*80)
        logger.info(f"Goal: {goal}\n")

        # Step 1: Route intent
        logger.info("Step 1: Routing intent...")
        intent_result = self.intent_router.classify(goal)
        logger.info(f"  Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")
        logger.info(f"  Routed to: {intent_result.agent_role}")

        # Step 2: Planner creates plan
        logger.info("\nStep 2: Creating plan...")
        plan_result = self.planner.execute_task({
            "type": "plan",
            "params": {"goal": goal}
        })
        logger.info(f"  Plan created: {len(plan_result.result['tasks'])} tasks")

        # Step 3: Handoff to Researcher
        logger.info("\nStep 3: Handoff to Researcher...")
        research_context = self.handoff_manager.initiate_handoff(
            from_agent=self.planner.agent_id,
            to_agent=self.researcher.agent_id,
            trigger=HandoffTrigger.TASK_COMPLETED,
            task_state={"plan": plan_result.result},
            shared_memory={"goal": goal}
        )

        # Researcher gathers information
        research_result = self.researcher.execute_task({
            "type": "research",
            "params": {"topic": "binary search algorithms"}
        })
        logger.info(f"  Research completed: {research_result.result['confidence']} confidence")

        self.handoff_manager.complete_handoff(research_context.handoff_id, success=True)

        # Step 4: Handoff to Coder
        logger.info("\nStep 4: Handoff to Coder...")
        code_context = self.handoff_manager.initiate_handoff(
            from_agent=self.researcher.agent_id,
            to_agent=self.coder.agent_id,
            trigger=HandoffTrigger.TASK_COMPLETED,
            task_state={"research": research_result.result},
            shared_memory={"goal": goal}
        )

        # Coder implements
        code_result = self.coder.execute_task({
            "type": "write_code",
            "params": {
                "specification": "Binary search function",
                "language": "python"
            }
        })
        logger.info(f"  Code written: {code_result.result['lines_of_code']} lines")

        self.handoff_manager.complete_handoff(code_context.handoff_id, success=True)

        # Step 5: Handoff to Safety for validation
        logger.info("\nStep 5: Safety validation...")
        safety_context = self.handoff_manager.initiate_handoff(
            from_agent=self.coder.agent_id,
            to_agent=self.safety.agent_id,
            trigger=HandoffTrigger.TASK_COMPLETED,
            task_state={"code": code_result.result},
            shared_memory={"goal": goal}
        )

        # Safety validates
        safety_result = self.safety.execute_task({
            "type": "review_code",
            "params": {"code": code_result.result["code"]}
        })
        logger.info(f"  Safety check: {'PASSED' if safety_result.success else 'FAILED'}")
        logger.info(f"  Security score: {safety_result.result.get('security_score', 'N/A')}")

        self.handoff_manager.complete_handoff(safety_context.handoff_id, success=True)

        # Summary
        logger.info("\nWorkflow 1 Complete!")
        logger.info(f"  Total handoffs: 3")
        logger.info(f"  All steps completed successfully")

        return {
            "workflow": "code_implementation",
            "goal": goal,
            "steps_completed": 5,
            "code": code_result.result,
            "safety_check": safety_result.result
        }

    def workflow_2_research_and_analysis(self, topic: str = "Quantum Computing"):
        """
        Workflow 2: Research and Analysis

        Steps:
        1. Route intent -> Researcher
        2. Researcher gathers information
        3. Researcher analyzes data
        4. Researcher generates report
        5. Return results
        """
        logger.info("\n" + "="*80)
        logger.info("WORKFLOW 2: Research and Analysis")
        logger.info("="*80)
        logger.info(f"Topic: {topic}\n")

        # Step 1: Route intent
        logger.info("Step 1: Routing intent...")
        intent_result = self.intent_router.classify(f"research {topic}")
        logger.info(f"  Intent: {intent_result.intent.value}")
        logger.info(f"  Routed to: {intent_result.agent_role}")

        # Step 2: Gather information
        logger.info("\nStep 2: Gathering information...")
        gather_result = self.researcher.execute_task({
            "type": "gather_information",
            "params": {"query": topic, "sources": ["academic", "web", "news"]}
        })
        logger.info(f"  Sources checked: {gather_result.result['sources_checked']}")

        # Step 3: Analyze data
        logger.info("\nStep 3: Analyzing data...")
        analysis_result = self.researcher.execute_task({
            "type": "analyze_data",
            "params": {"data": gather_result.result, "analysis_type": "comprehensive"}
        })
        logger.info(f"  Insights found: {len(analysis_result.result['insights'])}")

        # Step 4: Generate report
        logger.info("\nStep 4: Generating report...")
        report_result = self.researcher.execute_task({
            "type": "generate_report",
            "params": {"findings": analysis_result.result, "format": "markdown"}
        })
        logger.info(f"  Report sections: {len(report_result.result['sections'])}")

        # Summary
        logger.info("\nWorkflow 2 Complete!")
        logger.info(f"  Research depth: comprehensive")
        logger.info(f"  Report generated successfully")

        return {
            "workflow": "research_and_analysis",
            "topic": topic,
            "data": gather_result.result,
            "analysis": analysis_result.result,
            "report": report_result.result
        }

    def workflow_3_trading_workflow(self, symbol: str = "SOL", action: str = "buy", amount: float = 10.0):
        """
        Workflow 3: Trading Workflow

        Steps:
        1. Route intent -> Finance
        2. Finance analyzes market
        3. Finance assesses risk
        4. Safety validates trade
        5. Finance executes trade
        6. Finance monitors position
        """
        logger.info("\n" + "="*80)
        logger.info("WORKFLOW 3: Trading Workflow")
        logger.info("="*80)
        logger.info(f"Trade: {action} {amount} {symbol}\n")

        # Step 1: Route intent
        logger.info("Step 1: Routing intent...")
        intent_result = self.intent_router.classify(f"{action} {amount} {symbol}")
        logger.info(f"  Intent: {intent_result.intent.value}")
        logger.info(f"  Routed to: {intent_result.agent_role}")

        # Step 2: Market analysis
        logger.info("\nStep 2: Analyzing market...")
        market_result = self.finance.execute_task({
            "type": "analyze_market",
            "params": {"symbol": symbol, "chain": "solana"}
        })
        logger.info(f"  Current price: ${market_result.result['current_price']}")
        logger.info(f"  Market trend: {market_result.result['trend']}")

        # Step 3: Risk assessment
        logger.info("\nStep 3: Assessing risk...")
        risk_result = self.finance.execute_task({
            "type": "assess_risk",
            "params": {"action": action, "symbol": symbol, "amount": amount}
        })
        logger.info(f"  Risk score: {risk_result.result['risk_score']}/100")
        logger.info(f"  Risk level: {risk_result.result['risk_level']}")

        # Step 4: Safety validation
        logger.info("\nStep 4: Safety validation...")
        safety_result = self.safety.execute_task({
            "type": "validate_safety",
            "params": {
                "task": {
                    "type": "execute_trade",
                    "params": {"action": action, "symbol": symbol, "amount": amount}
                },
                "agent_role": "finance"
            }
        })
        logger.info(f"  Safety check: {'PASSED' if safety_result.result['is_safe'] else 'FAILED'}")
        logger.info(f"  Risk score: {safety_result.result['risk_score']}/100")

        # Step 5: Execute trade (only if safe)
        if safety_result.result['is_safe']:
            logger.info("\nStep 5: Executing trade...")
            trade_result = self.finance.execute_task({
                "type": "execute_trade",
                "params": {
                    "action": action,
                    "symbol": symbol,
                    "amount": amount,
                    "chain": "solana"
                }
            })
            logger.info(f"  Trade ID: {trade_result.result['trade_id']}")
            logger.info(f"  Status: {trade_result.result['status']}")

            # Step 6: Monitor position
            logger.info("\nStep 6: Monitoring position...")
            monitor_result = self.finance.execute_task({
                "type": "monitor_position",
                "params": {
                    "position_id": trade_result.result['trade_id'],
                    "symbol": symbol
                }
            })
            logger.info(f"  Position status: {monitor_result.result['status']}")
            logger.info(f"  Current PnL: ${monitor_result.result['pnl_usd']}")
        else:
            logger.warning("\nTrade blocked by safety validation!")
            trade_result = None
            monitor_result = None

        # Summary
        logger.info("\nWorkflow 3 Complete!")
        logger.info(f"  Trade executed: {trade_result is not None}")

        return {
            "workflow": "trading",
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "market_analysis": market_result.result,
            "risk_assessment": risk_result.result,
            "safety_validation": safety_result.result,
            "trade": trade_result.result if trade_result else None,
            "position": monitor_result.result if monitor_result else None
        }

    def workflow_4_task_graph_execution(self):
        """
        Workflow 4: Task Graph Execution

        Demonstrates parallel task execution with dependencies.
        """
        logger.info("\n" + "="*80)
        logger.info("WORKFLOW 4: Task Graph Execution")
        logger.info("="*80)
        logger.info("Executing complex DAG with parallel tasks\n")

        # Create task graph
        graph = TaskGraph(max_workers=4)

        # Define task handlers
        def init_task(params):
            logger.info("  [Task] Initialize")
            return {"initialized": True}

        def research_task(params):
            logger.info("  [Task] Research (parallel)")
            result = self.researcher.execute_task({
                "type": "research",
                "params": {"topic": "AGI"}
            })
            return result.result

        def analyze_task(params):
            logger.info("  [Task] Analyze (parallel)")
            result = self.researcher.execute_task({
                "type": "analyze_data",
                "params": {"data": {}, "analysis_type": "summary"}
            })
            return result.result

        def code_task(params):
            logger.info("  [Task] Code implementation")
            result = self.coder.execute_task({
                "type": "write_code",
                "params": {"specification": "AGI core", "language": "python"}
            })
            return result.result

        def safety_task(params):
            logger.info("  [Task] Safety validation")
            result = self.safety.execute_task({
                "type": "check_security",
                "params": {"operation": "deployment"}
            })
            return result.result

        def deploy_task(params):
            logger.info("  [Task] Deploy")
            return {"deployed": True}

        # Build graph
        logger.info("Building task graph...")
        graph.add_task("init", "Initialize", init_task)
        graph.add_task("research", "Research", research_task, dependencies=["init"])
        graph.add_task("analyze", "Analyze", analyze_task, dependencies=["init"])
        graph.add_task("code", "Code", code_task, dependencies=["research", "analyze"])
        graph.add_task("safety", "Safety", safety_task, dependencies=["code"])
        graph.add_task("deploy", "Deploy", deploy_task, dependencies=["safety"])

        logger.info(f"  Tasks: {len(graph.tasks)}")
        logger.info(f"  Execution order: {' -> '.join(graph.topological_sort())}")

        # Execute graph
        logger.info("\nExecuting task graph...")
        result = graph.execute(enable_parallel=True, stop_on_error=True)

        # Summary
        logger.info("\nWorkflow 4 Complete!")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Completed: {result.completed_tasks}/{result.total_tasks}")
        logger.info(f"  Execution time: {result.execution_time_sec:.2f}s")

        # Export visualization
        from pathlib import Path
        viz_path = Path("var/orchestration/task_graph.dot")
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        graph.export_dot(viz_path)
        logger.info(f"  Visualization exported: {viz_path}")

        return {
            "workflow": "task_graph",
            "status": result.status,
            "total_tasks": result.total_tasks,
            "completed_tasks": result.completed_tasks,
            "execution_time_sec": result.execution_time_sec,
            "results": result.results
        }

    def demo_all_workflows(self):
        """Run all example workflows"""
        logger.info("\n" + "="*80)
        logger.info("MULTI-AGENT ORCHESTRATION DEMO")
        logger.info("Running all example workflows")
        logger.info("="*80)

        results = {}

        # Workflow 1: Code Implementation
        results["workflow_1"] = self.workflow_1_code_implementation()

        # Workflow 2: Research and Analysis
        results["workflow_2"] = self.workflow_2_research_and_analysis()

        # Workflow 3: Trading
        results["workflow_3"] = self.workflow_3_trading_workflow()

        # Workflow 4: Task Graph
        results["workflow_4"] = self.workflow_4_task_graph_execution()

        # Final statistics
        logger.info("\n" + "="*80)
        logger.info("DEMO COMPLETE - STATISTICS")
        logger.info("="*80)

        # Intent router stats
        router_stats = self.intent_router.get_stats()
        logger.info(f"\nIntent Router:")
        logger.info(f"  Total classifications: {router_stats['total_classifications']}")
        logger.info(f"  Rule-based: {router_stats['rule_based']} ({router_stats['rule_based_rate']:.1%})")
        logger.info(f"  Fallback: {router_stats['fallback']} ({router_stats['fallback_rate']:.1%})")

        # Handoff stats
        handoff_stats = self.handoff_manager.get_handoff_stats()
        logger.info(f"\nHandoff Manager:")
        logger.info(f"  Total handoffs: {handoff_stats['total_handoffs']}")
        logger.info(f"  Successful: {handoff_stats['successful']}")
        logger.info(f"  Success rate: {handoff_stats['success_rate']:.1%}")

        # Resource stats
        resource_stats = self.resource_governor.get_stats()
        logger.info(f"\nResource Governor:")
        logger.info(f"  Total agents: {resource_stats['total_agents']}")
        logger.info(f"  Total checks: {resource_stats['total_checks']}")
        logger.info(f"  Violations: {resource_stats['total_violations']}")

        # Agent stats
        logger.info(f"\nAgent Statistics:")
        for agent in [self.planner, self.researcher, self.coder, self.operator, self.finance, self.safety]:
            status = agent.get_status()
            logger.info(f"  {agent.role}: {status['total_tasks']} tasks, {status['success_rate']:.1%} success")

        logger.info("\n" + "="*80 + "\n")

        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for example workflows"""
    workflows = OrchestrationWorkflows()

    # Run all workflows
    results = workflows.demo_all_workflows()

    # Save results
    import json
    from pathlib import Path

    output_file = Path("var/orchestration/workflow_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
