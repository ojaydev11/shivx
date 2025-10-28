# ShivX Multi-Agent Orchestration - Quick Start Guide

**Version:** 1.0.0
**Date:** 2025-10-28

## 5-Minute Quick Start

### 1. Basic Agent Usage

```python
from core.agents import PlannerAgent, CoderAgent, SafetyAgent

# Initialize agents
planner = PlannerAgent()
planner.start()

# Execute a task
result = planner.execute_task({
    "type": "plan",
    "params": {"goal": "implement feature X"}
})

print(f"Success: {result.success}")
print(f"Tasks: {result.result['tasks']}")
```

### 2. Intent Routing

```python
from core.orchestration import get_intent_router

router = get_intent_router()

# Classify user intent
result = router.classify("write a function to sort an array")

print(f"Intent: {result.intent.value}")        # "code"
print(f"Agent: {result.agent_role}")           # "coder"
print(f"Confidence: {result.confidence:.2f}")  # 0.90
```

### 3. Task Graph Execution

```python
from core.orchestration import TaskGraph

graph = TaskGraph()

# Add tasks with dependencies
graph.add_task("init", "Initialize", init_handler)
graph.add_task("process", "Process", process_handler, dependencies=["init"])
graph.add_task("finalize", "Finalize", final_handler, dependencies=["process"])

# Execute
result = graph.execute(enable_parallel=True)
print(f"Status: {result.status}")
```

### 4. Agent Handoff

```python
from core.orchestration import get_handoff_manager, HandoffTrigger

manager = get_handoff_manager()

# Transfer task between agents
context = manager.initiate_handoff(
    from_agent="planner",
    to_agent="researcher",
    trigger=HandoffTrigger.TASK_COMPLETED,
    task_state={"plan": plan_data}
)

# Complete handoff
manager.complete_handoff(context.handoff_id, success=True)
```

### 5. Resource Management

```python
from core.orchestration import get_resource_governor, ResourceType

governor = get_resource_governor()

# Set quota
governor.set_quota(
    agent_id="coder",
    resource_type=ResourceType.API_CALLS,
    limit=1000.0
)

# Track usage
governor.track_usage("coder", ResourceType.API_CALLS, 1.0)

# Check status
status = governor.get_agent_status("coder")
```

## API Quick Reference

### Route Intent
```bash
curl -X POST http://localhost:8000/api/orchestration/route \
  -H "Content-Type: application/json" \
  -d '{"user_input": "write a sorting function"}'
```

### Execute Task Graph
```bash
curl -X POST http://localhost:8000/api/orchestration/execute \
  -H "Content-Type: application/json" \
  -d '{"goal": "implement feature X", "parallel": true}'
```

### List Agents
```bash
curl http://localhost:8000/api/orchestration/agents
```

### Get Agent Status
```bash
curl http://localhost:8000/api/orchestration/agents/planner/status
```

## Example Workflows

### Run All Workflows
```bash
python examples/orchestration_workflows.py
```

### Individual Workflow
```python
from examples.orchestration_workflows import OrchestrationWorkflows

workflows = OrchestrationWorkflows()

# Run specific workflow
result = workflows.workflow_1_code_implementation()
# or
result = workflows.workflow_3_trading_workflow("SOL", "buy", 10.0)
```

## Testing

### Import Verification
```bash
python -c "from core.agents import *; print('✓ OK')"
python -c "from core.orchestration import *; print('✓ OK')"
```

### Agent Test
```bash
python -c "
from core.agents import PlannerAgent
planner = PlannerAgent()
planner.start()
result = planner.execute_task({'type': 'plan', 'params': {'goal': 'test'}})
print(f'✓ Success: {result.success}')
"
```

### Task Graph Test
```bash
python -c "
from core.orchestration import TaskGraph
graph = TaskGraph()
graph.add_task('t1', 'Test', lambda p: 'result')
result = graph.execute()
print(f'✓ Completed: {result.completed_tasks}/{result.total_tasks}')
"
```

## Common Patterns

### Pattern 1: Plan → Execute → Validate
```python
# Step 1: Plan
plan_result = planner.execute_task({
    "type": "plan",
    "params": {"goal": goal}
})

# Step 2: Execute
code_result = coder.execute_task({
    "type": "write_code",
    "params": plan_result.result
})

# Step 3: Validate
safety_result = safety.execute_task({
    "type": "validate_safety",
    "params": {"task": code_result.result}
})
```

### Pattern 2: Research → Analyze → Report
```python
# Gather
gather_result = researcher.execute_task({
    "type": "gather_information",
    "params": {"query": topic}
})

# Analyze
analysis_result = researcher.execute_task({
    "type": "analyze_data",
    "params": {"data": gather_result.result}
})

# Report
report = researcher.execute_task({
    "type": "generate_report",
    "params": {"findings": analysis_result.result}
})
```

### Pattern 3: Analyze → Risk Check → Execute
```python
# Market analysis
market = finance.execute_task({
    "type": "analyze_market",
    "params": {"symbol": "SOL"}
})

# Risk assessment
risk = finance.execute_task({
    "type": "assess_risk",
    "params": {"action": "buy", "amount": 10.0}
})

# Execute if safe
if risk.result["approved"]:
    trade = finance.execute_task({
        "type": "execute_trade",
        "params": {...}
    })
```

## Agent Capabilities

| Agent | Best For | Example Tasks |
|-------|----------|---------------|
| Planner | Breaking down goals | plan, decompose, generate_tasks |
| Researcher | Information gathering | research, web_search, analyze_data |
| Coder | Code generation | write_code, review_code, refactor |
| Operator | System management | execute_command, monitor_resources |
| Finance | Trading operations | execute_trade, analyze_market |
| Safety | Validation | validate_safety, check_security |

## Troubleshooting

### Agent Not Starting
```python
agent = PlannerAgent()
agent.start()  # Don't forget to start!
print(agent.status)  # Should be "idle"
```

### Task Failing
```python
result = agent.execute_task(task)
if not result.success:
    print(f"Error: {result.error}")
    print(f"Time: {result.execution_time_sec}s")
```

### Quota Exceeded
```python
governor = get_resource_governor()
governor.reset_quota("agent_id")
# or check status
status = governor.get_agent_status("agent_id")
```

## Performance Tips

1. **Use Parallel Execution** when tasks are independent
2. **Set Appropriate Quotas** to prevent resource exhaustion
3. **Monitor Agent Status** to identify bottlenecks
4. **Use Handoffs** for task specialization
5. **Cache Intent Classifications** for repeated queries

## Next Steps

- Read full documentation: `docs/MULTI_AGENT_FRAMEWORK.md`
- Run example workflows: `examples/orchestration_workflows.py`
- Review tests: `tests/test_agents.py`
- Explore API: `app/routers/orchestration.py`

---

**Quick Start Complete!** Ready to build autonomous multi-agent systems.
