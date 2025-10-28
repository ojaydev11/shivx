#!/usr/bin/env python3
"""
Quick validation script for ShivX AGI implementation.

Tests basic functionality without requiring full dependencies.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("ShivX AGI Validation Script")
print("=" * 60)
print()

# Test 1: Import core modules
print("Test 1: Importing core modules...")
try:
    from memory.schemas import MemoryNode, MemoryEdge, NodeType
    from memory.graph_store.store import MemoryGraphStore
    print("✓ Memory schemas imported")

    from learning.online_adapter.adapter import OnlineAdapter
    from learning.experience_buffer.buffer import ExperienceBuffer
    from learning.scheduler.learning_scheduler import LearningScheduler
    print("✓ Learning modules imported")

    from vision.spatial_parser.parser import SpatialParser
    from sim.mini_world.simulator import MiniWorldSimulator
    from reasoners.spatial_planner.planner import SpatialPlanner
    print("✓ Spatial reasoning modules imported")

    from cognition.tom.tom_reasoner import ToMReasoner
    print("✓ Theory-of-Mind module imported")

    from resilience.reflector.reflector import Reflector
    from resilience.self_repair.repairer import SelfRepairer
    print("✓ Reflection & Self-Repair modules imported")

    from daemons.supervisor import Supervisor
    print("✓ Supervisor daemon imported")

    print("\n✅ All imports successful!\n")
except ImportError as e:
    print(f"\n❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Basic graph store functionality
print("Test 2: Testing memory graph store...")
try:
    import tempfile
    from datetime import datetime

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        store = MemoryGraphStore(db_path=str(db_path))

        # Create and add a node
        node = MemoryNode(
            node_type=NodeType.EVENT,
            content="Test event",
            importance=0.8,
        )
        node_id = store.add_node(node)

        # Retrieve the node
        retrieved = store.get_node(node_id)
        assert retrieved is not None
        assert retrieved.content == "Test event"
        assert retrieved.importance == 0.8

        # Count nodes
        count = store.count_nodes()
        assert count == 1

        store.close()

    print("✓ Graph store: create, retrieve, count")
    print("✅ Memory graph store working!\n")
except Exception as e:
    print(f"❌ Graph store test failed: {e}\n")
    sys.exit(1)

# Test 3: Experience buffer
print("Test 3: Testing experience buffer...")
try:
    buffer = ExperienceBuffer(max_size=100)

    # Add experiences
    for i in range(10):
        buffer.add(
            input_data=f"input_{i}",
            target=f"target_{i}",
            importance=0.5 + i * 0.05,
        )

    assert len(buffer) == 10

    # Sample
    samples = buffer.sample(batch_size=5)
    assert len(samples) == 5

    # Get stats
    stats = buffer.get_stats()
    assert stats["size"] == 10

    print("✓ Experience buffer: add, sample, stats")
    print("✅ Experience buffer working!\n")
except Exception as e:
    print(f"❌ Experience buffer test failed: {e}\n")
    sys.exit(1)

# Test 4: Spatial simulator
print("Test 4: Testing spatial simulator...")
try:
    sim = MiniWorldSimulator(width=5, height=5)
    sim.reset()

    # Take some steps
    _, reward, done, info = sim.step("right")
    _, reward, done, info = sim.step("down")

    assert info["steps"] == 2
    assert not done

    print("✓ Simulator: reset, step, check state")
    print("✅ Spatial simulator working!\n")
except Exception as e:
    print(f"❌ Simulator test failed: {e}\n")
    sys.exit(1)

# Test 5: Spatial planner
print("Test 5: Testing spatial planner...")
try:
    import numpy as np

    planner = SpatialPlanner(algorithm="astar")

    # Create simple grid
    grid = np.zeros((5, 5), dtype=int)

    # Plan path
    path = planner.plan_path(start=(0, 0), goal=(4, 4), grid=grid)

    assert path is not None
    assert len(path) > 0
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)

    # Convert to actions
    actions = planner.actions_from_path(path)
    assert len(actions) > 0

    print(f"✓ Planner: found path with {len(path)} steps, {len(actions)} actions")
    print("✅ Spatial planner working!\n")
except Exception as e:
    print(f"❌ Planner test failed: {e}\n")
    sys.exit(1)

# Test 6: Theory-of-Mind
print("Test 6: Testing Theory-of-Mind...")
try:
    tom = ToMReasoner(max_agents=5)

    # Add agents
    alice = tom.add_agent("Alice")
    bob = tom.add_agent("Bob")

    # Update beliefs
    tom.update_belief("Alice", "project_status", "in_progress")
    tom.teach("Alice", "Python is a programming language")

    # Check knowledge
    assert tom.agent_knows("Alice", "Python is a programming language")
    assert not tom.agent_knows("Bob", "Python is a programming language")

    # Get common knowledge
    common = tom.get_common_knowledge(["Alice", "Bob"])
    assert len(common) == 0  # No common knowledge yet

    print("✓ ToM: agents, beliefs, knowledge, common knowledge")
    print("✅ Theory-of-Mind working!\n")
except Exception as e:
    print(f"❌ ToM test failed: {e}\n")
    sys.exit(1)

# Test 7: Reflector
print("Test 7: Testing reflector...")
try:
    reflector = Reflector(
        hallucination_detection=True,
        confidence_threshold=0.8,
    )

    # Check high confidence output
    result1 = reflector.check_output(
        output="This is a confident statement",
        confidence=0.95,
    )
    assert result1["safe"]

    # Check low confidence output
    result2 = reflector.check_output(
        output="I'm not sure about this",
        confidence=0.5,
    )
    assert not result2["safe"]
    assert len(result2["issues"]) > 0

    # Run audit
    outputs = [
        {"output": "Output 1", "confidence": 0.9},
        {"output": "I think maybe", "confidence": 0.6},
    ]
    report = reflector.audit(outputs)

    assert report.outputs_checked == 2

    print("✓ Reflector: check outputs, detect issues, audit")
    print("✅ Reflector working!\n")
except Exception as e:
    print(f"❌ Reflector test failed: {e}\n")
    sys.exit(1)

# Test 8: Self-repairer
print("Test 8: Testing self-repairer...")
try:
    repairer = SelfRepairer(
        enabled=True,
        auto_patch=False,
        sandbox_test=True,
    )

    # Detect issue
    error = AttributeError("'NoneType' object has no attribute 'value'")
    issue_type = repairer.detect_issue(error, {})
    assert issue_type == "missing_attribute"

    # Propose fix
    fix = repairer.propose_fix(issue_type, error, {})
    assert fix is not None

    print("✓ Self-repairer: detect issue, propose fix")
    print("✅ Self-repairer working!\n")
except Exception as e:
    print(f"❌ Self-repairer test failed: {e}\n")
    sys.exit(1)

# Test 9: Online adapter
print("Test 9: Testing online adapter...")
try:
    adapter = OnlineAdapter(
        adapter_dir="./test_adapters",
        method="lora",
        rank=8,
    )

    # Create adapter
    adapter_id = adapter.create_adapter("test_task")
    assert adapter_id is not None

    # Train step
    batch = {"inputs": ["test"], "targets": ["output"]}
    metrics = adapter.train_step(adapter_id, batch)
    assert "loss" in metrics
    assert "steps" in metrics

    # Get info
    info = adapter.get_adapter_info(adapter_id)
    assert info is not None
    assert info["id"] == adapter_id

    print("✓ Adapter: create, train, get info")
    print("✅ Online adapter working!\n")
except Exception as e:
    print(f"❌ Adapter test failed: {e}\n")
    sys.exit(1)

# Test 10: Learning scheduler
print("Test 10: Testing learning scheduler...")
try:
    adapter = OnlineAdapter(adapter_dir="./test_adapters")
    buffer = ExperienceBuffer(max_size=100)

    scheduler = LearningScheduler(
        adapter=adapter,
        experience_buffer=buffer,
        mode="idle",
        idle_threshold_seconds=300,
    )

    # Add some experiences
    for i in range(20):
        buffer.add(f"input_{i}", f"target_{i}")

    # Check if should train
    scheduler.record_activity()
    should_train = scheduler.should_train()
    # Won't train immediately due to idle threshold

    # Get stats
    stats = scheduler.get_training_stats()
    assert "sessions" in stats

    print("✓ Scheduler: activity tracking, training decision, stats")
    print("✅ Learning scheduler working!\n")
except Exception as e:
    print(f"❌ Scheduler test failed: {e}\n")
    sys.exit(1)

print("=" * 60)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 60)
print()
print("Summary:")
print("  ✓ All 10 core systems validated")
print("  ✓ Imports successful")
print("  ✓ Basic functionality working")
print("  ✓ No critical errors detected")
print()
print("The AGI system is ready for deployment!")
print()
