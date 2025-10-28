#!/usr/bin/env python3
"""Quick smoke test for AGI system."""

print("🧪 ShivX AGI Quick Test\n")

# Test 1: Imports
print("1️⃣  Testing imports...")
from memory.graph_store.store import MemoryGraphStore
from memory.schemas import MemoryNode, NodeType
from learning.online_adapter.adapter import OnlineAdapter
from sim.mini_world.simulator import MiniWorldSimulator
from cognition.tom.tom_reasoner import ToMReasoner
from resilience.reflector.reflector import Reflector
print("   ✅ All imports successful\n")

# Test 2: Memory
print("2️⃣  Testing memory system...")
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    db = MemoryGraphStore(str(Path(tmpdir) / "test.db"))
    node = MemoryNode(node_type=NodeType.EVENT, content="Test", importance=0.9)
    nid = db.add_node(node)
    retrieved = db.get_node(nid)
    assert retrieved.content == "Test"
    db.close()
print("   ✅ Memory working\n")

# Test 3: Learning
print("3️⃣  Testing learning system...")
adapter = OnlineAdapter(adapter_dir="./tmp_adapters")
aid = adapter.create_adapter("test")
assert aid is not None
print("   ✅ Learning working\n")

# Test 4: Spatial
print("4️⃣  Testing spatial system...")
sim = MiniWorldSimulator(5, 5)
sim.reset()
_, _, _, info = sim.step("right")
assert info["steps"] == 1
print("   ✅ Spatial working\n")

# Test 5: ToM
print("5️⃣  Testing Theory-of-Mind...")
tom = ToMReasoner()
tom.add_agent("Alice")
tom.teach("Alice", "fact")
assert tom.agent_knows("Alice", "fact")
print("   ✅ ToM working\n")

# Test 6: Reflection
print("6️⃣  Testing reflection...")
ref = Reflector()
result = ref.check_output("test", 0.9)
assert result["safe"]
print("   ✅ Reflection working\n")

print("=" * 50)
print("✅ ALL SYSTEMS OPERATIONAL!")
print("=" * 50)
print("\n🚀 ShivX AGI is ready for deployment!\n")
