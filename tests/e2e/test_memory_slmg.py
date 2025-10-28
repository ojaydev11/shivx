"""
End-to-end tests for Semantic Long-Term Memory Graph (SLMG).

Tests:
- Storage and retrieval of memories
- Episodic, semantic, and procedural memory
- Hybrid retrieval
- Consolidation
- Persistence across sessions
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memory.api import MemoryAPI
from memory.schemas import MemoryMode


class TestSLMGBasics:
    """Basic SLMG functionality tests."""

    @pytest.fixture
    def memory_api(self):
        """Create temporary memory API for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            api = MemoryAPI(db_path=str(db_path), device="cpu")
            yield api
            api.close()

    def test_store_and_recall_event(self, memory_api):
        """Test storing and recalling episodic events."""
        # Store an event
        event_id = memory_api.store_event(
            description="Had a great meeting with the team",
            participants=["Alice", "Bob", "Charlie"],
            location="Conference Room A",
            outcome="Agreed on project timeline",
            importance=0.8,
            tags=["meeting", "team"],
        )

        assert event_id is not None

        # Recall the event
        results = memory_api.recall("meeting with team", k=5)

        assert len(results.nodes) > 0
        assert results.nodes[0].id == event_id
        assert "meeting" in results.nodes[0].content.lower()

    def test_store_and_recall_fact(self, memory_api):
        """Test storing and recalling semantic facts."""
        # Store a fact
        fact_id = memory_api.store_fact(
            subject="Python",
            predicate="is a",
            object="programming language",
            confidence=1.0,
            source="common knowledge",
            importance=0.7,
        )

        assert fact_id is not None

        # Recall facts about Python
        facts = memory_api.recall_facts_about("Python", limit=5)

        assert len(facts) > 0
        assert facts[0].content == "Python is a programming language"

    def test_store_and_recall_skill(self, memory_api):
        """Test storing and recalling procedural skills."""
        # Store a skill
        skill_id = memory_api.store_skill(
            name="make_coffee",
            description="How to make coffee",
            steps=[
                "Boil water",
                "Grind coffee beans",
                "Add coffee to filter",
                "Pour hot water over coffee",
                "Wait 4 minutes",
                "Enjoy!",
            ],
            importance=0.6,
            tags=["cooking", "beverage"],
        )

        assert skill_id is not None

        # Recall the skill
        skill = memory_api.recall_skill("make_coffee")

        assert skill is not None
        assert skill.metadata["name"] == "make_coffee"
        assert len(skill.metadata["steps"]) == 6

    def test_store_code_snippet(self, memory_api):
        """Test storing code snippets."""
        code_id = memory_api.store_code(
            code="def hello():\n    print('Hello, World!')",
            description="Simple hello world function",
            language="python",
            tags=["example", "basic"],
        )

        assert code_id is not None

        # Recall the code
        results = memory_api.recall("hello world function", k=5)

        assert len(results.nodes) > 0
        code_node = results.nodes[0]
        assert "python" in code_node.metadata.get("language", "")


class TestSLMGRetrieval:
    """Test retrieval capabilities."""

    @pytest.fixture
    def memory_api_with_data(self):
        """Create memory API pre-loaded with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            api = MemoryAPI(db_path=str(db_path), device="cpu")

            # Add multiple events
            for i in range(10):
                api.store_event(
                    description=f"Event {i}: Important meeting about project planning",
                    participants=["Alice", "Bob"],
                    importance=0.5 + (i * 0.05),
                )

            # Add multiple facts
            api.store_fact("Alice", "works on", "Project X", importance=0.7)
            api.store_fact("Bob", "leads", "Team Y", importance=0.8)
            api.store_fact("Project X", "uses", "Python", importance=0.6)

            yield api
            api.close()

    def test_recall_with_hybrid_mode(self, memory_api_with_data):
        """Test hybrid retrieval mode."""
        results = memory_api_with_data.recall(
            "meeting about project",
            k=5,
            mode=MemoryMode.HYBRID,
        )

        assert len(results.nodes) > 0
        assert results.metadata["mode"] == "hybrid"
        assert results.metadata["latency_ms"] < 150  # < 150ms requirement

    def test_recall_with_importance_filter(self, memory_api_with_data):
        """Test filtering by importance."""
        results = memory_api_with_data.recall(
            "meeting",
            k=10,
            min_importance=0.7,
        )

        # All returned nodes should have importance >= 0.7
        for node in results.nodes:
            assert node.importance >= 0.7

    def test_recall_recent_events(self, memory_api_with_data):
        """Test recalling recent events."""
        recent = memory_api_with_data.recall_recent_events(days=7, limit=5)

        assert len(recent) > 0
        # All events should be recent
        cutoff = datetime.utcnow() - timedelta(days=7)
        for event in recent:
            assert event.created_at >= cutoff


class TestSLMGConsolidation:
    """Test memory consolidation."""

    @pytest.fixture
    def memory_api_for_consolidation(self):
        """Create memory API for consolidation testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            api = MemoryAPI(db_path=str(db_path), device="cpu")

            # Add similar events (should be merged)
            api.store_event(
                "Team meeting about Q4 planning",
                importance=0.5,
            )
            api.store_event(
                "Team meeting regarding Q4 planning",
                importance=0.5,
            )

            # Add low-importance noise
            for i in range(5):
                api.store_event(
                    f"Unimportant event {i}",
                    importance=0.05,
                )

            yield api
            api.close()

    def test_consolidation_merges_similar(self, memory_api_for_consolidation):
        """Test that consolidation merges similar nodes."""
        initial_count = memory_api_for_consolidation.graph_store.count_nodes()

        # Run consolidation
        report = memory_api_for_consolidation.consolidate()

        final_count = memory_api_for_consolidation.graph_store.count_nodes()

        # Should have merged some nodes
        assert report.nodes_merged > 0 or report.nodes_pruned > 0
        assert final_count < initial_count

    def test_consolidation_prunes_low_importance(
        self, memory_api_for_consolidation
    ):
        """Test that consolidation prunes low-importance nodes."""
        report = memory_api_for_consolidation.consolidate()

        # Should have pruned low-importance nodes
        assert report.nodes_pruned > 0


class TestSLMGPersistence:
    """Test memory persistence across sessions."""

    def test_persistence_across_sessions(self):
        """Test that memories persist after closing and reopening."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"

            # Session 1: Store memories
            api1 = MemoryAPI(db_path=str(db_path), device="cpu")
            event_id = api1.store_event(
                "Important event to remember",
                importance=0.9,
            )
            api1.close()

            # Session 2: Recall memories
            api2 = MemoryAPI(db_path=str(db_path), device="cpu")
            results = api2.recall("important event", k=5)

            assert len(results.nodes) > 0
            assert results.nodes[0].id == event_id
            assert results.nodes[0].importance == 0.9

            api2.close()

    def test_export_and_stats(self):
        """Test graph export and statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            api = MemoryAPI(db_path=str(db_path), device="cpu")

            # Add some data
            api.store_event("Event 1")
            api.store_fact("X", "relates to", "Y")
            api.store_skill("skill_1", "Description", ["Step 1", "Step 2"])

            # Get stats
            stats = api.get_stats()
            assert stats["total_nodes"] >= 3
            assert stats["node_types"]["events"] >= 1
            assert stats["node_types"]["facts"] >= 1
            assert stats["node_types"]["skills"] >= 1

            # Export graph
            export_path = Path(tmpdir) / "export.json"
            api.export_graph(str(export_path))

            assert export_path.exists()
            assert export_path.stat().st_size > 0

            api.close()


class TestSLMGPerformance:
    """Performance tests for SLMG."""

    @pytest.fixture
    def large_memory_api(self):
        """Create memory API with large dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            api = MemoryAPI(db_path=str(db_path), device="cpu")

            # Add 1000 events
            for i in range(1000):
                api.store_event(
                    f"Event {i}: Meeting about various topics",
                    importance=0.3 + (i % 10) * 0.05,
                )

            yield api
            api.close()

    def test_recall_latency_with_large_dataset(self, large_memory_api):
        """Test recall latency with large dataset."""
        # Requirement: < 150ms for 50k nodes
        # We test with 1k nodes, should be much faster

        results = large_memory_api.recall("meeting", k=10)

        # Should return results quickly
        assert results.metadata["latency_ms"] < 150
        assert len(results.nodes) == 10

    def test_retrieval_accuracy(self, large_memory_api):
        """Test retrieval accuracy."""
        # Store a very specific event
        specific_id = large_memory_api.store_event(
            "Critical bug fix for authentication system",
            importance=0.9,
            tags=["bug", "auth", "critical"],
        )

        # Query should retrieve it in top results
        results = large_memory_api.recall(
            "authentication bug fix",
            k=5,
        )

        # The specific event should be in top-5
        top_ids = [node.id for node in results.nodes]
        assert specific_id in top_ids
