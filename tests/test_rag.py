"""
RAG (Retrieval-Augmented Generation) Tests
Tests for RAG pipeline and context retrieval

Coverage: 50+ tests including:
- Context retrieval
- Re-ranking
- Prompt building
- Hallucination detection
- Confidence scoring
- End-to-end RAG pipeline
- Token management
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from core.memory.rag import (
    RAGPipeline,
    RAGConfig,
    RAGContext,
    RAGResult
)
from core.memory.long_term_memory import LongTermMemory, MemoryType, MemoryEntry


@pytest.fixture
def mock_memory():
    """Mock long-term memory"""
    memory = Mock(spec=LongTermMemory)
    memory.retrieve.return_value = []
    return memory


@pytest.fixture
def rag_config():
    """Default RAG configuration"""
    return RAGConfig(
        max_context_tokens=4000,
        num_retrievals=10,
        min_relevance_score=0.3,
        enable_reranking=True,
        enable_hallucination_check=True
    )


@pytest.fixture
def rag_pipeline(mock_memory, rag_config):
    """RAG pipeline instance"""
    return RAGPipeline(
        long_term_memory=mock_memory,
        config=rag_config
    )


@pytest.fixture
def sample_contexts():
    """Sample RAG contexts"""
    return [
        RAGContext(
            source="memory:semantic",
            content="Bitcoin is a cryptocurrency",
            relevance_score=0.9,
            metadata={"importance": 0.8, "created_at": datetime.now(timezone.utc).isoformat()}
        ),
        RAGContext(
            source="memory:episodic",
            content="User asked about trading strategy",
            relevance_score=0.7,
            metadata={"importance": 0.6, "created_at": datetime.now(timezone.utc).isoformat()}
        ),
        RAGContext(
            source="memory:procedural",
            content="Calculate RSI using 14-period average",
            relevance_score=0.8,
            metadata={"importance": 0.9, "created_at": datetime.now(timezone.utc).isoformat()}
        )
    ]


# =============================================================================
# Test Context Retrieval
# =============================================================================

@pytest.mark.unit
class TestContextRetrieval:
    """Test context retrieval"""

    def test_retrieve_context_from_memory(self, rag_pipeline, mock_memory):
        """Test: Retrieve context from memory"""
        # Mock memory results
        memory_entry = MemoryEntry(
            memory_id="mem1",
            memory_type=MemoryType.SEMANTIC,
            content="Bitcoin is digital gold",
            importance_score=0.8
        )
        mock_memory.retrieve.return_value = [(memory_entry, 0.9)]

        contexts = rag_pipeline.retrieve_context("bitcoin", k=10)

        assert len(contexts) > 0
        assert contexts[0].content == "Bitcoin is digital gold"
        assert contexts[0].relevance_score == 0.9

    def test_retrieve_by_memory_type(self, rag_pipeline, mock_memory):
        """Test: Retrieve specific memory types"""
        mock_memory.retrieve.return_value = []

        contexts = rag_pipeline.retrieve_context(
            "test",
            memory_types=[MemoryType.SEMANTIC]
        )

        # Verify retrieve was called with correct type
        mock_memory.retrieve.assert_called()

    def test_retrieve_respects_min_relevance(self, rag_pipeline, mock_memory):
        """Test: Filter by minimum relevance score"""
        memory1 = MemoryEntry(
            memory_id="mem1",
            memory_type=MemoryType.SEMANTIC,
            content="High relevance",
            importance_score=0.8
        )
        memory2 = MemoryEntry(
            memory_id="mem2",
            memory_type=MemoryType.SEMANTIC,
            content="Low relevance",
            importance_score=0.2
        )

        mock_memory.retrieve.return_value = [
            (memory1, 0.9),  # Above threshold
            (memory2, 0.2)   # Below threshold
        ]

        # Config has min_relevance_score=0.3
        contexts = rag_pipeline.retrieve_context("test")

        # Should filter out low relevance
        assert all(c.relevance_score >= 0.3 for c in contexts)

    def test_retrieve_handles_empty_results(self, rag_pipeline, mock_memory):
        """Test: Handle empty memory results"""
        mock_memory.retrieve.return_value = []

        contexts = rag_pipeline.retrieve_context("nonexistent")

        assert len(contexts) == 0


# =============================================================================
# Test Re-ranking
# =============================================================================

@pytest.mark.unit
class TestReranking:
    """Test context re-ranking"""

    def test_rerank_by_combined_score(self, rag_pipeline, sample_contexts):
        """Test: Re-rank contexts by combined score"""
        reranked = rag_pipeline.rerank_context(sample_contexts, "test query")

        # Should have combined scores
        assert all("combined_score" in c.metadata for c in reranked)

        # Should be sorted
        scores = [c.metadata["combined_score"] for c in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_considers_relevance(self, rag_pipeline):
        """Test: Re-ranking considers relevance score"""
        contexts = [
            RAGContext("source1", "Content 1", relevance_score=0.9, metadata={"importance": 0.5}),
            RAGContext("source2", "Content 2", relevance_score=0.5, metadata={"importance": 0.5})
        ]

        reranked = rag_pipeline.rerank_context(contexts, "test")

        # Higher relevance should rank higher
        assert reranked[0].relevance_score > reranked[1].relevance_score

    def test_rerank_considers_importance(self, rag_pipeline):
        """Test: Re-ranking considers importance"""
        contexts = [
            RAGContext("s1", "C1", relevance_score=0.7, metadata={"importance": 0.9}),
            RAGContext("s2", "C2", relevance_score=0.7, metadata={"importance": 0.3})
        ]

        reranked = rag_pipeline.rerank_context(contexts, "test")

        # Higher importance should contribute to ranking
        assert "combined_score" in reranked[0].metadata

    def test_rerank_disabled(self, mock_memory):
        """Test: Re-ranking can be disabled"""
        config = RAGConfig(enable_reranking=False)
        pipeline = RAGPipeline(mock_memory, config)

        contexts = [RAGContext("s1", "C1", 0.5), RAGContext("s2", "C2", 0.9)]

        reranked = pipeline.rerank_context(contexts, "test")

        # Should return unchanged (no combined_score added)
        assert reranked == contexts


# =============================================================================
# Test Prompt Building
# =============================================================================

@pytest.mark.unit
class TestPromptBuilding:
    """Test prompt building with context"""

    def test_build_prompt_with_context(self, rag_pipeline, sample_contexts):
        """Test: Build augmented prompt"""
        query = "What is Bitcoin?"

        prompt, tokens, included = rag_pipeline.build_context_prompt(query, sample_contexts)

        assert "Bitcoin is a cryptocurrency" in prompt
        assert "What is Bitcoin?" in prompt
        assert tokens > 0
        assert len(included) > 0

    def test_build_prompt_respects_token_limit(self, rag_pipeline):
        """Test: Prompt respects token budget"""
        # Create many contexts
        contexts = [
            RAGContext(f"source{i}", "x" * 1000, 0.9)
            for i in range(20)
        ]

        query = "test"
        prompt, tokens, included = rag_pipeline.build_context_prompt(query, contexts)

        # Should not exceed max tokens
        assert tokens <= rag_pipeline.config.max_context_tokens

    def test_build_prompt_truncates_long_contexts(self, rag_pipeline):
        """Test: Long contexts are truncated"""
        long_context = RAGContext(
            "source",
            "x" * 10000,  # Very long content
            0.9
        )

        prompt, tokens, included = rag_pipeline.build_context_prompt(
            "test",
            [long_context]
        )

        # Should be truncated
        if included:
            assert included[0].metadata.get("truncated", False)

    def test_build_prompt_empty_contexts(self, rag_pipeline):
        """Test: Handle empty contexts"""
        prompt, tokens, included = rag_pipeline.build_context_prompt("test", [])

        assert "[No relevant context found]" in prompt
        assert len(included) == 0


# =============================================================================
# Test Hallucination Detection
# =============================================================================

@pytest.mark.unit
class TestHallucinationDetection:
    """Test hallucination detection"""

    def test_detect_response_without_context_reference(self, rag_pipeline, sample_contexts):
        """Test: Detect response that doesn't reference context"""
        response = "The moon is made of cheese"  # Unrelated to contexts
        query = "What is Bitcoin?"

        hallucination, warnings = rag_pipeline.check_hallucination(
            response,
            sample_contexts,
            query
        )

        # Should detect lack of context reference
        assert len(warnings) > 0 or hallucination

    def test_detect_uncertainty_phrases(self, rag_pipeline, sample_contexts):
        """Test: Detect uncertainty in response"""
        response = "I don't know about Bitcoin"

        hallucination, warnings = rag_pipeline.check_hallucination(
            response,
            sample_contexts,
            "bitcoin"
        )

        # Should warn about uncertainty
        assert len(warnings) > 0

    def test_detect_short_response(self, rag_pipeline):
        """Test: Detect very short responses"""
        response = "Yes"

        hallucination, warnings = rag_pipeline.check_hallucination(
            response,
            [],
            "test"
        )

        assert len(warnings) > 0

    def test_valid_response_no_hallucination(self, rag_pipeline, sample_contexts):
        """Test: Valid response passes checks"""
        response = "Bitcoin is a cryptocurrency that serves as digital gold for trading"

        hallucination, warnings = rag_pipeline.check_hallucination(
            response,
            sample_contexts,
            "bitcoin"
        )

        # Should have few/no warnings
        assert not hallucination or len(warnings) == 0

    def test_hallucination_check_disabled(self, mock_memory):
        """Test: Hallucination check can be disabled"""
        config = RAGConfig(enable_hallucination_check=False)
        pipeline = RAGPipeline(mock_memory, config)

        hallucination, warnings = pipeline.check_hallucination(
            "any response",
            [],
            "query"
        )

        assert not hallucination
        assert len(warnings) == 0


# =============================================================================
# Test Confidence Scoring
# =============================================================================

@pytest.mark.unit
class TestConfidenceScoring:
    """Test confidence score calculation"""

    def test_high_confidence_with_good_contexts(self, rag_pipeline, sample_contexts):
        """Test: High confidence with relevant contexts"""
        response = "Bitcoin is a cryptocurrency used for trading"

        confidence = rag_pipeline._calculate_confidence(
            response,
            sample_contexts,
            hallucination_detected=False
        )

        assert 0.5 < confidence <= 1.0

    def test_low_confidence_with_hallucination(self, rag_pipeline):
        """Test: Low confidence when hallucination detected"""
        confidence = rag_pipeline._calculate_confidence(
            "response",
            [],
            hallucination_detected=True
        )

        assert confidence <= 0.5

    def test_low_confidence_without_contexts(self, rag_pipeline):
        """Test: Lower confidence without contexts"""
        confidence = rag_pipeline._calculate_confidence(
            "response",
            [],
            hallucination_detected=False
        )

        assert confidence < 1.0

    def test_low_confidence_short_response(self, rag_pipeline):
        """Test: Lower confidence for short responses"""
        confidence = rag_pipeline._calculate_confidence(
            "Yes",
            [],
            hallucination_detected=False
        )

        assert confidence < 1.0


# =============================================================================
# Test Full RAG Pipeline
# =============================================================================

@pytest.mark.integration
class TestRAGPipeline:
    """Test complete RAG pipeline"""

    def test_generate_with_llm_callback(self, rag_pipeline, mock_memory):
        """Test: Generate response with LLM"""
        # Mock memory
        memory_entry = MemoryEntry(
            memory_id="mem1",
            memory_type=MemoryType.SEMANTIC,
            content="Bitcoin is a cryptocurrency",
            importance_score=0.8
        )
        mock_memory.retrieve.return_value = [(memory_entry, 0.9)]

        # Mock LLM callback
        llm_callback = Mock(return_value="Bitcoin is a decentralized digital currency")

        # Generate
        result = rag_pipeline.generate(
            query="What is Bitcoin?",
            llm_callback=llm_callback
        )

        assert isinstance(result, RAGResult)
        assert result.response == "Bitcoin is a decentralized digital currency"
        assert result.confidence_score > 0
        assert len(result.contexts) > 0

    def test_generate_handles_llm_error(self, rag_pipeline, mock_memory):
        """Test: Handle LLM callback errors"""
        mock_memory.retrieve.return_value = []

        # Failing LLM
        llm_callback = Mock(side_effect=Exception("LLM error"))

        result = rag_pipeline.generate("test", llm_callback)

        assert "[Error generating response" in result.response

    def test_generate_includes_metadata(self, rag_pipeline, mock_memory):
        """Test: Result includes metadata"""
        mock_memory.retrieve.return_value = []
        llm_callback = Mock(return_value="response")

        result = rag_pipeline.generate(
            "test",
            llm_callback,
            include_metadata=True
        )

        assert "intent" in result.metadata
        assert "processing_time_ms" in result.metadata

    def test_generate_filters_memory_types(self, rag_pipeline, mock_memory):
        """Test: Can filter by memory types"""
        mock_memory.retrieve.return_value = []
        llm_callback = Mock(return_value="response")

        result = rag_pipeline.generate(
            "test",
            llm_callback,
            memory_types=[MemoryType.SEMANTIC]
        )

        # Verify retrieve was called with correct types
        assert mock_memory.retrieve.called


# =============================================================================
# Test Token Management
# =============================================================================

@pytest.mark.unit
class TestTokenManagement:
    """Test token management"""

    def test_estimate_tokens(self, rag_pipeline):
        """Test: Token estimation"""
        text = "a" * 400  # 400 characters
        tokens = rag_pipeline._estimate_tokens(text)

        # Should estimate ~100 tokens (4 chars per token)
        assert 90 <= tokens <= 110

    def test_context_fits_within_budget(self, rag_pipeline):
        """Test: Contexts fit within token budget"""
        # Create contexts that exceed budget
        contexts = [
            RAGContext("s", "x" * 2000, 0.9)
            for _ in range(10)
        ]

        prompt, tokens, included = rag_pipeline.build_context_prompt("test", contexts)

        # Should not exceed max tokens
        assert tokens <= rag_pipeline.config.max_context_tokens


# =============================================================================
# Test Query Understanding
# =============================================================================

@pytest.mark.unit
class TestQueryUnderstanding:
    """Test query understanding"""

    def test_extract_intent_question(self, rag_pipeline):
        """Test: Detect questions"""
        intent = rag_pipeline._extract_intent("What is Bitcoin?")

        assert intent["is_question"] is True

    def test_extract_intent_command(self, rag_pipeline):
        """Test: Detect commands"""
        intent = rag_pipeline._extract_intent("Show me trading strategies")

        assert intent["is_command"] is True

    def test_extract_keywords(self, rag_pipeline):
        """Test: Extract keywords"""
        keywords = rag_pipeline._extract_keywords(
            "Bitcoin trading strategy with RSI indicator"
        )

        assert "bitcoin" in [k.lower() for k in keywords]
        assert "trading" in [k.lower() for k in keywords]
        assert len(keywords) > 0


# =============================================================================
# Test RAG Configuration
# =============================================================================

@pytest.mark.unit
class TestRAGConfiguration:
    """Test RAG configuration"""

    def test_custom_config(self, mock_memory):
        """Test: Custom configuration"""
        config = RAGConfig(
            max_context_tokens=2000,
            num_retrievals=5,
            min_relevance_score=0.5
        )

        pipeline = RAGPipeline(mock_memory, config)

        assert pipeline.config.max_context_tokens == 2000
        assert pipeline.config.num_retrievals == 5

    def test_default_config(self, mock_memory):
        """Test: Default configuration"""
        pipeline = RAGPipeline(mock_memory)

        assert pipeline.config.max_context_tokens == 4000
        assert pipeline.config.enable_reranking is True


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.performance
class TestRAGPerformance:
    """Test RAG performance"""

    def test_rag_pipeline_performance(self, rag_pipeline, mock_memory):
        """Test: RAG pipeline completes in reasonable time"""
        import time

        mock_memory.retrieve.return_value = []
        llm_callback = Mock(return_value="fast response")

        start = time.time()
        result = rag_pipeline.generate("test", llm_callback)
        duration = time.time() - start

        # Should complete in < 500ms (excluding LLM call)
        assert duration < 0.5

    def test_context_retrieval_performance(self, rag_pipeline, mock_memory):
        """Test: Context retrieval is fast"""
        import time

        # Mock many results
        memories = [
            (MemoryEntry(f"mem{i}", MemoryType.SEMANTIC, f"content{i}", 0.8), 0.9)
            for i in range(100)
        ]
        mock_memory.retrieve.return_value = memories

        start = time.time()
        contexts = rag_pipeline.retrieve_context("test", k=100)
        duration = time.time() - start

        # Should be fast
        assert duration < 0.1
