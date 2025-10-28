"""
Conversation Memory Tests
Tests for conversation session management and message history

Coverage: 20+ tests including:
- Session creation and management
- Message storage and retrieval
- Context building for LLM
- Session expiration
- Conversation statistics
- Persistence
"""

import pytest
import tempfile
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.memory.conversation_memory import (
    ConversationMemory,
    ConversationSession,
    ConversationManager,
    Message
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def conversation_memory(temp_storage):
    """Conversation memory instance"""
    return ConversationMemory(
        storage_path=temp_storage,
        default_timeout_minutes=60
    )


@pytest.fixture
def sample_session(conversation_memory):
    """Create sample session"""
    return conversation_memory.create_session(
        user_id="test_user",
        title="Test Session"
    )


# =============================================================================
# Test Session Creation
# =============================================================================

@pytest.mark.unit
class TestSessionCreation:
    """Test session creation"""

    def test_create_session_basic(self, conversation_memory):
        """Test: Create basic session"""
        session = conversation_memory.create_session()

        assert session is not None
        assert session.session_id is not None
        assert session.is_active is True

    def test_create_session_with_user_id(self, conversation_memory):
        """Test: Create session with user ID"""
        session = conversation_memory.create_session(
            user_id="user123",
            title="My Conversation"
        )

        assert session.user_id == "user123"
        assert session.title == "My Conversation"

    def test_create_session_with_custom_timeout(self, conversation_memory):
        """Test: Create session with custom timeout"""
        session = conversation_memory.create_session(
            timeout_minutes=30
        )

        assert session.timeout_minutes == 30

    def test_session_has_unique_id(self, conversation_memory):
        """Test: Each session has unique ID"""
        session1 = conversation_memory.create_session()
        session2 = conversation_memory.create_session()

        assert session1.session_id != session2.session_id


# =============================================================================
# Test Message Operations
# =============================================================================

@pytest.mark.unit
class TestMessageOperations:
    """Test message operations"""

    def test_add_user_message(self, sample_session):
        """Test: Add user message"""
        message = sample_session.add_message(
            role="user",
            content="Hello, how are you?"
        )

        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert len(sample_session.messages) == 1

    def test_add_assistant_message(self, sample_session):
        """Test: Add assistant message"""
        message = sample_session.add_message(
            role="assistant",
            content="I'm doing well, thank you!"
        )

        assert message.role == "assistant"
        assert len(sample_session.messages) == 1

    def test_add_multiple_messages(self, sample_session):
        """Test: Add multiple messages"""
        sample_session.add_message("user", "Message 1")
        sample_session.add_message("assistant", "Response 1")
        sample_session.add_message("user", "Message 2")

        assert len(sample_session.messages) == 3

    def test_add_message_with_metadata(self, sample_session):
        """Test: Add message with metadata"""
        message = sample_session.add_message(
            role="user",
            content="Test",
            tokens_used=50,
            model_used="gpt-4"
        )

        assert message.tokens_used == 50
        assert message.model_used == "gpt-4"

    def test_message_has_timestamp(self, sample_session):
        """Test: Message has creation timestamp"""
        message = sample_session.add_message("user", "Test")

        assert message.created_at is not None
        assert isinstance(message.created_at, datetime)


# =============================================================================
# Test Message Retrieval
# =============================================================================

@pytest.mark.unit
class TestMessageRetrieval:
    """Test message retrieval"""

    def test_get_all_messages(self, sample_session):
        """Test: Get all messages"""
        sample_session.add_message("user", "Message 1")
        sample_session.add_message("assistant", "Response 1")
        sample_session.add_message("user", "Message 2")

        messages = sample_session.get_messages()

        assert len(messages) == 3

    def test_get_last_n_messages(self, sample_session):
        """Test: Get last N messages"""
        for i in range(10):
            sample_session.add_message("user", f"Message {i}")

        messages = sample_session.get_messages(last_n=3)

        assert len(messages) == 3
        assert messages[-1].content == "Message 9"

    def test_filter_messages_by_role(self, sample_session):
        """Test: Filter messages by role"""
        sample_session.add_message("user", "User 1")
        sample_session.add_message("assistant", "Assistant 1")
        sample_session.add_message("user", "User 2")

        user_messages = sample_session.get_messages(role_filter="user")

        assert len(user_messages) == 2
        assert all(m.role == "user" for m in user_messages)


# =============================================================================
# Test Context Building
# =============================================================================

@pytest.mark.unit
class TestContextBuilding:
    """Test context building for LLM"""

    def test_get_context_for_llm(self, sample_session):
        """Test: Build context for LLM"""
        sample_session.add_message("user", "Hello")
        sample_session.add_message("assistant", "Hi there!")
        sample_session.add_message("user", "How are you?")

        context = sample_session.get_context(max_messages=10)

        assert len(context) == 3
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello"

    def test_context_respects_max_messages(self, sample_session):
        """Test: Context respects max messages limit"""
        for i in range(10):
            sample_session.add_message("user", f"Message {i}")

        context = sample_session.get_context(max_messages=5)

        assert len(context) <= 5

    def test_context_respects_token_limit(self, sample_session):
        """Test: Context respects token limit"""
        # Add long messages
        for i in range(5):
            sample_session.add_message("user", "x" * 1000)

        context = sample_session.get_context(max_tokens=100)

        # Should truncate to fit budget
        total_length = sum(len(m["content"]) for m in context)
        estimated_tokens = total_length // 4

        # Some context should be returned but within budget
        assert estimated_tokens <= 120  # Allow some overhead


# =============================================================================
# Test Session Management
# =============================================================================

@pytest.mark.unit
class TestSessionManagement:
    """Test session management"""

    def test_get_session_by_id(self, conversation_memory, sample_session):
        """Test: Retrieve session by ID"""
        retrieved = conversation_memory.get_session(sample_session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == sample_session.session_id

    def test_get_user_sessions(self, conversation_memory):
        """Test: Get all sessions for a user"""
        conversation_memory.create_session(user_id="user1", title="Session 1")
        conversation_memory.create_session(user_id="user1", title="Session 2")
        conversation_memory.create_session(user_id="user2", title="Session 3")

        user1_sessions = conversation_memory.get_user_sessions("user1")

        assert len(user1_sessions) == 2
        assert all(s.user_id == "user1" for s in user1_sessions)

    def test_get_active_sessions_only(self, conversation_memory):
        """Test: Get only active sessions"""
        session1 = conversation_memory.create_session(user_id="user1")
        session2 = conversation_memory.create_session(user_id="user1")

        # End one session
        conversation_memory.end_session(session1.session_id)

        active = conversation_memory.get_user_sessions("user1", active_only=True)

        assert len(active) == 1
        assert active[0].session_id == session2.session_id

    def test_end_session(self, conversation_memory, sample_session):
        """Test: End session"""
        result = conversation_memory.end_session(sample_session.session_id)

        assert result is True
        assert sample_session.is_active is False


# =============================================================================
# Test Session Expiration
# =============================================================================

@pytest.mark.unit
class TestSessionExpiration:
    """Test session expiration"""

    def test_session_not_expired_initially(self, sample_session):
        """Test: New session is not expired"""
        assert sample_session.is_expired() is False

    def test_session_expires_after_timeout(self):
        """Test: Session expires after timeout"""
        session = ConversationSession(timeout_minutes=1)

        # Manually set old update time
        session.updated_at = datetime.now(timezone.utc) - timedelta(minutes=2)

        assert session.is_expired() is True

    def test_cleanup_expired_sessions(self, conversation_memory):
        """Test: Cleanup expired inactive sessions"""
        # Create session and make it expired
        session = conversation_memory.create_session(timeout_minutes=1)
        session_id = session.session_id

        # End it and make it expired
        conversation_memory.end_session(session_id)
        session.updated_at = datetime.now(timezone.utc) - timedelta(minutes=2)

        # Cleanup
        count = conversation_memory.cleanup_expired_sessions()

        assert count >= 1
        # Session should be removed
        assert conversation_memory.get_session(session_id) is None


# =============================================================================
# Test Session Statistics
# =============================================================================

@pytest.mark.unit
class TestSessionStatistics:
    """Test session statistics"""

    def test_calculate_session_stats(self, sample_session):
        """Test: Calculate session statistics"""
        sample_session.add_message("user", "Message 1", tokens_used=10)
        sample_session.add_message("assistant", "Response 1", tokens_used=15)
        sample_session.add_message("user", "Message 2", tokens_used=8)

        stats = sample_session.calculate_stats()

        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 1
        assert stats["total_tokens"] == 33

    def test_memory_stats(self, conversation_memory):
        """Test: Get memory statistics"""
        conversation_memory.create_session(user_id="user1")
        conversation_memory.create_session(user_id="user2")

        stats = conversation_memory.get_stats()

        assert stats["total_sessions"] >= 2
        assert "active_sessions" in stats
        assert "total_messages" in stats


# =============================================================================
# Test Persistence
# =============================================================================

@pytest.mark.integration
class TestPersistence:
    """Test conversation persistence"""

    def test_session_persists_across_restarts(self, temp_storage):
        """Test: Sessions persist across instances"""
        # Create first instance
        memory1 = ConversationMemory(storage_path=temp_storage)
        session = memory1.create_session(user_id="user1", title="Persistent")
        session_id = session.session_id

        # Add messages
        memory1.add_message(session_id, "user", "Test message")

        # Create new instance
        memory2 = ConversationMemory(storage_path=temp_storage)

        # Should load existing session
        loaded_session = memory2.get_session(session_id)
        assert loaded_session is not None
        assert loaded_session.title == "Persistent"
        assert len(loaded_session.messages) == 1

    def test_messages_persist(self, temp_storage):
        """Test: Messages persist"""
        memory1 = ConversationMemory(storage_path=temp_storage)
        session = memory1.create_session()
        session_id = session.session_id

        memory1.add_message(session_id, "user", "Message 1")
        memory1.add_message(session_id, "assistant", "Response 1")

        # Reload
        memory2 = ConversationMemory(storage_path=temp_storage)
        loaded = memory2.get_session(session_id)

        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Message 1"


# =============================================================================
# Test Conversation Manager
# =============================================================================

@pytest.mark.unit
class TestConversationManager:
    """Test conversation manager"""

    def test_start_conversation(self, temp_storage):
        """Test: Start new conversation"""
        manager = ConversationManager(storage_path=temp_storage)

        session_id = manager.start_conversation(user_id="user1", title="Test")

        assert session_id is not None
        assert manager._active_session == session_id

    def test_send_message(self, temp_storage):
        """Test: Send message in conversation"""
        manager = ConversationManager(storage_path=temp_storage)
        session_id = manager.start_conversation()

        message = manager.send_message("Hello", session_id=session_id)

        assert message is not None
        assert message.content == "Hello"

    def test_send_message_uses_active_session(self, temp_storage):
        """Test: Send message uses active session"""
        manager = ConversationManager(storage_path=temp_storage)
        manager.start_conversation()

        # Don't specify session_id
        message = manager.send_message("Test")

        assert message is not None

    def test_get_context_from_manager(self, temp_storage):
        """Test: Get context from manager"""
        manager = ConversationManager(storage_path=temp_storage)
        session_id = manager.start_conversation()

        manager.send_message("Message 1")
        manager.send_message("Message 2", role="assistant")

        context = manager.get_context()

        assert len(context) == 2

    def test_end_conversation(self, temp_storage):
        """Test: End conversation"""
        manager = ConversationManager(storage_path=temp_storage)
        session_id = manager.start_conversation()

        result = manager.end_conversation(session_id)

        assert result is True
        assert manager._active_session is None


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestConversationMemoryIntegration:
    """Integration tests"""

    def test_full_conversation_flow(self, conversation_memory):
        """Test: Complete conversation flow"""
        # Create session
        session = conversation_memory.create_session(
            user_id="user1",
            title="Full Flow Test"
        )

        # Add messages
        conversation_memory.add_message(session.session_id, "user", "Hello")
        conversation_memory.add_message(session.session_id, "assistant", "Hi!")
        conversation_memory.add_message(session.session_id, "user", "How are you?")

        # Get context
        context = session.get_context()

        assert len(context) == 3

        # Get stats
        stats = session.calculate_stats()
        assert stats["total_messages"] == 3

        # End session
        conversation_memory.end_session(session.session_id)
        assert session.is_active is False

    def test_multi_user_sessions(self, conversation_memory):
        """Test: Multiple users with separate sessions"""
        user1_session = conversation_memory.create_session(user_id="user1")
        user2_session = conversation_memory.create_session(user_id="user2")

        conversation_memory.add_message(user1_session.session_id, "user", "User 1 message")
        conversation_memory.add_message(user2_session.session_id, "user", "User 2 message")

        user1_sessions = conversation_memory.get_user_sessions("user1")
        user2_sessions = conversation_memory.get_user_sessions("user2")

        assert len(user1_sessions) == 1
        assert len(user2_sessions) == 1
        assert user1_sessions[0].session_id != user2_sessions[0].session_id
