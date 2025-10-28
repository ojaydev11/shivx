"""
Tests for WebSocket functionality
"""

import pytest
import asyncio
import json
from fastapi import WebSocket
from unittest.mock import Mock, AsyncMock, patch

from app.websocket import (
    ConnectionManager,
    websocket_endpoint,
    broadcast_agent_status,
    broadcast_task_update,
    verify_token,
)


class TestConnectionManager:
    """Test WebSocket connection manager"""

    def test_init(self):
        """Test connection manager initialization"""
        manager = ConnectionManager()
        assert manager.active_connections == {}
        assert manager.user_connections == {}
        assert manager.rate_limit == 100

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connection acceptance"""
        manager = ConnectionManager()
        websocket = Mock(spec=WebSocket)
        websocket.accept = AsyncMock()

        await manager.connect(websocket, "conn1", "user1")

        websocket.accept.assert_called_once()
        assert "conn1" in manager.active_connections
        assert "user1" in manager.user_connections
        assert "conn1" in manager.user_connections["user1"]

    def test_disconnect(self):
        """Test connection removal"""
        manager = ConnectionManager()
        websocket = Mock(spec=WebSocket)
        manager.active_connections["conn1"] = websocket
        manager.user_connections["user1"] = {"conn1"}

        manager.disconnect("conn1", "user1")

        assert "conn1" not in manager.active_connections
        assert "conn1" not in manager.user_connections["user1"]

    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending message to specific connection"""
        manager = ConnectionManager()
        websocket = Mock(spec=WebSocket)
        websocket.send_json = AsyncMock()
        manager.active_connections["conn1"] = websocket

        message = {"type": "test", "data": {"value": 123}}
        await manager.send_personal_message(message, "conn1")

        websocket.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting to all connections"""
        manager = ConnectionManager()
        ws1 = Mock(spec=WebSocket)
        ws1.send_json = AsyncMock()
        ws2 = Mock(spec=WebSocket)
        ws2.send_json = AsyncMock()

        manager.active_connections["conn1"] = ws1
        manager.active_connections["conn2"] = ws2

        message = {"type": "broadcast", "data": {}}
        await manager.broadcast(message)

        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_user(self):
        """Test broadcasting to specific user's connections"""
        manager = ConnectionManager()
        ws1 = Mock(spec=WebSocket)
        ws1.send_json = AsyncMock()
        ws2 = Mock(spec=WebSocket)
        ws2.send_json = AsyncMock()

        manager.active_connections["conn1"] = ws1
        manager.active_connections["conn2"] = ws2
        manager.user_connections["user1"] = {"conn1", "conn2"}

        message = {"type": "user_message", "data": {}}
        await manager.broadcast_to_user(message, "user1")

        ws1.send_json.assert_called_once_with(message)
        ws2.send_json.assert_called_once_with(message)

    def test_rate_limit_check(self):
        """Test rate limiting"""
        manager = ConnectionManager()
        manager.rate_limit = 5

        # Should pass initially
        for i in range(5):
            assert manager.check_rate_limit("conn1") is True
            manager.increment_message_count("conn1")

        # Should fail after limit
        assert manager.check_rate_limit("conn1") is False

    @pytest.mark.asyncio
    async def test_reset_rate_limits(self):
        """Test rate limit reset"""
        manager = ConnectionManager()
        manager.message_counts["conn1"] = 100
        manager.message_counts["conn2"] = 50

        await manager.reset_rate_limits()

        assert len(manager.message_counts) == 0


class TestTokenVerification:
    """Test JWT token verification"""

    @patch("app.websocket.jwt.decode")
    def test_verify_token_success(self, mock_decode):
        """Test successful token verification"""
        mock_decode.return_value = {"sub": "user123", "exp": 9999999999}

        payload = verify_token("valid_token")

        assert payload is not None
        assert payload["sub"] == "user123"

    @patch("app.websocket.jwt.decode")
    def test_verify_token_failure(self, mock_decode):
        """Test failed token verification"""
        from jose import JWTError

        mock_decode.side_effect = JWTError("Invalid token")

        payload = verify_token("invalid_token")

        assert payload is None


class TestBroadcastFunctions:
    """Test broadcast utility functions"""

    @pytest.mark.asyncio
    async def test_broadcast_agent_status(self):
        """Test agent status broadcast"""
        with patch("app.websocket.manager.broadcast") as mock_broadcast:
            await broadcast_agent_status("agent1", "active", {"cpu": 50})

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0][0]
            assert call_args["type"] == "agent_status"
            assert call_args["data"]["id"] == "agent1"
            assert call_args["data"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_broadcast_task_update(self):
        """Test task update broadcast"""
        with patch("app.websocket.manager.broadcast") as mock_broadcast:
            await broadcast_task_update("task1", "completed", {"result": "success"})

            mock_broadcast.assert_called_once()
            call_args = mock_broadcast.call_args[0][0]
            assert call_args["type"] == "task_update"
            assert call_args["data"]["id"] == "task1"
            assert call_args["data"]["status"] == "completed"


@pytest.mark.asyncio
async def test_websocket_endpoint_no_token():
    """Test WebSocket endpoint without token"""
    websocket = Mock(spec=WebSocket)
    websocket.close = AsyncMock()

    await websocket_endpoint(websocket, token=None)

    websocket.close.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_endpoint_invalid_token():
    """Test WebSocket endpoint with invalid token"""
    websocket = Mock(spec=WebSocket)
    websocket.close = AsyncMock()

    with patch("app.websocket.verify_token", return_value=None):
        await websocket_endpoint(websocket, token="invalid")

    websocket.close.assert_called_once()
