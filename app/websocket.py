"""
ShivX WebSocket Server
Provides real-time communication for frontend updates

Features:
- JWT authentication
- Connection management
- Event broadcasting
- Heartbeat/ping-pong
- Rate limiting
"""

import json
import asyncio
import logging
from typing import Dict, Set
from datetime import datetime
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect, Query, status
from jose import jwt, JWTError

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.message_counts: Dict[str, int] = defaultdict(int)
        self.rate_limit = 100  # messages per minute

    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept and register a new connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.user_connections[user_id].add(connection_id)
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "data": {"status": "connected", "connection_id": connection_id},
                "timestamp": datetime.utcnow().isoformat(),
            },
            connection_id,
        )

    def disconnect(self, connection_id: str, user_id: str):
        """Remove a connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal_message(self, message: dict, connection_id: str):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connections"""
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

    async def broadcast_to_user(self, message: dict, user_id: str):
        """Broadcast message to all connections of a specific user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(message, connection_id)

    def check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection exceeds rate limit"""
        count = self.message_counts.get(connection_id, 0)
        return count < self.rate_limit

    def increment_message_count(self, connection_id: str):
        """Increment message count for rate limiting"""
        self.message_counts[connection_id] += 1

    async def reset_rate_limits(self):
        """Reset rate limit counters (call every minute)"""
        self.message_counts.clear()


# Global connection manager
manager = ConnectionManager()


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None


async def websocket_endpoint(websocket: WebSocket, token: str = Query(None)):
    """
    WebSocket endpoint for real-time updates

    Args:
        websocket: WebSocket connection
        token: JWT authentication token
    """
    import uuid

    connection_id = str(uuid.uuid4())
    user_id = None

    # Authenticate
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    payload = verify_token(token)
    if not payload:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_id = payload.get("sub")  # User ID from token

    # Connect
    await manager.connect(websocket, connection_id, user_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            # Check rate limit
            if not manager.check_rate_limit(connection_id):
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "data": {"message": "Rate limit exceeded"},
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    connection_id,
                )
                continue

            manager.increment_message_count(connection_id)

            try:
                message = json.loads(data)
                message_type = message.get("type")

                # Handle different message types
                if message_type == "ping":
                    # Respond to ping with pong
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        connection_id,
                    )

                elif message_type == "subscribe":
                    # Subscribe to specific events
                    channels = message.get("channels", [])
                    await manager.send_personal_message(
                        {
                            "type": "subscribed",
                            "data": {"channels": channels},
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        connection_id,
                    )

                elif message_type == "command":
                    # Handle commands (e.g., request agent status)
                    command = message.get("command")
                    logger.info(f"WebSocket command from {user_id}: {command}")
                    # Process command and send response
                    # This would integrate with your agent system

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from {connection_id}")

    except WebSocketDisconnect:
        manager.disconnect(connection_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        manager.disconnect(connection_id, user_id)


# Utility functions for broadcasting events


async def broadcast_agent_status(agent_id: str, status: str, data: dict):
    """Broadcast agent status update"""
    await manager.broadcast(
        {
            "type": "agent_status",
            "data": {"id": agent_id, "status": status, **data},
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_task_update(task_id: str, status: str, data: dict):
    """Broadcast task update"""
    await manager.broadcast(
        {
            "type": "task_update",
            "data": {"id": task_id, "status": status, **data},
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_health_alert(health_data: dict):
    """Broadcast system health alert"""
    await manager.broadcast(
        {
            "type": "health_alert",
            "data": health_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_log_entry(log_entry: dict):
    """Broadcast log entry"""
    await manager.broadcast(
        {
            "type": "log_entry",
            "data": log_entry,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_trade_executed(trade_data: dict):
    """Broadcast trade execution"""
    await manager.broadcast(
        {
            "type": "trade_executed",
            "data": trade_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


async def broadcast_position_update(position_data: dict):
    """Broadcast position update"""
    await manager.broadcast(
        {
            "type": "position_update",
            "data": position_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# Background task to reset rate limits
async def rate_limit_reset_task():
    """Background task to reset rate limits every minute"""
    while True:
        await asyncio.sleep(60)
        await manager.reset_rate_limits()
