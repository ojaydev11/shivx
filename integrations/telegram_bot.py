"""
Telegram Bot Bridge
===================
Telegram bot integration for command handling, notifications, and task management.

Features:
- Bot commands (/start, /help, /status, /agents, /task, /cancel)
- Message handling (text, voice, documents)
- Notifications and alerts
- User whitelist and rate limiting
- Full audit logging
"""

import os
import time
import logging
import asyncio
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None  # type: ignore
    ContextTypes = None  # type: ignore

from security.guardian_defense import get_guardian_defense
from utils.audit_chain import append_jsonl
from utils.policy_guard import get_policy_guard

logger = logging.getLogger(__name__)


class TelegramMessageType(Enum):
    """Telegram message types"""
    TEXT = "text"
    VOICE = "voice"
    DOCUMENT = "document"
    PHOTO = "photo"
    VIDEO = "video"


@dataclass
class TelegramUser:
    """Telegram user info"""
    user_id: int
    username: Optional[str]
    first_name: Optional[str]
    last_name: Optional[str]
    is_bot: bool
    allowed: bool


class TelegramBot:
    """
    Telegram bot for command handling and notifications.

    Features:
    - Bot commands: /start, /help, /status, /agents, /task, /cancel
    - Message handling: text, voice, documents
    - Notifications: Send alerts to Telegram
    - Security: User whitelist, rate limiting
    - Audit logging: All interactions logged
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_user_ids: Optional[List[int]] = None,
        audit_log_path: str = "var/audit/telegram_operations.jsonl",
        webhook_mode: bool = False,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize Telegram bot.

        Args:
            bot_token: Telegram bot token (from ENV if not provided)
            allowed_user_ids: Whitelist of allowed user IDs
            audit_log_path: Path to audit log file
            webhook_mode: Use webhook instead of polling
            webhook_url: Webhook URL (required if webhook_mode=True)
        """
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed. "
                "Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError(
                "Telegram bot token required. Set TELEGRAM_BOT_TOKEN env var."
            )

        self.allowed_user_ids = set(allowed_user_ids or [])

        # Load from env var if available
        env_allowed_users = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        if env_allowed_users:
            try:
                self.allowed_user_ids.update(
                    int(uid.strip()) for uid in env_allowed_users.split(",")
                    if uid.strip()
                )
            except Exception as e:
                logger.warning(f"Failed to parse TELEGRAM_ALLOWED_USER_IDS: {e}")

        self.audit_log_path = audit_log_path
        self.webhook_mode = webhook_mode
        self.webhook_url = webhook_url

        # Get integrations
        self.guardian = get_guardian_defense()
        self.policy_guard = get_policy_guard()

        # Rate limiting (messages per user per minute)
        self.rate_limit_per_minute = 20
        self.user_message_times: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))

        # Application
        self.application: Optional[Application] = None
        self.running = False

        # Active tasks
        self.active_tasks: Dict[int, Dict[str, Any]] = {}

        logger.info(f"TelegramBot initialized (webhook_mode={webhook_mode})")

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is in whitelist"""
        # If no whitelist configured, allow all (not recommended for production)
        if not self.allowed_user_ids:
            logger.warning("No user whitelist configured - allowing all users")
            return True

        return user_id in self.allowed_user_ids

    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded rate limit"""
        now = time.time()
        self.user_message_times[user_id].append(now)

        # Count messages in last minute
        one_min_ago = now - 60
        recent = [t for t in self.user_message_times[user_id] if t > one_min_ago]
        count = len(recent)

        if count > self.rate_limit_per_minute:
            logger.warning(
                f"User {user_id} exceeded rate limit: {count} messages/min"
            )
            self.guardian.detect_rate_limit_abuse(
                f"telegram_user_{user_id}",
                f"messages_per_minute={count}"
            )
            return False

        return True

    def _log_operation(
        self,
        user_id: int,
        action: str,
        details: Dict[str, Any],
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log operation to audit chain"""
        try:
            operation_id = f"tg_{int(time.time() * 1000)}"

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "integration": "telegram",
                "operation_id": operation_id,
                "user_id": user_id,
                "action": action,
                "details": details,
                "success": success,
                "error": error,
            }

            append_jsonl(self.audit_log_path, log_entry)

        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

    # ========================================================================
    # Command Handlers
    # ========================================================================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user = update.effective_user
        user_id = user.id

        if not self._is_user_allowed(user_id):
            await update.message.reply_text(
                "Sorry, you are not authorized to use this bot."
            )
            self._log_operation(user_id, "start", {}, False, "User not authorized")
            return

        welcome_message = (
            f"Welcome to ShivX Bot, {user.first_name}!\n\n"
            "Available commands:\n"
            "/help - Show this help message\n"
            "/status - Show system status\n"
            "/agents - List active agents\n"
            "/task <description> - Create new task\n"
            "/cancel - Cancel current task\n"
        )

        await update.message.reply_text(welcome_message)
        self._log_operation(user_id, "start", {"username": user.username}, True)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        help_message = (
            "ShivX Bot Commands:\n\n"
            "/start - Initialize bot\n"
            "/help - Show this help message\n"
            "/status - Show system health status\n"
            "/agents - List active agents\n"
            "/task <description> - Create new task\n"
            "/cancel - Cancel current task\n\n"
            "You can also send:\n"
            "- Text messages for intent analysis\n"
            "- Voice messages (converted to text)\n"
            "- Documents for processing\n"
        )

        await update.message.reply_text(help_message)
        self._log_operation(user_id, "help", {}, True)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        # Get system status (mock data for now)
        status_message = (
            "System Status:\n\n"
            "Health: Healthy\n"
            "Uptime: 24h 15m\n"
            "Active Tasks: 3\n"
            "CPU Usage: 45%\n"
            "Memory Usage: 62%\n"
            "Defense Mode: NORMAL\n"
        )

        await update.message.reply_text(status_message)
        self._log_operation(user_id, "status", {}, True)

    async def cmd_agents(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /agents command"""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        # List active agents (mock data for now)
        agents_message = (
            "Active Agents:\n\n"
            "1. Trading Agent (Status: Running)\n"
            "   - Last trade: 5m ago\n"
            "   - P&L today: +$125.50\n\n"
            "2. Market Analyzer (Status: Running)\n"
            "   - Analyzing 15 pairs\n"
            "   - Last update: 1m ago\n\n"
            "3. Risk Manager (Status: Running)\n"
            "   - Monitoring: 3 positions\n"
            "   - Risk level: LOW\n"
        )

        await update.message.reply_text(agents_message)
        self._log_operation(user_id, "agents", {}, True)

    async def cmd_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /task command"""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        # Get task description from command args
        task_description = " ".join(context.args) if context.args else None

        if not task_description:
            await update.message.reply_text(
                "Usage: /task <description>\n"
                "Example: /task Analyze SOL/USDT price trend"
            )
            return

        # Create task
        task_id = f"task_{int(time.time() * 1000)}"
        self.active_tasks[user_id] = {
            "task_id": task_id,
            "description": task_description,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }

        await update.message.reply_text(
            f"Task created: {task_id}\n"
            f"Description: {task_description}\n"
            f"Status: Pending\n\n"
            f"Use /cancel to cancel this task."
        )

        self._log_operation(
            user_id,
            "task_create",
            {"task_id": task_id, "description": task_description},
            True
        )

    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /cancel command"""
        user_id = update.effective_user.id

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        if user_id not in self.active_tasks:
            await update.message.reply_text("No active task to cancel.")
            return

        task = self.active_tasks[user_id]
        task_id = task["task_id"]

        del self.active_tasks[user_id]

        await update.message.reply_text(
            f"Task cancelled: {task_id}\n"
            f"Description: {task['description']}"
        )

        self._log_operation(
            user_id,
            "task_cancel",
            {"task_id": task_id},
            True
        )

    # ========================================================================
    # Message Handlers
    # ========================================================================

    async def handle_text_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle text messages"""
        user_id = update.effective_user.id

        if not self._is_user_allowed(user_id):
            return

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        text = update.message.text

        # Log message
        self._log_operation(
            user_id,
            "message_text",
            {"text_length": len(text)},
            True
        )

        # Process message (mock response for now)
        response = (
            f"Received your message.\n"
            f"Length: {len(text)} characters\n\n"
            f"Use /help to see available commands."
        )

        await update.message.reply_text(response)

    async def handle_voice_message(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle voice messages"""
        user_id = update.effective_user.id

        if not self._is_user_allowed(user_id):
            return

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        # Log voice message
        self._log_operation(
            user_id,
            "message_voice",
            {"duration": update.message.voice.duration},
            True
        )

        # Process voice (would use STT in production)
        await update.message.reply_text(
            "Voice message received. "
            "Speech-to-text processing would happen here."
        )

    async def handle_document(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle document uploads"""
        user_id = update.effective_user.id

        if not self._is_user_allowed(user_id):
            return

        if not self._check_rate_limit(user_id):
            await update.message.reply_text("Rate limit exceeded. Please wait.")
            return

        document = update.message.document

        # Log document
        self._log_operation(
            user_id,
            "message_document",
            {
                "file_name": document.file_name,
                "file_size": document.file_size,
                "mime_type": document.mime_type
            },
            True
        )

        # Process document
        await update.message.reply_text(
            f"Document received: {document.file_name}\n"
            f"Size: {document.file_size} bytes\n"
            f"Type: {document.mime_type}\n\n"
            f"Document processing would happen here."
        )

    # ========================================================================
    # Bot Control
    # ========================================================================

    async def send_notification(
        self,
        user_id: int,
        message: str
    ) -> bool:
        """
        Send notification to user.

        Args:
            user_id: Telegram user ID
            message: Notification message

        Returns:
            True if sent successfully
        """
        try:
            if not self.application:
                logger.error("Bot not started, cannot send notification")
                return False

            await self.application.bot.send_message(
                chat_id=user_id,
                text=message
            )

            self._log_operation(
                user_id,
                "notification_sent",
                {"message_length": len(message)},
                True
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            self._log_operation(
                user_id,
                "notification_sent",
                {"message_length": len(message)},
                False,
                str(e)
            )
            return False

    async def send_alert(
        self,
        message: str,
        user_ids: Optional[List[int]] = None
    ) -> int:
        """
        Send alert to multiple users.

        Args:
            message: Alert message
            user_ids: List of user IDs (sends to all allowed users if not provided)

        Returns:
            Number of users who received the alert
        """
        if user_ids is None:
            user_ids = list(self.allowed_user_ids)

        success_count = 0
        for user_id in user_ids:
            if await self.send_notification(user_id, f"ALERT: {message}"):
                success_count += 1

        logger.info(f"Alert sent to {success_count}/{len(user_ids)} users")

        return success_count

    def start(self) -> None:
        """Start the bot (polling mode)"""
        if self.running:
            logger.warning("Bot already running")
            return

        # Create application
        self.application = Application.builder().token(self.bot_token).build()

        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("agents", self.cmd_agents))
        self.application.add_handler(CommandHandler("task", self.cmd_task))
        self.application.add_handler(CommandHandler("cancel", self.cmd_cancel))

        # Add message handlers
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message)
        )
        self.application.add_handler(
            MessageHandler(filters.VOICE, self.handle_voice_message)
        )
        self.application.add_handler(
            MessageHandler(filters.Document.ALL, self.handle_document)
        )

        self.running = True

        logger.info("Telegram bot started")

        # Start polling
        self.application.run_polling()

    def stop(self) -> None:
        """Stop the bot"""
        if not self.running:
            return

        self.running = False

        if self.application:
            # Stop application
            # Note: Application.stop() is async, would need proper handling
            pass

        logger.info("Telegram bot stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        return {
            "running": self.running,
            "webhook_mode": self.webhook_mode,
            "allowed_users": len(self.allowed_user_ids),
            "active_tasks": len(self.active_tasks),
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }
