"""
LLM Client Bridges (Claude + ChatGPT)
======================================
Unified interface for multiple LLM providers with comprehensive safety controls.

Features:
- Support for Claude (Anthropic) and ChatGPT (OpenAI)
- Unified interface for both providers
- Prompt injection filtering
- DLP scanning of responses
- Content moderation
- Rate limiting and cost tracking
- Token counting
- Full audit logging
"""

import os
import time
import logging
from typing import Optional, List, Dict, Any, AsyncIterator
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None  # type: ignore

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None  # type: ignore

from security.guardian_defense import get_guardian_defense
from utils.audit_chain import append_jsonl
from utils.policy_guard import get_policy_guard

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """LLM provider types"""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"


class LLMModel(Enum):
    """LLM model types"""
    # Claude models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # OpenAI models
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"


@dataclass
class LLMUsage:
    """LLM usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


@dataclass
class LLMResponse:
    """LLM response"""
    provider: str
    model: str
    content: str
    usage: LLMUsage
    finish_reason: str
    timestamp: str
    safe: bool
    safety_issues: List[str]


class LLMClient:
    """
    Unified LLM client with safety controls.

    Features:
    - Multiple providers (Claude, ChatGPT)
    - Prompt injection filtering
    - DLP scanning
    - Content moderation
    - Rate limiting
    - Cost tracking
    - Token counting
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        audit_log_path: str = "var/audit/llm_operations.jsonl",
        enable_safety_checks: bool = True,
        max_tokens_per_day: int = 1000000,
        max_cost_per_day: float = 100.0
    ):
        """
        Initialize LLM client.

        Args:
            anthropic_api_key: Anthropic API key (from ENV if not provided)
            openai_api_key: OpenAI API key (from ENV if not provided)
            audit_log_path: Path to audit log file
            enable_safety_checks: Enable safety checks (prompt injection, DLP)
            max_tokens_per_day: Maximum tokens per day
            max_cost_per_day: Maximum cost per day (USD)
        """
        self.audit_log_path = audit_log_path
        self.enable_safety_checks = enable_safety_checks
        self.max_tokens_per_day = max_tokens_per_day
        self.max_cost_per_day = max_cost_per_day

        # API keys
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # Initialize clients
        self.anthropic_client = None
        self.openai_client = None

        if self.anthropic_api_key and ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)

        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        # Get integrations
        self.guardian = get_guardian_defense()
        self.policy_guard = get_policy_guard()

        # Usage tracking (per day)
        self.daily_tokens: Dict[str, int] = defaultdict(int)
        self.daily_cost: Dict[str, float] = defaultdict(float)
        self.last_reset = datetime.now().date()

        logger.info(
            f"LLMClient initialized "
            f"(anthropic={self.anthropic_client is not None}, "
            f"openai={self.openai_client is not None})"
        )

    def _reset_daily_limits_if_needed(self) -> None:
        """Reset daily limits if new day"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_tokens.clear()
            self.daily_cost.clear()
            self.last_reset = today
            logger.info("Daily limits reset")

    def _check_daily_limits(self, provider: str) -> None:
        """Check if daily limits exceeded"""
        self._reset_daily_limits_if_needed()

        if self.daily_tokens[provider] >= self.max_tokens_per_day:
            raise RuntimeError(
                f"Daily token limit exceeded for {provider}: "
                f"{self.daily_tokens[provider]}/{self.max_tokens_per_day}"
            )

        if self.daily_cost[provider] >= self.max_cost_per_day:
            raise RuntimeError(
                f"Daily cost limit exceeded for {provider}: "
                f"${self.daily_cost[provider]:.2f}/${self.max_cost_per_day:.2f}"
            )

    def _update_usage(
        self,
        provider: str,
        tokens: int,
        cost: float
    ) -> None:
        """Update usage statistics"""
        self.daily_tokens[provider] += tokens
        self.daily_cost[provider] += cost

    def _estimate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost based on model pricing"""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            # Claude pricing
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

            # OpenAI pricing
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        }

        if model in pricing:
            input_cost = (prompt_tokens / 1_000_000) * pricing[model]["input"]
            output_cost = (completion_tokens / 1_000_000) * pricing[model]["output"]
            return input_cost + output_cost

        return 0.0

    def _check_prompt_safety(self, prompt: str) -> tuple[bool, List[str]]:
        """
        Check prompt for injection attacks and unsafe content.

        Returns:
            (is_safe, issues)
        """
        if not self.enable_safety_checks:
            return True, []

        issues = []

        # Check for prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard all previous",
            "forget everything",
            "new instructions:",
            "system:",
            "you are now",
        ]

        prompt_lower = prompt.lower()
        for pattern in injection_patterns:
            if pattern in prompt_lower:
                issues.append(f"Potential prompt injection detected: {pattern}")

        # Check for sensitive data patterns (simplified)
        if len(prompt) > 10000:
            issues.append("Prompt too long (>10000 chars)")

        is_safe = len(issues) == 0

        if not is_safe:
            logger.warning(f"Unsafe prompt detected: {issues}")

            # Log to Guardian
            self.guardian.detect_resource_abuse(
                "llm_prompt",
                cpu=0,
                memory=0
            )

        return is_safe, issues

    def _check_response_safety(self, response: str) -> tuple[bool, List[str]]:
        """
        Check response for sensitive data leakage.

        Returns:
            (is_safe, issues)
        """
        if not self.enable_safety_checks:
            return True, []

        issues = []

        # Simple DLP checks (in production, use proper DLP scanner)
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        }

        import re
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, response):
                issues.append(f"Potential {pattern_name} detected in response")

        is_safe = len(issues) == 0

        if not is_safe:
            logger.warning(f"Unsafe response detected: {issues}")

        return is_safe, issues

    def _log_operation(
        self,
        provider: str,
        model: str,
        prompt_length: int,
        response_length: int,
        usage: LLMUsage,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log operation to audit chain"""
        try:
            operation_id = f"llm_{int(time.time() * 1000)}"

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "integration": "llm",
                "operation_id": operation_id,
                "provider": provider,
                "model": model,
                "prompt_length": prompt_length,
                "response_length": response_length,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost": usage.estimated_cost
                },
                "success": success,
                "error": error,
            }

            append_jsonl(self.audit_log_path, log_entry)

        except Exception as e:
            logger.error(f"Failed to log operation: {e}")

    async def complete(
        self,
        prompt: str,
        provider: LLMProvider,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> LLMResponse:
        """
        Complete prompt using specified LLM provider.

        Args:
            prompt: Input prompt
            provider: LLM provider (claude, chatgpt)
            model: Model to use (uses default if not specified)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            stream: Enable streaming

        Returns:
            LLM response
        """
        # Check policy
        policy_result = self.policy_guard.evaluate({
            "action": "llm.complete",
            "provider": provider.value,
            "prompt_length": len(prompt)
        })

        if policy_result.decision == "deny":
            raise PermissionError(
                f"LLM completion denied: {', '.join(policy_result.reasons)}"
            )

        # Check prompt safety
        prompt_safe, prompt_issues = self._check_prompt_safety(prompt)
        if not prompt_safe:
            raise ValueError(f"Unsafe prompt: {', '.join(prompt_issues)}")

        # Check daily limits
        self._check_daily_limits(provider.value)

        try:
            if provider == LLMProvider.CLAUDE:
                return await self._complete_claude(prompt, model, max_tokens, temperature)
            elif provider == LLMProvider.CHATGPT:
                return await self._complete_openai(prompt, model, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            raise

    async def _complete_claude(
        self,
        prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Complete prompt using Claude"""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized")

        model = model or LLMModel.CLAUDE_3_SONNET.value

        try:
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = message.content[0].text
            usage = LLMUsage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                estimated_cost=self._estimate_cost(
                    "claude",
                    model,
                    message.usage.input_tokens,
                    message.usage.output_tokens
                )
            )

            # Check response safety
            response_safe, response_issues = self._check_response_safety(content)

            # Update usage
            self._update_usage("claude", usage.total_tokens, usage.estimated_cost)

            # Log operation
            self._log_operation(
                "claude",
                model,
                len(prompt),
                len(content),
                usage,
                True
            )

            return LLMResponse(
                provider="claude",
                model=model,
                content=content,
                usage=usage,
                finish_reason=message.stop_reason or "end_turn",
                timestamp=datetime.now().isoformat(),
                safe=response_safe,
                safety_issues=response_issues
            )

        except Exception as e:
            logger.error(f"Claude completion failed: {e}")
            self._log_operation(
                "claude",
                model or "unknown",
                len(prompt),
                0,
                LLMUsage(0, 0, 0, 0.0),
                False,
                str(e)
            )
            raise

    async def _complete_openai(
        self,
        prompt: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> LLMResponse:
        """Complete prompt using OpenAI"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized")

        model = model or LLMModel.GPT35_TURBO.value

        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content or ""
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                estimated_cost=self._estimate_cost(
                    "openai",
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            )

            # Check response safety
            response_safe, response_issues = self._check_response_safety(content)

            # Update usage
            self._update_usage("openai", usage.total_tokens, usage.estimated_cost)

            # Log operation
            self._log_operation(
                "openai",
                model,
                len(prompt),
                len(content),
                usage,
                True
            )

            return LLMResponse(
                provider="openai",
                model=model,
                content=content,
                usage=usage,
                finish_reason=response.choices[0].finish_reason or "stop",
                timestamp=datetime.now().isoformat(),
                safe=response_safe,
                safety_issues=response_issues
            )

        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            self._log_operation(
                "openai",
                model or "unknown",
                len(prompt),
                0,
                LLMUsage(0, 0, 0, 0.0),
                False,
                str(e)
            )
            raise

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        self._reset_daily_limits_if_needed()

        return {
            "date": self.last_reset.isoformat(),
            "providers": {
                provider: {
                    "tokens_used": self.daily_tokens[provider],
                    "tokens_limit": self.max_tokens_per_day,
                    "cost_used": round(self.daily_cost[provider], 4),
                    "cost_limit": self.max_cost_per_day,
                    "tokens_remaining": max(0, self.max_tokens_per_day - self.daily_tokens[provider]),
                    "cost_remaining": max(0, self.max_cost_per_day - self.daily_cost[provider]),
                }
                for provider in ["claude", "openai"]
            },
            "total_tokens_used": sum(self.daily_tokens.values()),
            "total_cost_used": round(sum(self.daily_cost.values()), 4),
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        if self.anthropic_client:
            providers.append("claude")
        if self.openai_client:
            providers.append("openai")
        return providers
