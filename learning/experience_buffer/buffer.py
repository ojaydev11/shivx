"""
Experience buffer for storing and sampling training examples.

Implements importance sampling to prevent catastrophic forgetting.
"""

import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class Experience:
    """A single learning experience."""

    def __init__(
        self,
        input_data: Any,
        target: Any,
        loss: float = 0.0,
        importance: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        self.input_data = input_data
        self.target = target
        self.loss = loss
        self.importance = importance
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
        self.sample_count = 0


class ExperienceBuffer:
    """
    Buffer for storing and sampling training experiences.

    Uses importance sampling to prioritize:
    - High-loss examples (harder to learn)
    - Recent examples
    - User-corrected examples
    """

    def __init__(
        self,
        max_size: int = 10000,
        sampling_strategy: str = "importance",
        priority_alpha: float = 0.6,
    ):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum buffer size
            sampling_strategy: importance, uniform, or recency
            priority_alpha: Importance sampling exponent
        """
        self.max_size = max_size
        self.sampling_strategy = sampling_strategy
        self.priority_alpha = priority_alpha

        self.buffer: deque = deque(maxlen=max_size)
        self.total_samples = 0

        logger.info(
            f"Experience buffer initialized: "
            f"max_size={max_size}, strategy={sampling_strategy}"
        )

    def add(
        self,
        input_data: Any,
        target: Any,
        loss: float = 0.0,
        importance: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            input_data: Input data
            target: Target/label
            loss: Training loss (if available)
            importance: Manual importance score
            metadata: Additional metadata
        """
        experience = Experience(
            input_data=input_data,
            target=target,
            loss=loss,
            importance=importance,
            metadata=metadata,
        )

        self.buffer.append(experience)
        self.total_samples += 1

        logger.debug(
            f"Added experience to buffer "
            f"(size={len(self.buffer)}/{self.max_size})"
        )

    def sample(self, batch_size: int = 16) -> List[Experience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of samples

        Returns:
            List of sampled experiences
        """
        if len(self.buffer) == 0:
            return []

        batch_size = min(batch_size, len(self.buffer))

        if self.sampling_strategy == "uniform":
            samples = random.sample(list(self.buffer), batch_size)

        elif self.sampling_strategy == "recency":
            # Sample more recent experiences
            samples = list(self.buffer)[-batch_size:]

        elif self.sampling_strategy == "importance":
            # Importance sampling
            weights = [
                (exp.importance * (1 + exp.loss)) ** self.priority_alpha
                for exp in self.buffer
            ]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            samples = random.choices(
                list(self.buffer), weights=probabilities, k=batch_size
            )

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        # Update sample counts
        for exp in samples:
            exp.sample_count += 1

        logger.debug(f"Sampled {len(samples)} experiences")
        return samples

    def update_importance(self, index: int, new_importance: float) -> None:
        """Update importance of an experience."""
        if 0 <= index < len(self.buffer):
            self.buffer[index].importance = new_importance

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "max_size": self.max_size,
                "total_samples": self.total_samples,
            }

        avg_importance = sum(exp.importance for exp in self.buffer) / len(self.buffer)
        avg_loss = sum(exp.loss for exp in self.buffer) / len(self.buffer)

        return {
            "size": len(self.buffer),
            "max_size": self.max_size,
            "total_samples": self.total_samples,
            "avg_importance": avg_importance,
            "avg_loss": avg_loss,
            "strategy": self.sampling_strategy,
        }

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        logger.info("Experience buffer cleared")

    def __len__(self) -> int:
        return len(self.buffer)
