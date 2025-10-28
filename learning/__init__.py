"""
Continuous Lifelong Learning (CLL) system.

Enables the AGI to learn from experience without catastrophic forgetting.
"""

from .online_adapter.adapter import OnlineAdapter
from .experience_buffer.buffer import ExperienceBuffer
from .scheduler.learning_scheduler import LearningScheduler

__all__ = ["OnlineAdapter", "ExperienceBuffer", "LearningScheduler"]

__version__ = "1.0.0"
