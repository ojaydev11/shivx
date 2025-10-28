"""
Minimal gridworld simulator for spatial reasoning.
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


class MiniWorldSimulator:
    """
    Simple gridworld environment for spatial reasoning tasks.

    Grid-based world where agent can move and manipulate objects.
    """

    def __init__(self, width: int = 8, height: int = 8):
        """
        Initialize simulator.

        Args:
            width: Grid width
            height: Grid height
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.agent_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.steps = 0

        logger.info(f"MiniWorld simulator initialized: {width}x{height}")

    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.width - 1, self.height - 1)
        self.steps = 0

        # Mark agent and goal
        self.grid[self.agent_pos] = 1
        self.grid[self.goal_pos] = 9

        return self.grid.copy()

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Action to take (up, down, left, right)

        Returns:
            observation, reward, done, info
        """
        # Clear agent from old position
        self.grid[self.agent_pos] = 0

        # Compute new position
        x, y = self.agent_pos
        if action == "up" and y > 0:
            y -= 1
        elif action == "down" and y < self.height - 1:
            y += 1
        elif action == "left" and x > 0:
            x -= 1
        elif action == "right" and x < self.width - 1:
            x += 1

        self.agent_pos = (x, y)
        self.steps += 1

        # Place agent in new position
        self.grid[self.agent_pos] = 1

        # Check if reached goal
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01

        info = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "steps": self.steps,
        }

        return self.grid.copy(), reward, done, info

    def render(self) -> str:
        """Render grid as string."""
        symbols = {0: ".", 1: "A", 9: "G"}
        lines = []
        for row in self.grid:
            line = "".join(symbols.get(cell, "?") for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self.grid.copy()
