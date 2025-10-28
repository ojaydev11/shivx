"""
Spatial planner for pathfinding and action planning.
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


class SpatialPlanner:
    """
    Planner for spatial reasoning tasks using A* search.
    """

    def __init__(self, algorithm: str = "astar"):
        """
        Initialize planner.

        Args:
            algorithm: Planning algorithm (astar, rrt, neural)
        """
        self.algorithm = algorithm
        logger.info(f"Spatial planner initialized: {algorithm}")

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: np.ndarray,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Plan path from start to goal.

        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            grid: Occupancy grid (0=free, 1=occupied)

        Returns:
            Path as list of (x, y) positions, or None if no path
        """
        if self.algorithm == "astar":
            return self._astar(start, goal, grid)
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm} not implemented")

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: np.ndarray,
    ) -> Optional[List[Tuple[int, int]]]:
        """A* pathfinding."""
        from heapq import heappop, heappush

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_set = []
        heappush(open_set, (heuristic(start), 0, start, [start]))

        visited = set()

        while open_set:
            _, cost, current, path = heappop(open_set)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Explore neighbors
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy

                # Check bounds
                if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
                    continue

                # Check if free
                if grid[ny, nx] != 0 and (nx, ny) != goal:
                    continue

                neighbor = (nx, ny)
                if neighbor in visited:
                    continue

                new_cost = cost + 1
                priority = new_cost + heuristic(neighbor)
                heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))

        return None  # No path found

    def actions_from_path(
        self, path: List[Tuple[int, int]]
    ) -> List[str]:
        """
        Convert path to action sequence.

        Args:
            path: List of (x, y) positions

        Returns:
            List of actions (up, down, left, right)
        """
        actions = []
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]

            dx = next_pos[0] - curr[0]
            dy = next_pos[1] - curr[1]

            if dx == 1:
                actions.append("right")
            elif dx == -1:
                actions.append("left")
            elif dy == 1:
                actions.append("down")
            elif dy == -1:
                actions.append("up")

        return actions
