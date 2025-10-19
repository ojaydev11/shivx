"""
Parallel Reasoning Engine
=========================
Process multiple reasoning paths simultaneously
"""

import asyncio
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ParallelReasoning:
    """
    Execute multiple reasoning tasks in parallel

    Example:
        Query: "Build a trading bot"
        Parallel tasks:
        - Research APIs (Agent 1)
        - Design architecture (Agent 2)
        - Security analysis (Agent 3)
        → Combine results
    """

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"[PARALLEL] Initialized with {max_workers} workers")

    async def reason_in_parallel(
        self,
        tasks: List[Dict[str, Any]],
        synthesizer=None
    ) -> Dict[str, Any]:
        """
        Execute multiple reasoning tasks in parallel

        Args:
            tasks: List of {"task": "...", "handler": async_function}
            synthesizer: Optional function to combine results

        Returns:
            Combined results
        """
        logger.info(f"[PARALLEL] Starting {len(tasks)} tasks in parallel")

        # Execute all tasks concurrently
        results = await asyncio.gather(*[
            self._execute_task(task) for task in tasks
        ], return_exceptions=True)

        # Filter out errors
        successful_results = [
            r for r in results if not isinstance(r, Exception)
        ]

        # Combine results
        if synthesizer:
            combined = await synthesizer(successful_results)
        else:
            combined = {
                "results": successful_results,
                "total_tasks": len(tasks),
                "successful": len(successful_results),
                "failed": len(tasks) - len(successful_results)
            }

        logger.info(f"[PARALLEL] Completed {len(successful_results)}/{len(tasks)} tasks")

        return combined

    async def _execute_task(self, task: Dict) -> Any:
        """Execute a single task"""
        task_name = task.get("task", "unknown")
        handler = task.get("handler")

        try:
            logger.debug(f"[PARALLEL] Executing: {task_name}")
            result = await handler()
            return {"task": task_name, "result": result, "success": True}
        except Exception as e:
            logger.error(f"[PARALLEL] Task '{task_name}' failed: {e}")
            return {"task": task_name, "error": str(e), "success": False}


# Singleton
_parallel_engine = None

def get_parallel_engine():
    """Get the global Parallel Reasoning Engine instance"""
    global _parallel_engine
    if _parallel_engine is None:
        _parallel_engine = ParallelReasoning()
    return _parallel_engine
