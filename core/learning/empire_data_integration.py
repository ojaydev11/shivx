"""
Empire Data Integration - Automatic data collection from all empire operations

This integrates the DataCollector with empire routes to capture:
- Every empire management decision (Empire Manager)
- Every trading decision (Aayan AI)
- Every autonomous action (Sewago, Halobuzz, SolsniperPro)

All operations are tracked for AGI training.

Part of ShivX Personal Empire AGI (Week 2).
"""

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime

from core.learning.data_collector import (
    get_collector,
    TaskDomain,
    TaskType,
)

logger = logging.getLogger(__name__)


class EmpireDataIntegration:
    """
    Integration layer between empire operations and data collector.

    Automatically tracks all empire decisions for AGI training.
    """

    def __init__(self):
        self.collector = get_collector()
        logger.info("Empire Data Integration initialized")

    def track_empire_operation(
        self,
        operation_name: str,
        project_name: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to track empire operations.

        Usage:
            @track_empire_operation("deploy_project")
            async def deploy_project(project_id, ...):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Determine domain from project name or operation
                domain = self._infer_domain(project_name, operation_name, kwargs)

                # Determine task type
                task_type = self._infer_task_type(operation_name)

                # Build context
                context = {
                    "operation": operation_name,
                    "project_name": project_name,
                    "request_data": request_data or {},
                    "args": str(args)[:200],  # Truncate to avoid huge contexts
                    "kwargs_keys": list(kwargs.keys()),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Extract query from args/kwargs
                query = self._extract_query(operation_name, args, kwargs)

                # Start tracking
                task_id = self.collector.start_task(
                    domain=domain,
                    task_type=task_type,
                    context=context,
                    query=query,
                )

                try:
                    # Execute operation
                    start_time = datetime.utcnow()
                    result = await func(*args, **kwargs)
                    duration = (datetime.utcnow() - start_time).total_seconds()

                    # Determine success
                    success = self._is_successful(result)

                    # Record action
                    self.collector.record_action(
                        task_id,
                        action_taken=f"{operation_name} executed",
                        reasoning=self._extract_reasoning(operation_name, kwargs),
                        confidence=0.8,  # Default confidence
                    )

                    # Complete task
                    self.collector.complete_task(
                        task_id,
                        outcome=self._extract_outcome(result),
                        success=success,
                        user_feedback=None,  # Will be added later if user provides feedback
                    )

                    logger.debug(
                        f"Tracked empire operation: {operation_name} "
                        f"(domain={domain.value}, success={success}, duration={duration:.2f}s)"
                    )

                    return result

                except Exception as e:
                    # Record failure
                    self.collector.complete_task(
                        task_id,
                        outcome=f"Operation failed: {str(e)[:200]}",
                        success=False,
                    )

                    logger.error(f"Empire operation failed: {operation_name} - {str(e)}")
                    raise

            return wrapper
        return decorator

    def track_aayan_decision(
        self,
        decision_type: str,
        symbol: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to track Aayan AI trading decisions.

        Usage:
            @track_aayan_decision("arbitrage_scan", symbol="SOL/USD")
            async def scan_arbitrage(...):
                ...
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Build context
                context = {
                    "decision_type": decision_type,
                    "symbol": symbol,
                    "args": str(args)[:200],
                    "kwargs_keys": list(kwargs.keys()),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Extract query
                query = f"Aayan AI: {decision_type}" + (f" for {symbol}" if symbol else "")

                # Start tracking
                task_id = self.collector.start_task(
                    domain=TaskDomain.SOLSNIPER,
                    task_type=TaskType.TRADING_DECISION,
                    context=context,
                    query=query,
                )

                try:
                    # Execute decision
                    start_time = datetime.utcnow()
                    result = await func(*args, **kwargs)
                    duration = (datetime.utcnow() - start_time).total_seconds()

                    # Determine success
                    success = self._is_successful(result)

                    # Record decision
                    self.collector.record_action(
                        task_id,
                        action_taken=self._extract_trading_action(decision_type, result),
                        reasoning=self._extract_trading_reasoning(decision_type, result),
                        confidence=self._extract_confidence(result),
                    )

                    # Complete task
                    self.collector.complete_task(
                        task_id,
                        outcome=self._extract_outcome(result),
                        success=success,
                    )

                    logger.debug(
                        f"Tracked Aayan decision: {decision_type} "
                        f"(success={success}, duration={duration:.2f}s)"
                    )

                    return result

                except Exception as e:
                    # Record failure
                    self.collector.complete_task(
                        task_id,
                        outcome=f"Decision failed: {str(e)[:200]}",
                        success=False,
                    )

                    logger.error(f"Aayan decision failed: {decision_type} - {str(e)}")
                    raise

            return wrapper
        return decorator

    def _infer_domain(
        self,
        project_name: Optional[str],
        operation_name: str,
        kwargs: Dict[str, Any],
    ) -> TaskDomain:
        """Infer which empire domain this operation belongs to"""
        # Check project name
        if project_name:
            project_lower = project_name.lower()
            if "sewago" in project_lower or "sewa" in project_lower:
                return TaskDomain.SEWAGO
            elif "halobuzz" in project_lower or "halo" in project_lower:
                return TaskDomain.HALOBUZZ
            elif "solsniper" in project_lower or "aayan" in project_lower:
                return TaskDomain.SOLSNIPER
            elif "nepvest" in project_lower:
                return TaskDomain.NEPVEST

        # Check operation name
        op_lower = operation_name.lower()
        if "trading" in op_lower or "arbitrage" in op_lower or "wallet" in op_lower:
            return TaskDomain.SOLSNIPER
        elif "content" in op_lower or "social" in op_lower:
            return TaskDomain.HALOBUZZ

        # Check kwargs for domain hints
        if "project_id" in kwargs:
            # Could check project registry here
            pass

        # Default to ShivX core
        return TaskDomain.SHIVX_CORE

    def _infer_task_type(self, operation_name: str) -> TaskType:
        """Infer task type from operation name"""
        op_lower = operation_name.lower()

        if "fix" in op_lower or "error" in op_lower:
            return TaskType.BUG_FIXING
        elif "deploy" in op_lower:
            return TaskType.DECISION_MAKING
        elif "refactor" in op_lower or "optimize" in op_lower:
            return TaskType.CODE_GENERATION
        elif "feature" in op_lower:
            return TaskType.CODE_GENERATION
        elif "trade" in op_lower or "arbitrage" in op_lower:
            return TaskType.TRADING_DECISION
        elif "content" in op_lower:
            return TaskType.CONTENT_CREATION
        elif "scale" in op_lower or "upgrade" in op_lower:
            return TaskType.SYSTEM_OPTIMIZATION
        else:
            return TaskType.DECISION_MAKING

    def _extract_query(
        self,
        operation_name: str,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> str:
        """Extract meaningful query from operation"""
        # Try to get description from kwargs
        if "description" in kwargs:
            return f"{operation_name}: {kwargs['description']}"

        # Try to get project_id
        if "project_id" in kwargs:
            return f"{operation_name} for project {kwargs['project_id']}"

        # Default
        return f"Execute {operation_name}"

    def _extract_reasoning(
        self,
        operation_name: str,
        kwargs: Dict[str, Any],
    ) -> str:
        """Extract reasoning for the operation"""
        # Check for explicit goal or reason
        if "goal" in kwargs:
            return f"Goal: {kwargs['goal']}"

        if "reason" in kwargs:
            return kwargs["reason"]

        # Default reasoning based on operation type
        if "deploy" in operation_name:
            return "Deploying to production for users"
        elif "fix" in operation_name:
            return "Fixing errors to improve system reliability"
        elif "upgrade" in operation_name:
            return "Upgrading to improve security and performance"
        elif "scale" in operation_name:
            return "Scaling to handle increased load"
        else:
            return f"Executing {operation_name} as requested"

    def _is_successful(self, result: Any) -> bool:
        """Determine if operation was successful"""
        if isinstance(result, dict):
            # Check for success field
            if "success" in result:
                return bool(result["success"])

            # Check for error field
            if "error" in result:
                return False

            # Default to True if no error
            return True

        # Non-dict results are considered successful if no exception
        return True

    def _extract_outcome(self, result: Any) -> str:
        """Extract outcome description from result"""
        if isinstance(result, dict):
            # Try to get message
            if "message" in result:
                return result["message"]

            # Try to get deployment URL
            if "deployment_url" in result:
                return f"Deployed to {result['deployment_url']}"

            # Generic outcome based on fields
            return f"Operation completed with {len(result)} result fields"

        return "Operation completed successfully"

    def _extract_trading_action(self, decision_type: str, result: Any) -> str:
        """Extract trading action from result"""
        if isinstance(result, dict):
            # Check for opportunities found
            if "opportunities_found" in result:
                count = result["opportunities_found"]
                return f"Found {count} trading opportunities"

            # Check for trades executed
            if "trades_executed" in result:
                return f"Executed {result['trades_executed']} trades"

            # Check for scan results
            if "scanned_symbols" in result:
                return f"Scanned {result['scanned_symbols']} symbols"

        return f"Executed {decision_type}"

    def _extract_trading_reasoning(self, decision_type: str, result: Any) -> str:
        """Extract trading reasoning"""
        if isinstance(result, dict):
            # Check for top opportunity
            if "opportunities" in result and len(result["opportunities"]) > 0:
                top_opp = result["opportunities"][0]
                if isinstance(top_opp, dict) and "profit_percent" in top_opp:
                    return f"Best opportunity: {top_opp['profit_percent']}% profit potential"

            # Check for analysis
            if "analysis" in result:
                return "Detailed market analysis performed"

        return f"Applied {decision_type} strategy based on market conditions"

    def _extract_confidence(self, result: Any) -> float:
        """Extract confidence from result"""
        if isinstance(result, dict):
            if "confidence" in result:
                return float(result["confidence"])

            # Infer confidence from success indicators
            if "success" in result and result["success"]:
                return 0.8

        return 0.7  # Default moderate confidence


# Global singleton
_integration = None


def get_empire_integration() -> EmpireDataIntegration:
    """Get or create global empire data integration"""
    global _integration

    if _integration is None:
        _integration = EmpireDataIntegration()

    return _integration


# Convenience decorators for quick usage
def track_empire(operation_name: str, project_name: Optional[str] = None):
    """
    Quick decorator for empire operations.

    Usage:
        @track_empire("deploy_project", project_name="sewago")
        async def deploy_project(...):
            ...
    """
    integration = get_empire_integration()
    return integration.track_empire_operation(operation_name, project_name)


def track_trading(decision_type: str, symbol: Optional[str] = None):
    """
    Quick decorator for trading decisions.

    Usage:
        @track_trading("arbitrage_scan", symbol="SOL/USD")
        async def scan_arbitrage(...):
            ...
    """
    integration = get_empire_integration()
    return integration.track_aayan_decision(decision_type, symbol)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test integration
    import asyncio

    async def test_integration():
        """Test data integration"""
        print("\n=== Empire Data Integration Test ===\n")

        integration = EmpireDataIntegration()

        # Test empire operation
        @integration.track_empire_operation("deploy_project", project_name="sewago")
        async def test_deploy():
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "success": True,
                "message": "Deployment completed",
                "deployment_url": "https://sewago.com",
            }

        result1 = await test_deploy()
        print(f"Deploy result: {result1}")

        # Test trading decision
        @integration.track_aayan_decision("arbitrage_scan", symbol="SOL/USD")
        async def test_scan():
            await asyncio.sleep(0.1)  # Simulate work
            return {
                "success": True,
                "opportunities_found": 3,
                "opportunities": [
                    {"profit_percent": 2.5, "symbol": "SOL/USD"},
                ],
            }

        result2 = await test_scan()
        print(f"Scan result: {result2}")

        # Show stats
        stats = integration.collector.get_stats()
        print(f"\nCollected {stats['dataset']['total']} examples")
        print(f"Success rate: {stats['dataset']['success_rate']:.1%}")

    asyncio.run(test_integration())
