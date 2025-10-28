"""
Finance Agent - Multi-Agent Framework
======================================

Executes financial operations and trading tasks.

Capabilities:
- Market analysis and data gathering
- Trade execution and monitoring
- Portfolio management
- Risk assessment
- Arbitrage detection

Features:
- Multi-chain support (Solana, Ethereum, etc.)
- Real-time price monitoring
- Risk-aware trading
- Guardian defense integration for safety
- Audit logging for compliance
"""

import logging
from typing import Dict, Any
from datetime import datetime

from core.agents.base_agent import BaseAgent, AgentCapability, TaskResult, AgentStatus

logger = logging.getLogger(__name__)


class FinanceAgent(BaseAgent):
    """
    Executes financial and trading operations autonomously.
    """

    def __init__(self, agent_id: str = "finance"):
        super().__init__(
            agent_id=agent_id,
            role="finance",
            capabilities=[
                AgentCapability.TRADING,
                AgentCapability.MARKET_ANALYSIS,
            ]
        )

        # Supported chains
        self.supported_chains = ["solana", "ethereum", "binance"]

        # Risk limits
        self.max_position_size_usd = 10000.0
        self.max_slippage_percent = 2.0

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if finance agent can handle task"""
        task_type = task.get("type", "")
        return task_type in [
            "execute_trade",
            "analyze_market",
            "check_portfolio",
            "assess_risk",
            "find_arbitrage",
            "monitor_position",
            "get_price",
            "calculate_pnl"
        ]

    def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute financial task"""
        task_id = task.get("task_id", str(datetime.now().timestamp()))
        start_time = datetime.now()

        self.status = AgentStatus.BUSY
        self.current_task = task_id
        self.total_tasks += 1

        try:
            # Enhanced safety validation for financial operations
            if not self._validate_safety(task):
                raise ValueError("Task failed safety validation")

            # Check trading resource quota
            if not self._track_resource_usage("api_calls", 1.0):
                raise RuntimeError("API call quota exceeded")

            task_type = task.get("type")
            params = task.get("params", {})

            if task_type == "execute_trade":
                result = self._execute_trade(params)
            elif task_type == "analyze_market":
                result = self._analyze_market(params)
            elif task_type == "check_portfolio":
                result = self._check_portfolio(params)
            elif task_type == "assess_risk":
                result = self._assess_risk(params)
            elif task_type == "find_arbitrage":
                result = self._find_arbitrage(params)
            elif task_type == "monitor_position":
                result = self._monitor_position(params)
            elif task_type == "get_price":
                result = self._get_price(params)
            elif task_type == "calculate_pnl":
                result = self._calculate_pnl(params)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.IDLE
            self.current_task = None
            self.successful_tasks += 1
            self.completed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=True,
                result=result,
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)
            return task_result

        except Exception as e:
            logger.error(f"Finance task failed: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()

            self.status = AgentStatus.IDLE
            self.current_task = None
            self.failed_tasks_count += 1
            self.failed_tasks.append(task_id)

            task_result = TaskResult(
                task_id=task_id,
                agent_id=self.agent_id,
                success=False,
                error=str(e),
                execution_time_sec=execution_time
            )

            self._log_task_execution(task_id, task_result)
            return task_result

    def _execute_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade (simulated for safety in demo)"""
        action = params.get("action", "buy")  # buy, sell
        symbol = params.get("symbol", "SOL")
        amount = float(params.get("amount", 0))
        chain = params.get("chain", "solana")
        max_slippage = params.get("max_slippage", self.max_slippage_percent)

        logger.info(f"Executing trade: {action} {amount} {symbol} on {chain}")

        # Validate chain support
        if chain not in self.supported_chains:
            raise ValueError(f"Unsupported chain: {chain}")

        # Validate position size
        estimated_value = amount * 100  # Simulated price
        if estimated_value > self.max_position_size_usd:
            raise ValueError(f"Position size {estimated_value} exceeds limit {self.max_position_size_usd}")

        # Simulated trade execution
        # In production, would integrate with actual DEX/CEX APIs
        trade_result = {
            "trade_id": f"trade_{datetime.now().timestamp()}",
            "action": action,
            "symbol": symbol,
            "amount": amount,
            "chain": chain,
            "executed_price": 100.0,  # Simulated
            "slippage_percent": 0.5,
            "fees_usd": 0.25,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Trade executed: {trade_result['trade_id']}")

        return trade_result

    def _analyze_market(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions"""
        symbol = params.get("symbol", "SOL")
        chain = params.get("chain", "solana")
        timeframe = params.get("timeframe", "1h")

        logger.info(f"Analyzing market: {symbol} on {chain} ({timeframe})")

        # Simulated market analysis
        # In production, would fetch real data from APIs
        analysis = {
            "symbol": symbol,
            "chain": chain,
            "timeframe": timeframe,
            "current_price": 100.0,
            "price_change_24h": 5.2,
            "volume_24h_usd": 1000000.0,
            "market_cap_usd": 10000000.0,
            "trend": "bullish",  # bullish, bearish, neutral
            "support_levels": [95.0, 90.0, 85.0],
            "resistance_levels": [105.0, 110.0, 115.0],
            "indicators": {
                "rsi": 65.0,  # 0-100
                "macd": "bullish_crossover",
                "moving_average_50": 98.0,
                "moving_average_200": 95.0,
            },
            "sentiment": "positive",
            "confidence": 0.75,
            "timestamp": datetime.now().isoformat()
        }

        return analysis

    def _check_portfolio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check portfolio status"""
        wallet_address = params.get("wallet_address", "")
        chain = params.get("chain", "solana")

        logger.info(f"Checking portfolio: {wallet_address[:8]}... on {chain}")

        # Simulated portfolio data
        portfolio = {
            "wallet_address": wallet_address,
            "chain": chain,
            "total_value_usd": 5000.0,
            "positions": [
                {
                    "symbol": "SOL",
                    "amount": 20.0,
                    "value_usd": 2000.0,
                    "cost_basis": 1800.0,
                    "pnl_usd": 200.0,
                    "pnl_percent": 11.1
                },
                {
                    "symbol": "USDC",
                    "amount": 3000.0,
                    "value_usd": 3000.0,
                    "cost_basis": 3000.0,
                    "pnl_usd": 0.0,
                    "pnl_percent": 0.0
                }
            ],
            "total_pnl_usd": 200.0,
            "total_pnl_percent": 4.0,
            "timestamp": datetime.now().isoformat()
        }

        return portfolio

    def _assess_risk(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for trade or position"""
        action = params.get("action", "buy")
        symbol = params.get("symbol", "SOL")
        amount = float(params.get("amount", 0))

        logger.info(f"Assessing risk: {action} {amount} {symbol}")

        estimated_value = amount * 100  # Simulated price

        # Risk assessment
        risk_assessment = {
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "estimated_value_usd": estimated_value,
            "risk_score": 45.0,  # 0-100 (higher = riskier)
            "risk_level": "medium",  # low, medium, high, extreme
            "factors": [
                {
                    "factor": "position_size",
                    "score": 30.0,
                    "description": "Position size within acceptable range"
                },
                {
                    "factor": "volatility",
                    "score": 50.0,
                    "description": "Moderate volatility in recent trading"
                },
                {
                    "factor": "liquidity",
                    "score": 20.0,
                    "description": "High liquidity available"
                }
            ],
            "recommendations": [
                "Consider setting stop-loss at 5% below entry",
                "Monitor position closely in first 24 hours",
                "Diversify if position exceeds 40% of portfolio"
            ],
            "max_recommended_size": self.max_position_size_usd,
            "approved": estimated_value <= self.max_position_size_usd,
            "timestamp": datetime.now().isoformat()
        }

        return risk_assessment

    def _find_arbitrage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Find arbitrage opportunities"""
        symbol = params.get("symbol", "SOL")
        min_profit_percent = params.get("min_profit_percent", 0.5)

        logger.info(f"Finding arbitrage opportunities for {symbol} (min profit: {min_profit_percent}%)")

        # Simulated arbitrage detection
        # In production, would scan multiple exchanges in real-time
        opportunities = [
            {
                "opportunity_id": f"arb_{datetime.now().timestamp()}",
                "symbol": symbol,
                "buy_exchange": "Jupiter",
                "sell_exchange": "Raydium",
                "buy_price": 100.0,
                "sell_price": 101.5,
                "profit_percent": 1.5,
                "estimated_profit_usd": 15.0,
                "volume_available": 1000.0,
                "execution_time_sec": 2.0,
                "confidence": 0.85,
                "risk_level": "low"
            }
        ]

        return {
            "symbol": symbol,
            "opportunities_found": len(opportunities),
            "opportunities": opportunities,
            "total_potential_profit_usd": sum(o["estimated_profit_usd"] for o in opportunities),
            "timestamp": datetime.now().isoformat()
        }

    def _monitor_position(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor existing position"""
        position_id = params.get("position_id", "")
        symbol = params.get("symbol", "SOL")

        logger.info(f"Monitoring position: {position_id} ({symbol})")

        # Simulated position monitoring
        position_status = {
            "position_id": position_id,
            "symbol": symbol,
            "status": "active",
            "entry_price": 95.0,
            "current_price": 100.0,
            "amount": 10.0,
            "current_value_usd": 1000.0,
            "pnl_usd": 50.0,
            "pnl_percent": 5.26,
            "duration_hours": 24,
            "alerts": [
                {
                    "type": "target_reached",
                    "message": "Position reached 5% profit target",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "recommendations": [
                "Consider taking partial profits",
                "Move stop-loss to break-even"
            ],
            "timestamp": datetime.now().isoformat()
        }

        return position_status

    def _get_price(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current price for symbol"""
        symbol = params.get("symbol", "SOL")
        chain = params.get("chain", "solana")

        logger.info(f"Getting price: {symbol} on {chain}")

        # Simulated price data
        price_data = {
            "symbol": symbol,
            "chain": chain,
            "price_usd": 100.0,
            "price_change_1h": 0.5,
            "price_change_24h": 5.2,
            "price_change_7d": 12.8,
            "volume_24h_usd": 1000000.0,
            "high_24h": 105.0,
            "low_24h": 95.0,
            "timestamp": datetime.now().isoformat()
        }

        return price_data

    def _calculate_pnl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate profit/loss for position"""
        entry_price = float(params.get("entry_price", 0))
        current_price = float(params.get("current_price", 0))
        amount = float(params.get("amount", 0))
        fees_usd = float(params.get("fees_usd", 0))

        logger.info(f"Calculating PnL: {amount} units @ {entry_price} -> {current_price}")

        entry_value = entry_price * amount
        current_value = current_price * amount
        gross_pnl = current_value - entry_value
        net_pnl = gross_pnl - fees_usd
        pnl_percent = (net_pnl / entry_value * 100) if entry_value > 0 else 0.0

        pnl_result = {
            "entry_price": entry_price,
            "current_price": current_price,
            "amount": amount,
            "entry_value_usd": entry_value,
            "current_value_usd": current_value,
            "gross_pnl_usd": gross_pnl,
            "fees_usd": fees_usd,
            "net_pnl_usd": net_pnl,
            "pnl_percent": pnl_percent,
            "direction": "profit" if net_pnl > 0 else "loss" if net_pnl < 0 else "breakeven",
            "timestamp": datetime.now().isoformat()
        }

        return pnl_result
