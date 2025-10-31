"""
Trading Service Layer
Connects API routers to core trading AI implementations
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from core.income.advanced_trading_ai import AdvancedTradingAI, MarketState, TradingSignal as CoreSignal
from core.income.jupiter_client import JupiterClient
from app.models.trading import Position, TradeSignal, TradeExecution, Strategy, PositionStatus, TradeAction
from config.settings import Settings

logger = logging.getLogger(__name__)


class TradingService:
    """
    Service for trading operations
    Integrates Advanced Trading AI with database and Jupiter DEX
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.jupiter_client: Optional[JupiterClient] = None
        self.trading_ai: Optional[AdvancedTradingAI] = None

        # Initialize Trading AI
        config = {
            'min_profit_pct': settings.min_profit_threshold,
            'max_position_size_usd': settings.max_position_size
        }
        self.trading_ai = AdvancedTradingAI(config)

        logger.info("Trading service initialized")

    async def get_strategies(self, db: AsyncSession) -> List[Strategy]:
        """Get all trading strategies from database"""
        result = await db.execute(select(Strategy))
        strategies = result.scalars().all()

        # If no strategies in database, create defaults
        if not strategies:
            strategies = await self._create_default_strategies(db)

        return list(strategies)

    async def _create_default_strategies(self, db: AsyncSession) -> List[Strategy]:
        """Create default strategies"""
        default_strategies = [
            Strategy(
                name="RL Trading (PPO)",
                enabled=self.settings.feature_rl_trading,
                max_position_size=self.settings.max_position_size,
                stop_loss_pct=self.settings.stop_loss_pct,
                take_profit_pct=self.settings.take_profit_pct,
                risk_tolerance="medium"
            ),
            Strategy(
                name="Sentiment Analysis",
                enabled=self.settings.feature_sentiment_analysis,
                max_position_size=self.settings.max_position_size * 0.5,
                stop_loss_pct=0.03,
                take_profit_pct=0.07,
                risk_tolerance="low"
            ),
            Strategy(
                name="DEX Arbitrage",
                enabled=self.settings.feature_dex_arbitrage,
                max_position_size=self.settings.max_position_size * 0.3,
                stop_loss_pct=0.01,
                take_profit_pct=0.02,
                risk_tolerance="low"
            )
        ]

        for strategy in default_strategies:
            db.add(strategy)

        await db.commit()

        for strategy in default_strategies:
            await db.refresh(strategy)

        return default_strategies

    async def get_positions(
        self,
        db: AsyncSession,
        status: Optional[PositionStatus] = None,
        user_id: Optional[str] = None
    ) -> List[Position]:
        """Get trading positions"""
        query = select(Position)

        filters = []
        if status:
            filters.append(Position.status == status)
        if user_id:
            filters.append(Position.user_id == user_id)

        if filters:
            query = query.where(and_(*filters))

        result = await db.execute(query)
        return list(result.scalars().all())

    async def generate_signals(
        self,
        db: AsyncSession,
        token: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[TradeSignal]:
        """Generate AI trading signals"""
        if not self.settings.feature_advanced_trading:
            logger.warning("Advanced trading disabled")
            return []

        # Get market data (would integrate with real price feeds)
        prices = await self._get_market_prices()
        volumes = await self._get_market_volumes()

        # Analyze market with AI
        market_state = await self.trading_ai.analyze_market(prices, volumes)

        # Generate signals
        pairs = self._get_trading_pairs()
        core_signals = await self.trading_ai.generate_signals(market_state, pairs)

        # Convert to database models
        db_signals = []
        for core_signal in core_signals:
            # Check if signal already exists
            signal_id = f"sig_{datetime.utcnow().timestamp()}_{core_signal.pair}"

            db_signal = TradeSignal(
                signal_id=signal_id,
                token=core_signal.pair.split('/')[0],
                action=TradeAction(core_signal.action.lower()),
                confidence=core_signal.confidence,
                price_target=core_signal.expected_profit_pct,  # Simplified
                strategy=core_signal.strategy,
                reasoning=', '.join(core_signal.reasoning[:3]),  # Limit length
                generated_at=core_signal.timestamp,
                expected_profit_pct=core_signal.expected_profit_pct,
                expected_risk_pct=core_signal.expected_risk_pct,
                sharpe_ratio=core_signal.sharpe_ratio,
                position_size_pct=core_signal.position_size_pct
            )

            db.add(db_signal)
            db_signals.append(db_signal)

        await db.commit()

        # Refresh all signals
        for signal in db_signals:
            await db.refresh(signal)

        # Apply filters
        if token:
            db_signals = [s for s in db_signals if s.token == token]
        if strategy:
            db_signals = [s for s in db_signals if s.strategy == strategy]

        return db_signals

    async def execute_trade(
        self,
        db: AsyncSession,
        token: str,
        action: str,
        amount: float,
        slippage_bps: int,
        user_id: Optional[str] = None
    ) -> TradeExecution:
        """Execute a trade"""
        # Check trading mode
        if self.settings.trading_mode.value == "paper":
            return await self._execute_paper_trade(db, token, action, amount, slippage_bps, user_id)
        else:
            return await self._execute_live_trade(db, token, action, amount, slippage_bps, user_id)

    async def _execute_paper_trade(
        self,
        db: AsyncSession,
        token: str,
        action: str,
        amount: float,
        slippage_bps: int,
        user_id: Optional[str] = None
    ) -> TradeExecution:
        """Execute paper trade (simulation)"""
        trade_id = f"paper_trade_{datetime.utcnow().timestamp()}"

        execution = TradeExecution(
            trade_id=trade_id,
            token=token,
            action=TradeAction(action.lower()),
            amount_in=amount,
            amount_out=amount * 0.99,  # Simulate 1% slippage
            price=100.0,  # Mock price
            slippage_actual=0.01,
            executed_at=datetime.utcnow(),
            transaction_signature=None,
            status="success"
        )

        db.add(execution)
        await db.commit()
        await db.refresh(execution)

        return execution

    async def _execute_live_trade(
        self,
        db: AsyncSession,
        token: str,
        action: str,
        amount: float,
        slippage_bps: int,
        user_id: Optional[str] = None
    ) -> TradeExecution:
        """Execute live trade via Jupiter DEX"""
        logger.info(f"Executing live trade: {action} {amount} {token}")

        try:
            # Initialize Jupiter client if needed
            if not self.jupiter_client:
                self.jupiter_client = JupiterClient()
                await self.jupiter_client.__aenter__()

            # Get token mints
            input_mint = JupiterClient.TOKENS.get(token)
            output_mint = JupiterClient.TOKENS.get("USDC")  # Default to USDC

            if not input_mint or not output_mint:
                raise ValueError(f"Unknown token: {token}")

            # Get quote from Jupiter
            quote = await self.jupiter_client.get_quote(
                input_mint=input_mint if action.lower() == "sell" else output_mint,
                output_mint=output_mint if action.lower() == "sell" else input_mint,
                amount=int(amount * 1_000_000),  # Convert to base units
                slippage_bps=slippage_bps
            )

            if not quote:
                raise Exception("Failed to get quote from Jupiter")

            # For now, just create execution record
            # Real implementation would get swap transaction and sign it
            trade_id = f"live_trade_{datetime.utcnow().timestamp()}"

            execution = TradeExecution(
                trade_id=trade_id,
                token=token,
                action=TradeAction(action.lower()),
                amount_in=amount,
                amount_out=float(quote.out_amount) / 1_000_000,
                price=float(quote.out_amount) / float(quote.in_amount),
                slippage_actual=quote.price_impact_pct / 100,
                executed_at=datetime.utcnow(),
                transaction_signature=None,  # Would be set after signing
                status="pending"  # Would be "success" after confirmation
            )

            db.add(execution)
            await db.commit()
            await db.refresh(execution)

            return execution

        except Exception as e:
            logger.error(f"Live trade execution failed: {e}")

            # Create failed execution record
            trade_id = f"failed_trade_{datetime.utcnow().timestamp()}"
            execution = TradeExecution(
                trade_id=trade_id,
                token=token,
                action=TradeAction(action.lower()),
                amount_in=amount,
                amount_out=0.0,
                price=0.0,
                slippage_actual=0.0,
                executed_at=datetime.utcnow(),
                transaction_signature=None,
                status="failed",
                error_message=str(e)
            )

            db.add(execution)
            await db.commit()
            await db.refresh(execution)

            raise

    async def get_performance(
        self,
        db: AsyncSession,
        period: str = "24h"
    ) -> Dict[str, Any]:
        """Get trading performance metrics"""
        # Would query from database in real implementation
        metrics = self.trading_ai.get_performance_metrics()

        return {
            "period": period,
            "total_trades": self.trading_ai.total_trades,
            "winning_trades": self.trading_ai.winning_trades,
            "losing_trades": self.trading_ai.losing_trades,
            "win_rate": metrics.win_rate,
            "total_pnl": self.trading_ai.total_profit,
            "total_pnl_pct": self.trading_ai.total_profit / 100,  # Simplified
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "average_trade_duration_minutes": 45,  # Would calculate from DB
            "best_strategy": "RL Trading (PPO)",  # Would calculate from DB
            "updated_at": datetime.utcnow()
        }

    async def update_strategy_status(
        self,
        db: AsyncSession,
        strategy_name: str,
        enabled: bool
    ) -> Strategy:
        """Enable/disable a trading strategy"""
        result = await db.execute(
            select(Strategy).where(Strategy.name == strategy_name)
        )
        strategy = result.scalar_one_or_none()

        if not strategy:
            # Create it
            strategy = Strategy(
                name=strategy_name,
                enabled=enabled,
                max_position_size=self.settings.max_position_size,
                stop_loss_pct=self.settings.stop_loss_pct,
                take_profit_pct=self.settings.take_profit_pct
            )
            db.add(strategy)
        else:
            strategy.enabled = enabled

        await db.commit()
        await db.refresh(strategy)

        return strategy

    async def _get_market_prices(self) -> Dict[str, float]:
        """Get current market prices (placeholder)"""
        # Would integrate with real price feed (Jupiter, Pyth, etc.)
        return {
            'SOL': 191.0,
            'USDC': 1.0,
            'USDT': 1.0,
            'RAY': 5.2,
            'ORCA': 3.8
        }

    async def _get_market_volumes(self) -> Dict[str, float]:
        """Get market volumes (placeholder)"""
        # Would integrate with real volume data
        return {
            'SOL': 1000000,
            'USDC': 5000000,
            'USDT': 4500000,
            'RAY': 500000,
            'ORCA': 300000
        }

    def _get_trading_pairs(self) -> List[tuple]:
        """Get trading pairs"""
        return [
            ('SOL', 'USDC'),
            ('SOL', 'USDT'),
            ('RAY', 'USDC'),
            ('ORCA', 'USDC')
        ]

    async def cleanup(self):
        """Cleanup resources"""
        if self.jupiter_client:
            await self.jupiter_client.__aexit__(None, None, None)
