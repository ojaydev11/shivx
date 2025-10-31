"""
Trading API Router
Endpoints for trading operations, strategies, and positions
"""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, require_permission, get_settings
from app.dependencies.auth import TokenData
from app.database import get_db
from app.services.trading_service import TradingService
from app.models.trading import PositionStatus, TradeAction
from config.settings import Settings
from core.security.hardening import Permission


router = APIRouter(
    prefix="/api/trading",
    tags=["trading"],
    responses={404: {"description": "Not found"}},
)


# ============================================================================
# Pydantic Models
# ============================================================================

class StrategyConfig(BaseModel):
    """Trading strategy configuration"""
    name: str = Field(..., description="Strategy name")
    enabled: bool = Field(default=True, description="Enable/disable strategy")
    max_position_size: float = Field(..., ge=0, description="Max position size (USD)")
    stop_loss_pct: float = Field(..., ge=0, le=1, description="Stop loss percentage")
    take_profit_pct: float = Field(..., ge=0, le=1, description="Take profit percentage")
    risk_tolerance: str = Field(default="medium", description="Risk tolerance (low/medium/high)")


class Position(BaseModel):
    """Trading position"""
    position_id: str
    token: str
    size: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    opened_at: datetime
    status: str  # open, closed, liquidated


class TradeSignal(BaseModel):
    """Trading signal from AI"""
    signal_id: str
    token: str
    action: str  # buy, sell, hold
    confidence: float = Field(..., ge=0, le=1)
    price_target: Optional[float] = None
    strategy: str
    reasoning: str
    generated_at: datetime


class TradeExecution(BaseModel):
    """Trade execution request"""
    token: str = Field(..., description="Token symbol (e.g., SOL)")
    action: str = Field(..., description="buy or sell")
    amount: float = Field(..., gt=0, description="Amount to trade (USD)")
    slippage_bps: int = Field(default=50, ge=1, le=1000, description="Slippage tolerance (basis points)")


class TradeResult(BaseModel):
    """Trade execution result"""
    trade_id: str
    token: str
    action: str
    amount_in: float
    amount_out: float
    price: float
    slippage_actual: float
    executed_at: datetime
    transaction_signature: Optional[str] = None
    status: str  # success, failed, pending


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/strategies", response_model=List[StrategyConfig])
async def list_strategies(
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List all trading strategies

    Requires: READ permission
    """
    service = TradingService(settings)
    strategies = await service.get_strategies(db)

    return [
        StrategyConfig(
            name=strategy.name,
            enabled=strategy.enabled,
            max_position_size=strategy.max_position_size,
            stop_loss_pct=strategy.stop_loss_pct,
            take_profit_pct=strategy.take_profit_pct,
            risk_tolerance=strategy.risk_tolerance
        )
        for strategy in strategies
    ]


@router.get("/positions", response_model=List[Position])
async def list_positions(
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    List trading positions

    Requires: READ permission

    Args:
        status: Filter by status (open, closed, liquidated)
    """
    service = TradingService(settings)

    # Parse status
    position_status = PositionStatus(status) if status else None

    positions = await service.get_positions(
        db=db,
        status=position_status,
        user_id=current_user.username
    )

    return [
        Position(
            position_id=pos.position_id,
            token=pos.token,
            size=pos.size,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            pnl=pos.pnl,
            pnl_pct=pos.pnl_pct,
            opened_at=pos.opened_at,
            status=pos.status.value
        )
        for pos in positions
    ]


@router.get("/signals", response_model=List[TradeSignal])
async def get_signals(
    token: Optional[str] = None,
    strategy: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get AI-generated trade signals

    Requires: READ permission

    Args:
        token: Filter by token symbol
        strategy: Filter by strategy name
    """
    if not settings.feature_advanced_trading:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Advanced trading features are disabled"
        )

    service = TradingService(settings)
    signals = await service.generate_signals(db=db, token=token, strategy=strategy)

    return [
        TradeSignal(
            signal_id=sig.signal_id,
            token=sig.token,
            action=sig.action.value,
            confidence=sig.confidence,
            price_target=sig.price_target,
            strategy=sig.strategy,
            reasoning=sig.reasoning,
            generated_at=sig.generated_at
        )
        for sig in signals
    ]


@router.post("/execute", response_model=TradeResult)
async def execute_trade(
    trade: TradeExecution,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.EXECUTE)),
    settings: Settings = Depends(get_settings)
):
    """
    Execute a trade

    Requires: EXECUTE permission

    Args:
        trade: Trade execution details

    Returns:
        Trade execution result

    Raises:
        HTTPException: If trading is disabled or execution fails
    """
    try:
        service = TradingService(settings)

        execution = await service.execute_trade(
            db=db,
            token=trade.token,
            action=trade.action,
            amount=trade.amount,
            slippage_bps=trade.slippage_bps,
            user_id=current_user.username
        )

        return TradeResult(
            trade_id=execution.trade_id,
            token=execution.token,
            action=execution.action.value,
            amount_in=execution.amount_in,
            amount_out=execution.amount_out,
            price=execution.price,
            slippage_actual=execution.slippage_actual,
            executed_at=execution.executed_at,
            transaction_signature=execution.transaction_signature,
            status=execution.status
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trade execution failed: {str(e)}"
        )


@router.get("/performance")
async def get_performance(
    period: str = "24h",
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.READ)),
    settings: Settings = Depends(get_settings)
):
    """
    Get trading performance metrics

    Requires: READ permission

    Args:
        period: Time period (1h, 24h, 7d, 30d, all)

    Returns:
        Performance metrics
    """
    service = TradingService(settings)
    return await service.get_performance(db=db, period=period)


@router.post("/strategies/{strategy_name}/enable")
async def enable_strategy(
    strategy_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    settings: Settings = Depends(get_settings)
):
    """
    Enable a trading strategy

    Requires: WRITE permission
    """
    service = TradingService(settings)
    strategy = await service.update_strategy_status(db=db, strategy_name=strategy_name, enabled=True)
    return {"strategy": strategy.name, "enabled": strategy.enabled}


@router.post("/strategies/{strategy_name}/disable")
async def disable_strategy(
    strategy_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(require_permission(Permission.WRITE)),
    settings: Settings = Depends(get_settings)
):
    """
    Disable a trading strategy

    Requires: WRITE permission
    """
    service = TradingService(settings)
    strategy = await service.update_strategy_status(db=db, strategy_name=strategy_name, enabled=False)
    return {"strategy": strategy.name, "enabled": strategy.enabled}


@router.get("/mode")
async def get_trading_mode(
    settings: Settings = Depends(get_settings)
):
    """Get current trading mode (paper/live) - public endpoint"""
    return {
        "mode": settings.trading_mode.value,
        "max_position_size": settings.max_position_size,
        "stop_loss_pct": settings.stop_loss_pct,
        "take_profit_pct": settings.take_profit_pct
    }
