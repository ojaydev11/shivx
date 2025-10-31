"""
Trading Database Models
"""

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, Boolean, DateTime, JSON, ForeignKey, Enum as SQLEnum
from datetime import datetime
from typing import Optional, Dict, Any
import enum

from app.models.base import Base, TimestampMixin


class PositionStatus(str, enum.Enum):
    """Position status enum"""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class TradeAction(str, enum.Enum):
    """Trade action enum"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Position(Base, TimestampMixin):
    """Trading position model"""
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    token: Mapped[str] = mapped_column(String(50), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    status: Mapped[PositionStatus] = mapped_column(
        SQLEnum(PositionStatus),
        nullable=False,
        default=PositionStatus.OPEN
    )

    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    signals: Mapped[list["TradeSignal"]] = relationship(back_populates="position", cascade="all, delete-orphan")
    executions: Mapped[list["TradeExecution"]] = relationship(back_populates="position", cascade="all, delete-orphan")


class TradeSignal(Base, TimestampMixin):
    """AI-generated trading signal model"""
    __tablename__ = "trade_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    position_id: Mapped[Optional[int]] = mapped_column(ForeignKey("positions.id"), nullable=True)

    token: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[TradeAction] = mapped_column(SQLEnum(TradeAction), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    price_target: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    reasoning: Mapped[str] = mapped_column(String(500), nullable=False)

    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Additional fields
    expected_profit_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_risk_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    position_size_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationship
    position: Mapped[Optional["Position"]] = relationship(back_populates="signals")


class TradeExecution(Base, TimestampMixin):
    """Trade execution record model"""
    __tablename__ = "trade_executions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    position_id: Mapped[Optional[int]] = mapped_column(ForeignKey("positions.id"), nullable=True)

    token: Mapped[str] = mapped_column(String(50), nullable=False)
    action: Mapped[TradeAction] = mapped_column(SQLEnum(TradeAction), nullable=False)
    amount_in: Mapped[float] = mapped_column(Float, nullable=False)
    amount_out: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_actual: Mapped[float] = mapped_column(Float, nullable=False)

    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    transaction_signature: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # success, failed, pending

    # Error information if failed
    error_message: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationship
    position: Mapped[Optional["Position"]] = relationship(back_populates="executions")


class Strategy(Base, TimestampMixin):
    """Trading strategy configuration model"""
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    max_position_size: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False)
    take_profit_pct: Mapped[float] = mapped_column(Float, nullable=False)
    risk_tolerance: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")

    # Strategy parameters (JSON)
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Performance tracking
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_profit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
