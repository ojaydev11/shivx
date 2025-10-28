"""
Database Layer Tests

Tests for all database models, operations, and migrations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.models import (
    User,
    APIKey,
    Position,
    Order,
    SecurityAuditLog,
    PositionStatus,
    OrderAction,
    OrderType,
    OrderStatus,
)
from app.models.base import Base
from app.database import get_database_url
from config.settings import Settings


@pytest.fixture
async def test_engine():
    """Create test database engine with SQLite in-memory"""
    # Use in-memory SQLite for fast testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine):
    """Create test database session"""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


class TestUserModel:
    """Tests for User model"""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session):
        """Test creating a user"""
        user = User(
            username="testuser",
            email="test@example.com",
            password_hash="hash:salt",
            permissions={"read": True, "write": False},
            roles=["user"],
        )

        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        assert user.user_id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.failed_login_attempts == 0
        assert user.created_at is not None

    @pytest.mark.asyncio
    async def test_user_permissions(self, db_session):
        """Test user permission checking"""
        user = User(
            username="testuser2",
            email="test2@example.com",
            password_hash="hash:salt",
            permissions={"read": True, "write": True, "admin": False},
            roles=["user"],
        )

        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("admin") is False

    @pytest.mark.asyncio
    async def test_user_admin_role(self, db_session):
        """Test admin role has all permissions"""
        user = User(
            username="admin",
            email="admin@example.com",
            password_hash="hash:salt",
            permissions={},
            roles=["admin"],
        )

        assert user.has_admin_role is True
        assert user.has_permission("anything") is True

    @pytest.mark.asyncio
    async def test_failed_login_attempts(self, db_session):
        """Test failed login attempt tracking"""
        user = User(
            username="testuser3",
            email="test3@example.com",
            password_hash="hash:salt",
            permissions={},
            roles=[],
        )

        assert user.is_locked is False

        # Increment failed attempts
        user.increment_failed_login_attempts(lockout_duration_minutes=15)
        assert user.failed_login_attempts == 1
        assert user.is_locked is False

        # Lock after 5 attempts
        for _ in range(4):
            user.increment_failed_login_attempts()

        assert user.failed_login_attempts == 5
        assert user.is_locked is True
        assert user.locked_until is not None

        # Reset
        user.reset_failed_login_attempts()
        assert user.failed_login_attempts == 0
        assert user.is_locked is False


class TestAPIKeyModel:
    """Tests for APIKey model"""

    @pytest.mark.asyncio
    async def test_create_api_key(self, db_session):
        """Test creating an API key"""
        # Create user first
        user = User(
            username="keyuser",
            email="keyuser@example.com",
            password_hash="hash:salt",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create API key
        api_key = APIKey(
            key_hash="abc123hash",
            name="Test Key",
            permissions={"read": True},
            rate_limit=1000,
            user_id=user.user_id,
        )

        db_session.add(api_key)
        await db_session.commit()
        await db_session.refresh(api_key)

        assert api_key.key_id is not None
        assert api_key.name == "Test Key"
        assert api_key.rate_limit == 1000
        assert api_key.is_valid is True

    @pytest.mark.asyncio
    async def test_api_key_expiration(self, db_session):
        """Test API key expiration"""
        user = User(username="user2", email="user2@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Expired key
        expired_key = APIKey(
            key_hash="expired_hash",
            name="Expired Key",
            permissions={},
            user_id=user.user_id,
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        assert expired_key.is_expired is True
        assert expired_key.is_valid is False

        # Valid key
        valid_key = APIKey(
            key_hash="valid_hash",
            name="Valid Key",
            permissions={},
            user_id=user.user_id,
            expires_at=datetime.utcnow() + timedelta(days=30),
        )

        assert valid_key.is_expired is False
        assert valid_key.is_valid is True

    @pytest.mark.asyncio
    async def test_api_key_rate_limit(self, db_session):
        """Test rate limit checking"""
        user = User(username="user3", email="user3@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        api_key = APIKey(
            key_hash="ratelimit_hash",
            name="Rate Limited Key",
            permissions={},
            rate_limit=100,
            user_id=user.user_id,
        )

        assert api_key.check_rate_limit(50) is True
        assert api_key.check_rate_limit(99) is True
        assert api_key.check_rate_limit(100) is False
        assert api_key.check_rate_limit(150) is False


class TestPositionModel:
    """Tests for Position model"""

    @pytest.mark.asyncio
    async def test_create_position(self, db_session):
        """Test creating a position"""
        user = User(username="trader", email="trader@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        position = Position(
            user_id=user.user_id,
            token="SOL",
            size=Decimal("10.5"),
            entry_price=Decimal("100.0"),
            current_price=Decimal("105.0"),
            status=PositionStatus.OPEN,
        )

        db_session.add(position)
        await db_session.commit()
        await db_session.refresh(position)

        assert position.position_id is not None
        assert position.token == "SOL"
        assert position.size == Decimal("10.5")

    @pytest.mark.asyncio
    async def test_position_pnl_calculation(self, db_session):
        """Test P&L calculations"""
        user = User(username="trader2", email="trader2@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        position = Position(
            user_id=user.user_id,
            token="BTC",
            size=Decimal("1.0"),
            entry_price=Decimal("50000.0"),
            current_price=Decimal("55000.0"),
            status=PositionStatus.OPEN,
        )

        # P&L = (current - entry) * size = (55000 - 50000) * 1 = 5000
        assert position.pnl == Decimal("5000.0")

        # P&L % = ((current - entry) / entry) * 100 = 10%
        assert position.pnl_pct == Decimal("10.0")

    @pytest.mark.asyncio
    async def test_stop_loss_take_profit(self, db_session):
        """Test stop loss and take profit triggers"""
        user = User(username="trader3", email="trader3@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        position = Position(
            user_id=user.user_id,
            token="ETH",
            size=Decimal("10.0"),
            entry_price=Decimal("2000.0"),
            current_price=Decimal("2000.0"),
            stop_loss=Decimal("1800.0"),
            take_profit=Decimal("2500.0"),
            status=PositionStatus.OPEN,
        )

        # No trigger at current price
        assert position.should_stop_loss() is False
        assert position.should_take_profit() is False

        # Stop loss trigger
        position.update_price(Decimal("1750.0"))
        assert position.should_stop_loss() is True

        # Take profit trigger
        position.update_price(Decimal("2600.0"))
        assert position.should_take_profit() is True

    @pytest.mark.asyncio
    async def test_close_position(self, db_session):
        """Test closing a position"""
        user = User(username="trader4", email="trader4@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        position = Position(
            user_id=user.user_id,
            token="SOL",
            size=Decimal("5.0"),
            entry_price=Decimal("100.0"),
            current_price=Decimal("110.0"),
            status=PositionStatus.OPEN,
        )

        assert position.status == PositionStatus.OPEN
        assert position.closed_at is None

        position.close(Decimal("115.0"))

        assert position.status == PositionStatus.CLOSED
        assert position.closed_at is not None
        assert position.current_price == Decimal("115.0")


class TestOrderModel:
    """Tests for Order model"""

    @pytest.mark.asyncio
    async def test_create_order(self, db_session):
        """Test creating an order"""
        user = User(username="orderuser", email="orderuser@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        order = Order(
            user_id=user.user_id,
            token="SOL",
            action=OrderAction.BUY,
            order_type=OrderType.MARKET,
            amount_in=Decimal("1000.0"),
            amount_out=Decimal("10.0"),
            price=Decimal("100.0"),
            slippage_bps=100,
        )

        db_session.add(order)
        await db_session.commit()
        await db_session.refresh(order)

        assert order.order_id is not None
        assert order.action == OrderAction.BUY
        assert order.status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_order_execution(self, db_session):
        """Test order execution"""
        user = User(username="orderuser2", email="orderuser2@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        order = Order(
            user_id=user.user_id,
            token="BTC",
            action=OrderAction.SELL,
            order_type=OrderType.LIMIT,
            amount_in=Decimal("1.0"),
            amount_out=Decimal("50000.0"),
            price=Decimal("50000.0"),
        )

        assert order.is_pending is True
        assert order.is_executed is False

        order.execute("tx_signature_abc123", Decimal("49900.0"), Decimal("0.2"))

        assert order.is_executed is True
        assert order.transaction_signature == "tx_signature_abc123"
        assert order.executed_at is not None
        assert order.slippage_actual == Decimal("0.2")

    @pytest.mark.asyncio
    async def test_order_failure(self, db_session):
        """Test order failure"""
        user = User(username="orderuser3", email="orderuser3@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        order = Order(
            user_id=user.user_id,
            token="ETH",
            action=OrderAction.BUY,
            order_type=OrderType.MARKET,
            amount_in=Decimal("5000.0"),
            amount_out=Decimal("2.5"),
            price=Decimal("2000.0"),
        )

        order.fail("Insufficient liquidity")

        assert order.is_failed is True
        assert order.failed_reason == "Insufficient liquidity"
        assert order.executed_at is not None


class TestSecurityAuditLogModel:
    """Tests for SecurityAuditLog model"""

    @pytest.mark.asyncio
    async def test_create_audit_log(self, db_session):
        """Test creating audit log entry"""
        user = User(username="audituser", email="audituser@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        log = SecurityAuditLog.create_entry(
            event_type="authentication",
            resource="/api/login",
            action="login",
            success=True,
            user_id=user.user_id,
            ip_address="192.168.1.1",
            details={"method": "password"},
            request_id="req_123",
        )

        db_session.add(log)
        await db_session.commit()
        await db_session.refresh(log)

        assert log.log_id is not None
        assert log.event_type == "authentication"
        assert log.success is True
        assert log.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_audit_log_query(self, db_session):
        """Test querying audit logs"""
        user = User(username="audituser2", email="audituser2@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create multiple log entries
        for i in range(5):
            log = SecurityAuditLog.create_entry(
                event_type="data_access",
                resource=f"/api/resource/{i}",
                action="read",
                success=True,
                user_id=user.user_id,
            )
            db_session.add(log)

        await db_session.commit()

        # Query logs by user
        result = await db_session.execute(
            select(SecurityAuditLog).where(SecurityAuditLog.user_id == user.user_id)
        )
        logs = result.scalars().all()

        assert len(logs) == 5


class TestRelationships:
    """Tests for model relationships"""

    @pytest.mark.asyncio
    async def test_user_api_keys_relationship(self, db_session):
        """Test User -> APIKey relationship"""
        user = User(username="reluser", email="reluser@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create multiple API keys
        for i in range(3):
            api_key = APIKey(
                key_hash=f"hash_{i}",
                name=f"Key {i}",
                permissions={},
                user_id=user.user_id,
            )
            db_session.add(api_key)

        await db_session.commit()

        # Query user with API keys
        result = await db_session.execute(
            select(User).where(User.user_id == user.user_id)
        )
        user_with_keys = result.scalar_one()

        # Load relationship
        await db_session.refresh(user_with_keys, ["api_keys"])
        assert len(user_with_keys.api_keys) == 3

    @pytest.mark.asyncio
    async def test_user_positions_relationship(self, db_session):
        """Test User -> Position relationship"""
        user = User(username="reluser2", email="reluser2@example.com", password_hash="hash:salt")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create positions
        position = Position(
            user_id=user.user_id,
            token="SOL",
            size=Decimal("10.0"),
            entry_price=Decimal("100.0"),
            current_price=Decimal("105.0"),
        )
        db_session.add(position)
        await db_session.commit()

        # Query with relationship
        result = await db_session.execute(
            select(User).where(User.user_id == user.user_id)
        )
        user_with_pos = result.scalar_one()
        await db_session.refresh(user_with_pos, ["positions"])

        assert len(user_with_pos.positions) == 1
        assert user_with_pos.positions[0].token == "SOL"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
