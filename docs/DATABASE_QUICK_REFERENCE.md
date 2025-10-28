# Database Quick Reference Guide

## Table of Contents
- [Environment Setup](#environment-setup)
- [Database Initialization](#database-initialization)
- [Common Operations](#common-operations)
- [Migration Commands](#migration-commands)
- [Model Reference](#model-reference)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Required Environment Variables
```bash
# In .env file
SHIVX_DATABASE_URL=postgresql://user:pass@localhost:5432/shivx
# OR for development
SHIVX_DATABASE_URL=sqlite:///./data/shivx.db

# Connection pool settings (PostgreSQL only)
SHIVX_DB_POOL_SIZE=5
SHIVX_DB_POOL_TIMEOUT=30
SHIVX_DB_ECHO=false  # Set to true to see SQL queries
```

---

## Database Initialization

### On Application Startup (FastAPI)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import init_db, close_db
from config.settings import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    await init_db(settings)
    yield
    # Shutdown
    await close_db()

app = FastAPI(lifespan=lifespan)
```

### Manual Initialization
```python
from app.database import init_db, create_tables

# Initialize database engine and session factory
await init_db()

# Create tables (development only - use Alembic in production)
await create_tables()
```

---

## Common Operations

### Get Database Session
```python
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.dependencies.database import get_db

@app.get("/example")
async def example(db: AsyncSession = Depends(get_db)):
    # Use db here
    pass
```

### Create a User
```python
from app.models import User

user = User(
    username="johndoe",
    email="john@example.com",
    password_hash="pbkdf2_hash:salt",  # Use hardening.py to hash
    permissions={"read": True, "write": True},
    roles=["user"],
)

db.add(user)
await db.commit()
await db.refresh(user)
```

### Query Users
```python
from sqlalchemy import select
from app.models import User

# Get all users
result = await db.execute(select(User))
users = result.scalars().all()

# Get one user by ID
result = await db.execute(select(User).where(User.user_id == user_id))
user = result.scalar_one_or_none()

# Get user with relationships
result = await db.execute(select(User).where(User.user_id == user_id))
user = result.scalar_one()
await db.refresh(user, ["api_keys", "positions", "orders"])
```

### Create a Position
```python
from decimal import Decimal
from app.models import Position, PositionStatus

position = Position(
    user_id=user.user_id,
    token="SOL",
    size=Decimal("10.5"),
    entry_price=Decimal("100.0"),
    current_price=Decimal("100.0"),
    stop_loss=Decimal("95.0"),
    take_profit=Decimal("110.0"),
    status=PositionStatus.OPEN,
    strategy="RL Trading (PPO)",
)

db.add(position)
await db.commit()
await db.refresh(position)

# Access calculated properties
print(f"P&L: ${position.pnl} ({position.pnl_pct}%)")
```

### Create an Order
```python
from decimal import Decimal
from app.models import Order, OrderAction, OrderType, OrderStatus

order = Order(
    user_id=user.user_id,
    position_id=position.position_id,
    token="SOL",
    action=OrderAction.BUY,
    order_type=OrderType.MARKET,
    amount_in=Decimal("1000.0"),
    amount_out=Decimal("10.0"),
    price=Decimal("100.0"),
    slippage_bps=100,
)

db.add(order)
await db.commit()

# Execute order
order.execute(
    transaction_signature="tx_sig_123",
    actual_amount_out=Decimal("9.95"),
    actual_slippage=Decimal("0.5"),
)
await db.commit()
```

### Create Audit Log
```python
from app.models import SecurityAuditLog

log = SecurityAuditLog.create_entry(
    event_type="authentication",
    resource="/api/login",
    action="login",
    success=True,
    user_id=user.user_id,
    ip_address="192.168.1.1",
    details={"method": "password", "user_agent": "..."},
    request_id="req_12345",
)

db.add(log)
await db.commit()
```

### Update a Record
```python
# Fetch record
result = await db.execute(select(Position).where(Position.position_id == pos_id))
position = result.scalar_one()

# Update
position.update_price(Decimal("105.0"))
await db.commit()
```

### Soft Delete
```python
# Mark as inactive
user.is_active = False
await db.commit()

# Query only active users
result = await db.execute(
    select(User).where(User.is_active == True)
)
active_users = result.scalars().all()
```

---

## Migration Commands

### Create a New Migration
```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "Add new field to User"

# Create empty migration
alembic revision -m "Custom migration"
```

### Apply Migrations
```bash
# Apply all pending migrations
alembic upgrade head

# Apply specific number of migrations
alembic upgrade +2

# Apply to specific revision
alembic upgrade <revision_id>
```

### Rollback Migrations
```bash
# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision_id>

# Rollback all
alembic downgrade base
```

### View Migration Status
```bash
# Show current version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic heads
```

---

## Model Reference

### User
- **Primary Key**: `user_id` (UUID)
- **Unique Fields**: `username`, `email`
- **Relationships**: `api_keys`, `positions`, `orders`
- **Key Methods**: `has_permission()`, `increment_failed_login_attempts()`

### APIKey
- **Primary Key**: `key_id` (UUID)
- **Foreign Keys**: `user_id`
- **Unique Fields**: `key_hash`
- **Key Methods**: `is_valid`, `check_rate_limit()`, `increment_usage()`

### Position
- **Primary Key**: `position_id` (UUID)
- **Foreign Keys**: `user_id`
- **Enums**: `PositionStatus` (OPEN, CLOSED, LIQUIDATED)
- **Key Properties**: `pnl`, `pnl_pct`, `position_value`
- **Key Methods**: `should_stop_loss()`, `should_take_profit()`, `close()`

### Order
- **Primary Key**: `order_id` (UUID)
- **Foreign Keys**: `user_id`, `position_id` (nullable)
- **Enums**:
  - `OrderAction` (BUY, SELL)
  - `OrderType` (MARKET, LIMIT, STOP_LOSS, TAKE_PROFIT)
  - `OrderStatus` (PENDING, EXECUTED, FAILED, CANCELLED)
- **Key Methods**: `execute()`, `fail()`, `cancel()`

### SecurityAuditLog
- **Primary Key**: `log_id` (UUID)
- **Foreign Keys**: `user_id` (SET NULL on delete)
- **Immutable**: Cannot be updated or deleted
- **Factory Method**: `create_entry()`

---

## Troubleshooting

### Connection Pool Exhausted
```python
# Check pool statistics
from app.database import get_db_stats

stats = await get_db_stats()
print(stats)
# {"pool_size": 5, "checked_out": 5, ...}
```

**Solution**: Increase pool size in settings or check for connection leaks.

### Migration Conflicts
```bash
# If you have multiple heads
alembic merge <head1> <head2> -m "Merge branches"

# Then upgrade
alembic upgrade head
```

### Foreign Key Constraint Violation
**Error**: `FOREIGN KEY constraint failed`

**Solution**: Ensure parent record exists before creating child:
```python
# Check if user exists first
result = await db.execute(select(User).where(User.user_id == user_id))
user = result.scalar_one_or_none()
if not user:
    raise ValueError("User not found")

# Then create child record
position = Position(user_id=user_id, ...)
```

### Decimal Precision Issues
**Problem**: Float values causing rounding errors

**Solution**: Always use `Decimal` from Python's `decimal` module:
```python
from decimal import Decimal

# ❌ WRONG
price = 100.5

# ✅ CORRECT
price = Decimal("100.5")
```

### Async Context Issues
**Error**: `RuntimeError: Task attached to a different loop`

**Solution**: Ensure all database operations are awaited:
```python
# ❌ WRONG
user = db.execute(select(User))

# ✅ CORRECT
result = await db.execute(select(User))
user = result.scalar_one()
```

---

## Best Practices

### Always Use Transactions
```python
async with db.begin():
    # Multiple operations
    db.add(user)
    db.add(position)
    # Auto-commit if no exception
```

### Use Context Managers for Sessions
```python
async with async_sessionmaker() as session:
    # Do work
    pass
# Session automatically closed
```

### Batch Operations
```python
# Insert many records efficiently
users = [User(...) for _ in range(100)]
db.add_all(users)
await db.commit()
```

### Query Optimization
```python
# Use selectinload for eager loading
from sqlalchemy.orm import selectinload

result = await db.execute(
    select(User).options(selectinload(User.positions))
)
user = result.scalar_one()
# positions already loaded, no extra query
```

---

## Security Checklist

- [ ] Never store passwords in plaintext (use PBKDF2 from hardening.py)
- [ ] Never store API keys in plaintext (use SHA256 hash)
- [ ] Always validate foreign key references exist
- [ ] Use parameterized queries (SQLAlchemy does this automatically)
- [ ] Enable SSL for production database connections
- [ ] Rotate database credentials regularly
- [ ] Monitor audit logs for suspicious activity
- [ ] Back up database daily
- [ ] Test disaster recovery procedures

---

## Performance Tips

1. **Index frequently queried fields** (already done for most fields)
2. **Use connection pooling** (configured automatically)
3. **Avoid N+1 queries** (use `selectinload` or `joinedload`)
4. **Batch inserts** instead of one-by-one
5. **Use async for I/O operations** (already enforced)
6. **Monitor slow queries** (enable `db_echo` in dev)

---

**Last Updated**: October 28, 2025
**Version**: 1.0.0
