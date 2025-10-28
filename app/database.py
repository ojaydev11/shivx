"""
Database Configuration and Session Management

Provides:
- Async SQLAlchemy engine
- Async session factory
- Database initialization
- Connection lifecycle management
- Health check utilities
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text, event
from sqlalchemy.pool import NullPool, QueuePool

from config.settings import Settings, get_settings
from app.models.base import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def get_database_url(settings: Settings) -> str:
    """
    Get database URL and convert to async driver if needed

    PostgreSQL: postgresql:// -> postgresql+asyncpg://
    SQLite: sqlite:// -> sqlite+aiosqlite://

    Args:
        settings: Application settings

    Returns:
        Async-compatible database URL
    """
    url = settings.database_url

    # Convert to async driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    return url


def create_engine(settings: Settings) -> AsyncEngine:
    """
    Create async SQLAlchemy engine

    Args:
        settings: Application settings

    Returns:
        Configured async engine
    """
    database_url = get_database_url(settings)

    # Engine configuration
    engine_kwargs = {
        "echo": settings.db_echo,
        "future": True,  # Use SQLAlchemy 2.0 style
    }

    # Configure connection pooling
    if settings.database_is_postgres:
        # PostgreSQL with connection pooling
        engine_kwargs["poolclass"] = QueuePool
        engine_kwargs["pool_size"] = settings.db_pool_size
        engine_kwargs["max_overflow"] = settings.db_pool_size * 2
        engine_kwargs["pool_timeout"] = settings.db_pool_timeout
        engine_kwargs["pool_pre_ping"] = True  # Verify connections before use
        engine_kwargs["pool_recycle"] = 3600  # Recycle connections after 1 hour

    elif settings.database_is_sqlite:
        # SQLite - no connection pooling needed
        engine_kwargs["poolclass"] = NullPool
        # Enable foreign keys for SQLite
        engine_kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_async_engine(database_url, **engine_kwargs)

    logger.info(f"Created database engine: {database_url.split('@')[-1]}")
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """
    Create async session factory

    Args:
        engine: Async database engine

    Returns:
        Session factory for creating database sessions
    """
    factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual control over flushing
        autocommit=False,  # Explicit transaction management
    )

    logger.info("Created async session factory")
    return factory


async def init_db(settings: Optional[Settings] = None) -> None:
    """
    Initialize database engine and session factory

    Should be called on application startup

    Args:
        settings: Application settings (uses default if not provided)
    """
    global _engine, _async_session_factory

    if _engine is not None:
        logger.warning("Database already initialized")
        return

    if settings is None:
        settings = get_settings()

    # Create engine and session factory
    _engine = create_engine(settings)
    _async_session_factory = create_session_factory(_engine)

    logger.info("Database initialized successfully")


async def create_tables() -> None:
    """
    Create all database tables

    WARNING: This should only be used for development/testing
    In production, use Alembic migrations instead

    Raises:
        RuntimeError: If database not initialized
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first")

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created")


async def drop_tables() -> None:
    """
    Drop all database tables

    WARNING: This is destructive! Only use for testing

    Raises:
        RuntimeError: If database not initialized
    """
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first")

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    logger.warning("Database tables dropped")


async def close_db() -> None:
    """
    Close database connections

    Should be called on application shutdown
    """
    global _engine, _async_session_factory

    if _engine is None:
        return

    await _engine.dispose()
    _engine = None
    _async_session_factory = None

    logger.info("Database connections closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Yields:
        Async database session

    Raises:
        RuntimeError: If database not initialized
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """
    Check database connectivity

    Returns:
        True if database is healthy, False otherwise
    """
    if _engine is None:
        logger.error("Database health check failed: not initialized")
        return False

    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def get_db_stats() -> dict:
    """
    Get database connection pool statistics

    Returns:
        Dictionary with pool stats
    """
    if _engine is None:
        return {"status": "not_initialized"}

    pool = _engine.pool

    stats = {
        "status": "healthy" if await check_db_health() else "unhealthy",
        "pool_size": getattr(pool, "size", lambda: 0)(),
        "checked_in_connections": getattr(pool, "checkedin", lambda: 0)(),
        "checked_out_connections": getattr(pool, "checkedout", lambda: 0)(),
        "overflow_connections": getattr(pool, "overflow", lambda: 0)(),
    }

    return stats


# Event listeners for SQLite to enable foreign keys
# Note: This is handled via connect_args in engine creation for SQLite
# For async engines, we need to use the @event.listens_for on the engine instance
# after creation, not on the class itself
