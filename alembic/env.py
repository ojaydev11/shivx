"""
Alembic Environment Configuration

Supports both sync and async migrations
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import settings and models
from config.settings import get_settings
from app.models.base import Base
from app.database import get_database_url

# Import all models for autogenerate support
from app.models import (
    User,
    APIKey,
    Position,
    Order,
    SecurityAuditLog,
)

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get database URL from settings
settings = get_settings()
database_url = get_database_url(settings)

# Override with env database URL if provided
config.set_main_option("sqlalchemy.url", database_url)

# Set target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Compare column types
        compare_server_default=True,  # Compare server defaults
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """
    Run migrations with the given connection

    Args:
        connection: SQLAlchemy connection
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,  # Compare column types
        compare_server_default=True,  # Compare server defaults
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in async mode

    This is the primary method for async database engines
    """
    # Create async engine
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    For async engines, we use asyncio.run()
    """
    # Check if using async driver
    url = config.get_main_option("sqlalchemy.url")
    if "+asyncpg" in url or "+aiosqlite" in url:
        # Use async migrations
        asyncio.run(run_async_migrations())
    else:
        # Use sync migrations (legacy)
        from sqlalchemy import engine_from_config

        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
