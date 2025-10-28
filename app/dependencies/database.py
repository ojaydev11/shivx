"""
Database Dependencies
Provides database session to routes
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db as _get_db


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session

    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Yields:
        AsyncSession: Async database session
    """
    async for session in _get_db():
        yield session
