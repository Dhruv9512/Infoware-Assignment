from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# 1. Create the asynchronous engine
# For SQLite, we enforce write-ahead logging (WAL) mode and foreign keys
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Set to True if you want to see raw SQL queries in your logs
)

# 2. Create the asynchronous session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# 3. Create the Declarative Base class that our models will inherit from
class Base(DeclarativeBase):
    pass


# 4. FastAPI Dependency to provide a database session per web request
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields an asynchronous database session. Ensures the connection
    is safely closed after the web request completes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()