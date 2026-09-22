from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from backend.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base class that all SQLAlchemy ORM models inherit from."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency that yields a scoped async DB session and closes it after the request."""
    async with AsyncSessionLocal() as session:
        yield session
