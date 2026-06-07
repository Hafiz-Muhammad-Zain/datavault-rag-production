from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

# Convert the standard postgres:// URL to postgresql+asyncpg://
# asyncpg is the async PostgreSQL driver — SQLAlchemy needs this prefix to use it
# Example: postgresql://raguser:pass@localhost/ragdb
#       -> postgresql+asyncpg://raguser:pass@localhost/ragdb
async_database_url = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
)

# Create the async engine — this is the connection pool
# pool_size=5: keep 5 connections open and ready at all times
# max_overflow=10: allow up to 10 extra connections under heavy load
# pool_pre_ping=True: test connections before using them (detects stale connections)
engine = create_async_engine(
    async_database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,  # set True temporarily if you want to see SQL queries in logs
)

# Session factory — creates AsyncSession objects on demand
# expire_on_commit=False: keep objects usable after commit (important for async)
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# Base class for SQLAlchemy ORM models (used if we add ORM models later)
class Base(DeclarativeBase):
    pass


async def get_db():
    """
    FastAPI dependency — yields a database session for each request.
    Automatically closes the session when the request is done.
    Usage in a route: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        yield session


async def check_db_connection() -> bool:
    """
    Health check — verifies PostgreSQL is reachable.
    Called by the /health endpoint on startup.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True
    except Exception:
        return False
