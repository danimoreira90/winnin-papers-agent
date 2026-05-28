"""SQLAlchemy 2.0 async setup and DDL for thread memory.

Declares the ORM Base, threads/messages tables, cached AsyncEngine and
session factory, plus init_db() (idempotent DDL bootstrap) and
get_session() (FastAPI dependency). CRUD lives in infra/repository.py
(T6.1).
"""

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import ForeignKey, Index
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from papers_agent.core.config import get_settings
from papers_agent.core.logging import get_logger

log = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


class ThreadRow(Base):
    """Persistence row for a conversation thread."""

    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(primary_key=True)
    created_at: Mapped[str]

    messages: Mapped[list["MessageRow"]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
    )


class MessageRow(Base):
    """Persistence row for a single message in a thread."""

    __tablename__ = "messages"

    message_id: Mapped[str] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    role: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[str]

    thread: Mapped[ThreadRow] = relationship(back_populates="messages")

    __table_args__ = (Index("messages_thread_created_idx", "thread_id", "created_at"),)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the cached async engine; URL comes from Settings."""
    return create_async_engine(get_settings().database_url, future=True)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the cached async session factory bound to get_engine()."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def init_db() -> None:
    """Create all tables. Idempotent (CREATE TABLE IF NOT EXISTS semantics)."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db.init.done")


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a per-request AsyncSession."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
