"""Thread / Message CRUD repository.

Maps ORM rows (string ids, ISO timestamps) to domain entities (UUID,
datetime) and back. Infra layer: no try/except - DB errors propagate
to the route boundary (PLAN sec.1.7).
"""

import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Message, Role, Thread
from papers_agent.infra.db import MessageRow, ThreadRow

log = get_logger(__name__)


class ThreadRepository:
    """CRUD over threads and messages, scoped to one AsyncSession."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_thread(self) -> Thread:
        thread_id = uuid.uuid4()
        now = datetime.datetime.now(datetime.UTC)
        self._session.add(ThreadRow(thread_id=str(thread_id), created_at=now.isoformat()))
        await self._session.commit()
        return Thread(thread_id=thread_id, created_at=now, message_count=0)

    async def list_threads(self) -> list[Thread]:
        result = await self._session.execute(
            select(ThreadRow).options(selectinload(ThreadRow.messages))
        )
        rows = result.scalars().all()
        return [
            Thread(
                thread_id=uuid.UUID(r.thread_id),
                created_at=datetime.datetime.fromisoformat(r.created_at),
                message_count=len(r.messages),
            )
            for r in rows
        ]

    async def thread_exists(self, thread_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            select(ThreadRow.thread_id).where(ThreadRow.thread_id == str(thread_id))
        )
        return result.first() is not None

    async def add_message(self, thread_id: uuid.UUID, role: Role, content: str) -> Message:
        message_id = uuid.uuid4()
        now = datetime.datetime.now(datetime.UTC)
        self._session.add(
            MessageRow(
                message_id=str(message_id),
                thread_id=str(thread_id),
                role=role.value,
                content=content,
                created_at=now.isoformat(),
            )
        )
        await self._session.commit()
        return Message(
            message_id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=now,
        )

    async def list_messages(self, thread_id: uuid.UUID) -> list[Message]:
        result = await self._session.execute(
            select(MessageRow)
            .where(MessageRow.thread_id == str(thread_id))
            .order_by(MessageRow.created_at)
        )
        return [self._to_message(r) for r in result.scalars().all()]

    @staticmethod
    def _to_message(row: MessageRow) -> Message:
        return Message(
            message_id=uuid.UUID(row.message_id),
            thread_id=uuid.UUID(row.thread_id),
            role=Role(row.role),
            content=row.content,
            created_at=datetime.datetime.fromisoformat(row.created_at),
        )
