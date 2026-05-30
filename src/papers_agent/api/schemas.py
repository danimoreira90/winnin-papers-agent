"""API-specific request/response shapes.

GET endpoints reuse the domain models (Message, Thread) as response_model
because they are already Pydantic and the JSON shape matches what the
clients expect; duplicating them into DTOs would be premature.
"""

import uuid

from pydantic import BaseModel, Field


class CreateThreadResponse(BaseModel):
    """Body returned by POST /threads."""

    thread_id: uuid.UUID


class PostMessageRequest(BaseModel):
    """Body accepted by POST /threads/{id}/messages."""

    content: str = Field(min_length=1, max_length=10000)


class PostMessageResponse(BaseModel):
    """Body returned by POST /threads/{id}/messages."""

    thread_id: uuid.UUID
    response: str
