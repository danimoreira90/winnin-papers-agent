"""Domain entities and tool-shared value types.

Pydantic v2 models for the paper-analysis pipeline. Every model is
frozen (DTO / value-object semantics): construct once, read many.
Aggregations (ToolResult.metadata dict, PaperContextBundle.items list)
remain technically mutable - Pydantic's frozen blocks attribute
reassignment, not in-place container mutation. Producers should treat
results as immutable by convention.
"""

import datetime
import enum
import pathlib
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(enum.StrEnum):
    """Speaker role on a Message turn."""

    USER = "user"
    ASSISTANT = "assistant"


class PaperMetadata(BaseModel):
    """Static metadata for one of the five papers in the corpus."""

    model_config = ConfigDict(frozen=True)

    paper_id: Literal["attention", "bert", "rag", "react", "toolformer"]
    arxiv_id: str
    title: str
    pdf_path: pathlib.Path


class Chunk(BaseModel):
    """A contiguous slice of a paper's text after PDF extraction.

    chunk_id follows the convention f"{paper_id}:{idx:04d}",
    e.g. "attention:0042". The convention is documented but not
    enforced by validation.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    paper_id: str
    text: str = Field(max_length=8000)
    section: str | None
    page: int | None
    char_start: int
    char_end: int


class ToolResult(BaseModel):
    """Uniform return type for every Tool.run() call.

    metadata is free-form per tool; canonical keys consumers may rely
    on when present: duration_ms, model, n_chunks, tokens_used.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    success: bool
    data: BaseModel
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single turn in a conversation thread."""

    model_config = ConfigDict(frozen=True)

    message_id: uuid.UUID
    thread_id: uuid.UUID
    role: Role
    content: str = Field(min_length=1, max_length=10000)
    created_at: datetime.datetime


class Thread(BaseModel):
    """Conversation thread header (messages live in the repository)."""

    model_config = ConfigDict(frozen=True)

    thread_id: uuid.UUID
    created_at: datetime.datetime
    message_count: int


class PaperContextItem(BaseModel):
    """Pre-retrieved chunks for a single paper, ready to feed a tool."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    chunks: list[Chunk]


class PaperContextBundle(BaseModel):
    """Per-paper context items shared between agent and tool.

    The agent pre-fetches chunks once (RAG step) and passes the bundle
    so downstream tools (compare, summarize, rank) can read without
    re-querying the vector store.
    """

    model_config = ConfigDict(frozen=True)

    items: list[PaperContextItem]

    def get(self, paper_id: str) -> list[Chunk]:
        """Return chunks for the given paper_id, or [] when absent.

        The bundle is sparse: a missing paper means no context was
        retrieved, not an error. The caller decides whether emptiness
        is a problem.
        """
        for item in self.items:
            if item.paper_id == paper_id:
                return item.chunks
        return []

    def to_prompt_block(self) -> str:
        """Markdown-ish block: ## paper_id then chunks joined by '---'.

        Shared format used by compare_papers and rank_papers prompts.
        """
        blocks: list[str] = []
        for item in self.items:
            chunk_blocks = "\n\n---\n\n".join(c.text for c in item.chunks)
            blocks.append(f"## {item.paper_id}\n\n{chunk_blocks}")
        return "\n\n\n".join(blocks)
