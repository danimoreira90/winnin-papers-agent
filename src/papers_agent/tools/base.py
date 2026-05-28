"""Abstract Tool base class for atomic tools (SR-5).

Each tool declares its name, description, and Pydantic IO schemas as
ClassVars, implements async run(inp) -> ToolResult, and may use the
_measured helper to populate ToolResult.metadata['duration_ms'].
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from time import monotonic
from typing import ClassVar

from pydantic import BaseModel

from papers_agent.core.models import ToolResult


class Tool[InputT: BaseModel, OutputT: BaseModel](ABC):
    """Abstract base for all atomic tools.

    Subclasses MUST set the four ClassVars (name, description,
    input_schema, output_schema) and implement run().
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, inp: InputT) -> ToolResult:
        """Execute the tool. Must be idempotent and stateless (SR-5)."""

    async def _measured[ResultT](self, coro: Awaitable[ResultT]) -> tuple[ResultT, int]:
        """Await coro and return (result, elapsed_ms).

        Subclasses use this to populate ToolResult.metadata['duration_ms'].
        """
        t0 = monotonic()
        result = await coro
        return result, int((monotonic() - t0) * 1000)
