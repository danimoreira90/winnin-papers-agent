"""ExtractSectionTool: one atomic metadata-filter fetch + concat (SR-5).

Fetch all chunks of (paper_id, section) from Chroma via metadata-only
filter (no embedding involved), sort by char_start, concat with blank
lines. The RAGAgent decides when to call this; the tool just executes.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Chunk, ToolResult
from papers_agent.core.ports import VectorStoreClient
from papers_agent.tools.base import Tool

log = get_logger(__name__)


class ExtractSectionInput(BaseModel):
    """Input for ExtractSectionTool."""

    model_config = ConfigDict(frozen=True)

    paper_id: Literal["attention", "bert", "rag", "react", "toolformer"]
    section: Literal[
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "conclusion",
    ]


class ExtractSectionOutput(BaseModel):
    """Output for ExtractSectionTool."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    section: str
    text: str
    found: bool
    chunks: list[Chunk] = Field(default_factory=list)


class ExtractSectionTool(Tool[ExtractSectionInput]):
    """Extracts a named section of one paper from the vector store."""

    name = "extract_section"
    description = "Extracts a named section of one paper from the vector store."
    input_schema = ExtractSectionInput
    output_schema = ExtractSectionOutput

    def __init__(self, vs: VectorStoreClient) -> None:
        self._vs = vs

    async def run(self, inp: ExtractSectionInput) -> ToolResult:
        try:
            where = {
                "$and": [
                    {"paper_id": inp.paper_id},
                    {"section": inp.section},
                ]
            }
            result, ms = await self._measured(self._vs.get(where=where))
            if not result.ids:
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    data=ExtractSectionOutput(
                        paper_id=inp.paper_id,
                        section=inp.section,
                        text="",
                        found=False,
                        chunks=[],
                    ),
                    metadata={"duration_ms": ms, "n_chunks": 0},
                )
            chunks = sorted(result.to_chunks(), key=lambda c: c.char_start)
            text = "\n\n".join(c.text for c in chunks)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=ExtractSectionOutput(
                    paper_id=inp.paper_id,
                    section=inp.section,
                    text=text,
                    found=True,
                    chunks=chunks,
                ),
                metadata={"duration_ms": ms, "n_chunks": len(chunks)},
            )
        except Exception as exc:
            # Tool boundary per PLAN sec.1.7: capture, log, return success=False.
            log.exception(
                "tool.extract_section.error",
                paper_id=inp.paper_id,
                section=inp.section,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=ExtractSectionOutput(
                    paper_id=inp.paper_id,
                    section=inp.section,
                    text="",
                    found=False,
                    chunks=[],
                ),
                error=str(exc),
            )
