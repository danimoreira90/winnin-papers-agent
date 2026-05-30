"""SummarizeTool: one atomic LLM call to summarize a single paper (SR-5).

Receives pre-fetched context chunks from the caller (AnalystAgent),
formats them into a prompt, asks for up-to-max_bullets bullets, and
defensively truncates if the model overshoots.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Chunk, ToolResult
from papers_agent.core.ports import LLMClient
from papers_agent.tools.base import Tool

log = get_logger(__name__)


SUMMARIZE_PROMPT_TEMPLATE = """
Resuma o paper "{title}" em ATE {max_bullets} bullet points substantivos.

Cada bullet:
- Uma frase concreta com fato, contribuicao tecnica ou limitacao.
- PROIBIDO bullets vagos: "e importante", "e interessante", "traz contribuicoes",
  "tem impacto". Se nao houver substancia, omita o bullet.

Use APENAS o conteudo do contexto. Nao invente. Nao use conhecimento previo.

Contexto:
{context}
""".strip()


class SummarizeInput(BaseModel):
    """Input for SummarizeTool."""

    model_config = ConfigDict(frozen=True)

    paper_id: Literal["attention", "bert", "rag", "react", "toolformer"]
    title: str
    context_chunks: list[Chunk] = Field(min_length=1)
    max_bullets: int = Field(default=5, ge=1, le=10)


class SummarizeOutput(BaseModel):
    """Output for SummarizeTool."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    bullets: list[str] = Field(min_length=1)


class SummarizeTool(Tool[SummarizeInput, SummarizeOutput]):
    """Summarizes one paper into up to N substantive bullets."""

    name = "summarize"
    description = "Summarize one paper into up to max_bullets substantive bullets."
    input_schema = SummarizeInput
    output_schema = SummarizeOutput

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, inp: SummarizeInput) -> ToolResult:
        ctx = "\n\n---\n\n".join(c.text for c in inp.context_chunks)
        prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            title=inp.title,
            max_bullets=inp.max_bullets,
            context=ctx,
        )
        try:
            raw, ms = await self._measured(
                self._llm.generate(prompt, response_schema=SummarizeOutput)
            )
            data = SummarizeOutput.model_validate_json(raw)
            if len(data.bullets) > inp.max_bullets:
                data = data.model_copy(update={"bullets": data.bullets[: inp.max_bullets]})
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                metadata={"duration_ms": ms, "n_bullets": len(data.bullets)},
            )
        except Exception as exc:
            # Tool boundary per PLAN sec.1.7: capture, log, return success=False.
            log.exception("tool.summarize.error", paper_id=inp.paper_id)
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=SummarizeOutput(
                    paper_id=inp.paper_id,
                    title=inp.title,
                    bullets=["[summarization failed]"],
                ),
                error=str(exc),
            )
