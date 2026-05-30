"""ComparePapersTool: one atomic LLM call to compare N papers (SR-5).

Takes a pre-fetched PaperContextBundle from the caller (AnalystAgent
does the retrieval), formats it into a prompt, and asks the LLM to
emit a structured comparison. No retrieval, no decisions here.
"""

from pydantic import BaseModel, ConfigDict, Field

from papers_agent.core.logging import get_logger
from papers_agent.core.models import PaperContextBundle, ToolResult
from papers_agent.core.ports import LLMClient
from papers_agent.tools.base import Tool

log = get_logger(__name__)


COMPARE_PROMPT_TEMPLATE = """
Voce e um analista de papers de Machine Learning. Compare os papers abaixo
no aspecto: "{aspect}".

Para cada paper, produza:
- paper_id: o identificador do paper
- summary: sintese de 2-3 frases focada exclusivamente no aspecto pedido
- key_quotes: 1-3 trechos curtos (max 20 palavras cada) extraidos
  LITERALMENTE do contexto fornecido. Nao parafraseie. Se nao houver
  trecho aplicavel, lista vazia.

No final, produza uma "synthesis": paragrafo comparativo paralelo que
contrasta os papers no aspecto.

REGRAS:
- IDIOMA: o "summary" de cada paper e a "synthesis" final DEVEM ser escritos
  em portugues do Brasil, mesmo quando os contextos abaixo estiverem em
  ingles. Os "key_quotes" sao a UNICA excecao: por serem citacoes literais,
  preservam o idioma original do paper.
- Use APENAS o conteudo dos contextos fornecidos. Nao invente.
- Se um paper nao tiver contexto suficiente, declare-o no summary daquele
  paper (ex: "Contexto insuficiente para julgar este aspecto").

Contextos:
{contexts}
""".strip()


class PaperPerspective(BaseModel):
    """One paper's view on the compared aspect."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    summary: str
    key_quotes: list[str]


class ComparePapersInput(BaseModel):
    """Input for ComparePapersTool."""

    model_config = ConfigDict(frozen=True)

    paper_ids: list[str] = Field(min_length=2, max_length=5)
    aspect: str = Field(min_length=3, max_length=200)
    contexts: PaperContextBundle


class ComparePapersOutput(BaseModel):
    """Output for ComparePapersTool."""

    model_config = ConfigDict(frozen=True)

    aspect: str
    perspectives: list[PaperPerspective]
    synthesis: str


class ComparePapersTool(Tool[ComparePapersInput, ComparePapersOutput]):
    """Compares N papers on one aspect via a single LLM call."""

    name = "compare_papers"
    description = "Compare N papers on one aspect; returns per-paper perspectives + synthesis."
    input_schema = ComparePapersInput
    output_schema = ComparePapersOutput

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, inp: ComparePapersInput) -> ToolResult:
        ctx_str = self._format_contexts(inp.contexts)
        prompt = COMPARE_PROMPT_TEMPLATE.format(aspect=inp.aspect, contexts=ctx_str)
        try:
            raw, ms = await self._measured(
                self._llm.generate(prompt, response_schema=ComparePapersOutput)
            )
            data = ComparePapersOutput.model_validate_json(raw)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                metadata={"duration_ms": ms, "n_papers": len(inp.paper_ids)},
            )
        except Exception as exc:
            # Tool boundary per PLAN sec.1.7: capture, log, return success=False.
            log.exception("tool.compare_papers.error", aspect=inp.aspect)
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=ComparePapersOutput(aspect=inp.aspect, perspectives=[], synthesis=""),
                error=str(exc),
            )

    @staticmethod
    def _format_contexts(bundle: PaperContextBundle) -> str:
        """Markdown-ish block: ## paper_id then chunks joined by '---'."""
        blocks: list[str] = []
        for item in bundle.items:
            chunk_blocks = "\n\n---\n\n".join(c.text for c in item.chunks)
            blocks.append(f"## {item.paper_id}\n\n{chunk_blocks}")
        return "\n\n\n".join(blocks)
