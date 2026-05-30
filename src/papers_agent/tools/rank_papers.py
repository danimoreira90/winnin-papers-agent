"""RankPapersTool: one atomic LLM call to rank N papers by a criterion (SR-5).

Takes a pre-fetched PaperContextBundle, asks the LLM to produce a strict
ordinal ranking with concrete justifications, defensively re-sorts the
result by rank ASC before returning.
"""

from pydantic import BaseModel, ConfigDict, Field

from papers_agent.core.logging import get_logger
from papers_agent.core.models import PaperContextBundle, ToolResult
from papers_agent.core.ports import LLMClient
from papers_agent.tools.base import Tool

log = get_logger(__name__)


RANK_PROMPT_TEMPLATE = """
Voce e um analista de papers de ML. Ranqueie os papers abaixo segundo o
criterio: "{criterion}".

Para cada paper produza:
- rank: posicao no ranking (1 = mais relevante)
- paper_id: identificador
- justification: 2-3 frases citando ESPECIFICAMENTE o conteudo dos trechos
  fornecidos que sustentam a posicao. NAO use conhecimento previo. NAO seja
  vago ("e relevante porque e bom"). Cite mecanismos, exemplos, ou claims
  concretos do paper.

REGRAS:
- IDIOMA: cada "justification" DEVE ser escrita em portugues do Brasil, mesmo
  quando os contextos abaixo estiverem em ingles. Nao reproduza frases em
  ingles; traduza claims e mecanismos para o portugues.
- Ordem do array ranking DEVE refletir os ranks (rank 1 primeiro).
- Ranks 1..N sequenciais, sem buracos, sem empate.
- Use APENAS o conteudo dos contextos.

Contextos:
{contexts}
""".strip()


class RankedPaper(BaseModel):
    """One entry in the ranked list."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    paper_id: str
    justification: str


class RankPapersInput(BaseModel):
    """Input for RankPapersTool."""

    model_config = ConfigDict(frozen=True)

    criterion: str = Field(min_length=5, max_length=300)
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    contexts: PaperContextBundle


class RankPapersOutput(BaseModel):
    """Output for RankPapersTool."""

    model_config = ConfigDict(frozen=True)

    criterion: str
    ranking: list[RankedPaper]


class RankPapersTool(Tool[RankPapersInput]):
    """Ranks N papers on one criterion via a single LLM call."""

    name = "rank_papers"
    description = "Rank N papers by a criterion; returns rank/paper_id/justification."
    input_schema = RankPapersInput
    output_schema = RankPapersOutput

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, inp: RankPapersInput) -> ToolResult:
        prompt = RANK_PROMPT_TEMPLATE.format(
            criterion=inp.criterion, contexts=inp.contexts.to_prompt_block()
        )
        try:
            raw, ms = await self._measured(
                self._llm.generate(prompt, response_schema=RankPapersOutput)
            )
            data = RankPapersOutput.model_validate_json(raw)
            sorted_ranking = sorted(data.ranking, key=lambda r: r.rank)
            if sorted_ranking != data.ranking:
                data = data.model_copy(update={"ranking": sorted_ranking})
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                metadata={"duration_ms": ms, "n_papers": len(data.ranking)},
            )
        except Exception as exc:
            # Tool boundary per PLAN sec.1.7: capture, log, return success=False.
            log.exception("tool.rank_papers.error", criterion=inp.criterion)
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=RankPapersOutput(criterion=inp.criterion, ranking=[]),
                error=str(exc),
            )
