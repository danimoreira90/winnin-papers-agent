"""AnalystAgent: comparative, summary, and ranking analyses.

Stateless. Depends on a RAGAgent (injected) to pre-fetch context and on
a PaperCatalog for paper_id -> title resolution. Exposes three methods
to the orchestrator via function calling; each takes simple types so
google-genai can extract the schema cleanly.
"""

import asyncio
from typing import cast

from pydantic import ValidationError

from papers_agent.agents.rag_agent import RAGAgent
from papers_agent.core.catalog import PaperCatalog
from papers_agent.core.logging import get_logger
from papers_agent.core.models import PaperContextBundle, PaperContextItem
from papers_agent.tools.compare_papers import (
    ComparePapersInput,
    ComparePapersOutput,
    ComparePapersTool,
)
from papers_agent.tools.rank_papers import (
    RankPapersInput,
    RankPapersOutput,
    RankPapersTool,
)
from papers_agent.tools.summarize import (
    SummarizeInput,
    SummarizeOutput,
    SummarizeTool,
)

log = get_logger(__name__)


class AnalystAgent:
    """Analises comparativas, sinteses e rankings. Stateless."""

    def __init__(
        self,
        compare_tool: ComparePapersTool,
        summarize_tool: SummarizeTool,
        rank_tool: RankPapersTool,
        rag_agent: RAGAgent,
        paper_catalog: PaperCatalog,
    ) -> None:
        self._compare = compare_tool
        self._summarize = summarize_tool
        self._rank = rank_tool
        self._rag = rag_agent
        self._catalog = paper_catalog

    async def analyst_compare_papers(self, paper_ids: list[str], aspect: str) -> str:
        """Compara 2+ papers em um aspecto especifico (ex: 'uso de ferramentas
        externas', 'mecanismo de atencao'). Recupera contexto de cada paper e
        produz comparacao estruturada.

        Args:
            paper_ids: Lista de 2-5 paper_ids a comparar.
            aspect: O aspecto/dimensao da comparacao.

        Returns:
            Comparacao em markdown: perspectiva por paper + sintese.
        """
        chunk_lists = await asyncio.gather(
            *(self._rag.fetch_chunks_for_query(aspect, pid, top_k=8) for pid in paper_ids)
        )
        bundle = PaperContextBundle(
            items=[
                PaperContextItem(paper_id=pid, chunks=chunks)
                for pid, chunks in zip(paper_ids, chunk_lists, strict=True)
            ]
        )
        try:
            inp = ComparePapersInput(paper_ids=paper_ids, aspect=aspect, contexts=bundle)
        except ValidationError:
            return (
                "Parametros invalidos para comparacao: forneca 2 a 5 paper_ids e um "
                "aspect entre 3 e 200 caracteres."
            )
        result = await self._compare.run(inp)
        if not result.success:
            return f"Falha ao comparar papers: {result.error}"
        return self._format_compare_md(cast(ComparePapersOutput, result.data))

    async def analyst_summarize_paper(self, paper_id: str, max_bullets: int = 5) -> str:
        """Resume um paper em ate max_bullets bullet points.

        Args:
            paper_id: Um de: attention, bert, rag, react, toolformer.
            max_bullets: Numero maximo de bullets (1-10).

        Returns:
            Titulo + bullets em markdown.
        """
        try:
            title = self._catalog.get_title(paper_id)
        except KeyError:
            return f"paper_id desconhecido: '{paper_id}'."
        abstract_chunks, conclusion_chunks = await asyncio.gather(
            self._rag.fetch_section_chunks(paper_id, "abstract"),
            self._rag.fetch_section_chunks(paper_id, "conclusion"),
        )
        chunks = abstract_chunks + conclusion_chunks
        if not chunks:
            chunks = await self._rag.fetch_chunks_for_query(
                f"resumo contribuicoes {title}", paper_id, top_k=8
            )
        if not chunks:
            return f"Nenhum contexto recuperado para o paper '{paper_id}'."
        try:
            inp = SummarizeInput(
                paper_id=paper_id,  # type: ignore[arg-type]
                title=title,
                context_chunks=chunks,
                max_bullets=max_bullets,
            )
        except ValidationError:
            return (
                f"Parametros invalidos para resumir '{paper_id}': "
                "max_bullets deve estar entre 1 e 10."
            )
        result = await self._summarize.run(inp)
        if not result.success:
            return f"Erro ao resumir '{paper_id}': {result.error}"
        return self._format_bullets_md(cast(SummarizeOutput, result.data))

    async def analyst_rank_papers(self, criterion: str, paper_ids: list[str] | None = None) -> str:
        """Ranqueia papers segundo um criterio, com justificativa por posicao.

        Args:
            criterion: O criterio de ranqueamento.
            paper_ids: Papers a ranquear, ou None para todos os 5.

        Returns:
            Ranking ordenado em markdown com justificativas.
        """
        ids = paper_ids or self._catalog.all_paper_ids()
        chunk_lists = await asyncio.gather(
            *(self._rag.fetch_chunks_for_query(criterion, pid, top_k=8) for pid in ids)
        )
        bundle = PaperContextBundle(
            items=[
                PaperContextItem(paper_id=pid, chunks=chunks)
                for pid, chunks in zip(ids, chunk_lists, strict=True)
            ]
        )
        try:
            inp = RankPapersInput(criterion=criterion, paper_ids=ids, contexts=bundle)
        except ValidationError:
            return (
                "Parametros invalidos para ranquear: forneca 2 a 5 paper_ids e um "
                "criterion entre 5 e 300 caracteres."
            )
        result = await self._rank.run(inp)
        if not result.success:
            return f"Erro ao ranquear: {result.error}"
        return self._format_ranking_md(cast(RankPapersOutput, result.data))

    @staticmethod
    def _format_compare_md(data: ComparePapersOutput) -> str:
        lines = [f"# Comparacao: {data.aspect}", ""]
        for p in data.perspectives:
            lines.append(f"## [{p.paper_id}]")
            lines.append(p.summary)
            for q in p.key_quotes:
                lines.append(f"> {q}")
            lines.append("")
        lines.append("## Sintese")
        lines.append(data.synthesis)
        return "\n".join(lines)

    @staticmethod
    def _format_bullets_md(data: SummarizeOutput) -> str:
        lines = [f"## {data.title} [{data.paper_id}]", ""]
        lines.extend(f"- {b}" for b in data.bullets)
        return "\n".join(lines)

    @staticmethod
    def _format_ranking_md(data: RankPapersOutput) -> str:
        lines = [f"# Ranking: {data.criterion}", ""]
        for r in data.ranking:
            lines.append(f"{r.rank}. [{r.paper_id}] {r.justification}")
        return "\n".join(lines)
