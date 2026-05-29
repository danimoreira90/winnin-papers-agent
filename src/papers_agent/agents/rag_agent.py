"""RAGAgent: retrieval over the 5-paper corpus.

Exposes two methods to the orchestrator (function calling) and two
internal helpers used by AnalystAgent for context pre-fetching. The
function-calling methods take simple types (str, list[str], int) so
the google-genai SDK can extract a clean schema from the signatures.
"""

from typing import cast

from pydantic import ValidationError

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Chunk
from papers_agent.tools.extract_section import (
    ExtractSectionInput,
    ExtractSectionOutput,
    ExtractSectionTool,
)
from papers_agent.tools.search_documents import (
    SearchDocumentsInput,
    SearchDocumentsOutput,
    SearchDocumentsTool,
)

log = get_logger(__name__)


class RAGAgent:
    """Recupera contexto relevante da base vetorial. Stateless."""

    def __init__(
        self,
        search_tool: SearchDocumentsTool,
        extract_tool: ExtractSectionTool,
    ) -> None:
        self._search = search_tool
        self._extract = extract_tool

    # ---- Methods exposed to the orchestrator (function calling) ----

    async def rag_retrieve_context(
        self,
        query: str,
        paper_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> str:
        """Busca semantica na base vetorial dos 5 papers de ML.

        Use para encontrar trechos relevantes a uma pergunta. Pode filtrar
        por paper_ids especificos (ex: ["attention", "bert"]) ou buscar em
        todos (None).

        Args:
            query: Pergunta ou termo de busca em linguagem natural.
            paper_ids: Lista de paper_ids para filtrar, ou None para todos.
                Validos: attention, bert, rag, react, toolformer.
            top_k: Numero de trechos a retornar (1-20).

        Returns:
            Trechos formatados em markdown, cada um prefixado com [paper_id].
        """
        result = await self._search.run(
            SearchDocumentsInput(query=query, paper_ids=paper_ids, top_k=top_k)
        )
        data = cast(SearchDocumentsOutput, result.data)
        if not result.success or not data.chunks:
            return "Nenhum trecho relevante encontrado nos papers."
        return self._format_chunks_md(data.chunks)

    async def rag_extract_section(self, paper_id: str, section: str) -> str:
        """Extrai uma secao especifica de um paper (abstract, introduction,
        related_work, method, experiments, results, conclusion).

        Args:
            paper_id: Um de: attention, bert, rag, react, toolformer.
            section: Uma de: abstract, introduction, related_work, method,
                experiments, results, conclusion.

        Returns:
            Texto da secao com header markdown, ou mensagem se nao detectada.
        """
        try:
            inp = ExtractSectionInput(paper_id=paper_id, section=section)  # type: ignore[arg-type]
        except ValidationError:
            return (
                f"Parametros invalidos: paper_id='{paper_id}', section='{section}'. "
                "Verifique os valores aceitos."
            )
        result = await self._extract.run(inp)
        data = cast(ExtractSectionOutput, result.data)
        if not data.found:
            return f"Secao '{section}' nao detectada no paper '{paper_id}'."
        return f"## {section} ({paper_id})\n\n{data.text}"

    # ---- Internal helpers (used by AnalystAgent via injection) ----

    async def fetch_chunks_for_query(
        self, query: str, paper_id: str, top_k: int = 8
    ) -> list[Chunk]:
        """Return raw chunks (unformatted) for one paper. Internal use."""
        result = await self._search.run(
            SearchDocumentsInput(query=query, paper_ids=[paper_id], top_k=top_k)
        )
        data = cast(SearchDocumentsOutput, result.data)
        return list(data.chunks) if result.success else []

    async def fetch_section_chunks(self, paper_id: str, section: str) -> list[Chunk]:
        """Return raw chunks of a section. Internal use. [] if not found."""
        try:
            inp = ExtractSectionInput(paper_id=paper_id, section=section)  # type: ignore[arg-type]
        except ValidationError:
            return []
        result = await self._extract.run(inp)
        data = cast(ExtractSectionOutput, result.data)
        return list(data.chunks) if data.found else []

    @staticmethod
    def _format_chunks_md(chunks: list[Chunk]) -> str:
        """Format chunks as markdown blocks, each prefixed with [paper_id]."""
        blocks: list[str] = []
        for c in chunks:
            section_tag = f" ({c.section})" if c.section else ""
            blocks.append(f"[{c.paper_id}]{section_tag} {c.text}")
        return "\n\n".join(blocks)
