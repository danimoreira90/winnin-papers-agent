"""Static registry of the 5 papers in the corpus.

The catalog is the single source of truth for paper_id -> metadata.
build_catalog is a pure factory (pdf_dir in, PaperCatalog out);
get_paper_catalog wires it to Settings for production use.
"""

import pathlib
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict

from papers_agent.core.config import get_settings
from papers_agent.core.models import PaperMetadata

PaperId = Literal["attention", "bert", "rag", "react", "toolformer"]

_PAPER_RECORDS: tuple[tuple[PaperId, str, str], ...] = (
    ("attention", "1706.03762", "Attention Is All You Need"),
    (
        "bert",
        "1810.04805",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    ),
    (
        "rag",
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    ),
    (
        "react",
        "2210.03629",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
    ),
    (
        "toolformer",
        "2302.04761",
        "Toolformer: Language Models Can Teach Themselves to Use Tools",
    ),
)


class PaperCatalog(BaseModel):
    """Singleton with metadata for the 5 papers in the corpus."""

    model_config = ConfigDict(frozen=True)

    papers: dict[str, PaperMetadata]

    def get_title(self, paper_id: str) -> str:
        """Return the title for paper_id; raises KeyError on miss."""
        try:
            return self.papers[paper_id].title
        except KeyError as exc:
            raise KeyError(self._unknown_msg(paper_id)) from exc

    def get_pdf_path(self, paper_id: str) -> pathlib.Path:
        """Return the resolved PDF path for paper_id; raises KeyError on miss."""
        try:
            return self.papers[paper_id].pdf_path
        except KeyError as exc:
            raise KeyError(self._unknown_msg(paper_id)) from exc

    def all_paper_ids(self) -> list[str]:
        """Return paper_ids in canonical SPEC sec.4 order."""
        return list(self.papers.keys())

    def all(self) -> list[PaperMetadata]:
        """Return PaperMetadata entries in canonical SPEC sec.4 order."""
        return list(self.papers.values())

    def _unknown_msg(self, paper_id: str) -> str:
        return f"unknown paper_id: {paper_id!r}; valid ids: {list(self.papers.keys())}"


def build_catalog(pdf_dir: pathlib.Path) -> PaperCatalog:
    """Build the canonical 5-paper catalog rooted at pdf_dir."""
    papers: dict[str, PaperMetadata] = {
        paper_id: PaperMetadata(
            paper_id=paper_id,
            arxiv_id=arxiv_id,
            title=title,
            pdf_path=pdf_dir / f"{paper_id}.pdf",
        )
        for paper_id, arxiv_id, title in _PAPER_RECORDS
    }
    return PaperCatalog(papers=papers)


@lru_cache(maxsize=1)
def get_paper_catalog() -> PaperCatalog:
    """Lazy singleton; resolves pdf_dir from Settings."""
    return build_catalog(get_settings().pdf_dir)
