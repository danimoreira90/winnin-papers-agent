"""PDF parsing, section detection, and recursive chunking.

Pipeline: parse_pdf -> _build_page_map -> detect_sections -> chunk_text
(per section) -> Chunk entities. SPEC sec.10 governs heading regex and
abstract-by-convention rule for pre-heading text.
"""

import asyncio
import pathlib
import re
from bisect import bisect_right

import fitz  # type: ignore[import-untyped]

from papers_agent.core.logging import get_logger
from papers_agent.core.models import Chunk

log = get_logger(__name__)

SECTION_RE = re.compile(
    r"^(Abstract|Introduction|Related Work|Method[s]?|Experiments|Results|Conclusion)\b",
    re.MULTILINE | re.IGNORECASE,
)

SECTION_CANON: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "method": "method",
    "methods": "method",
    "experiments": "experiments",
    "results": "results",
    "conclusion": "conclusion",
}

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


async def parse_pdf(pdf_path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(page_num_1based, text), ...]. Sync fitz wrapped in to_thread."""

    def _sync() -> list[tuple[int, str]]:
        doc = fitz.open(str(pdf_path))
        try:
            return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        finally:
            doc.close()

    pages = await asyncio.to_thread(_sync)
    log.info("pdf.parse.done", path=str(pdf_path), n_pages=len(pages))
    return pages


def detect_sections(full_text: str) -> list[tuple[int, int, str | None]]:
    """Cover full_text with (start, end, canonical_label) contiguously.

    Pre-heading text and the no-match case both yield label 'abstract'
    (SPEC sec.10 convention).
    """
    matches = list(SECTION_RE.finditer(full_text))
    if not matches:
        return [(0, len(full_text), "abstract")]
    sections: list[tuple[int, int, str | None]] = []
    if matches[0].start() > 0:
        sections.append((0, matches[0].start(), "abstract"))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        sections.append((start, end, SECTION_CANON.get(match.group(1).lower())))
    return sections


def chunk_text(text: str, target_size: int = 800, overlap: int = 100) -> list[tuple[int, int, str]]:
    """Recursive-separator chunker. Approximation: 1 token ~= 4 chars."""
    target_chars = target_size * 4
    overlap_chars = overlap * 4
    if len(text) <= target_chars:
        return [(0, len(text), text)]
    fragments: list[tuple[int, int]] | None = None
    for sep in SEPARATORS:
        candidate = _split_with_offsets(text, sep)
        if candidate and all((e - s) <= target_chars for s, e in candidate):
            fragments = candidate
            break
    if fragments is None:
        return _hard_split(text, target_chars, overlap_chars)
    return _greedy_concat(text, fragments, target_chars, overlap_chars)


def _split_with_offsets(text: str, sep: str) -> list[tuple[int, int]]:
    """Split text by sep into non-empty (start, end) fragments."""
    fragments: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = text.find(sep, cursor)
        if idx == -1:
            if cursor < len(text):
                fragments.append((cursor, len(text)))
            return fragments
        if idx > cursor:
            fragments.append((cursor, idx))
        cursor = idx + len(sep)


def _greedy_concat(
    text: str,
    fragments: list[tuple[int, int]],
    target_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int, str]]:
    """Pack adjacent fragments into chunks <= target_chars with overlap."""
    chunks: list[tuple[int, int, str]] = []
    i = 0
    n = len(fragments)
    while i < n:
        chunk_start = fragments[i][0]
        chunk_end = fragments[i][1]
        j = i + 1
        while j < n and (fragments[j][1] - chunk_start) <= target_chars:
            chunk_end = fragments[j][1]
            j += 1
        chunks.append((chunk_start, chunk_end, text[chunk_start:chunk_end]))
        if j >= n:
            break
        overlap_target = chunk_end - overlap_chars
        next_i = j
        while next_i > i + 1 and fragments[next_i - 1][0] >= overlap_target:
            next_i -= 1
        i = next_i
    return chunks


def _hard_split(text: str, target_chars: int, overlap_chars: int) -> list[tuple[int, int, str]]:
    """Character-window fallback when no separator fits."""
    chunks: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + target_chars, len(text))
        chunks.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _build_page_map(pages: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """Concat page texts with '\\n\\n'; record cumulative end offsets."""
    parts: list[str] = []
    cum_ends: list[int] = []
    cursor = 0
    sep = "\n\n"
    for idx, (_, page_text) in enumerate(pages):
        parts.append(page_text)
        cursor += len(page_text)
        cum_ends.append(cursor)
        if idx < len(pages) - 1:
            parts.append(sep)
            cursor += len(sep)
    return "".join(parts), cum_ends


def _page_for_offset(offset: int, cum_ends: list[int]) -> int:
    """1-based page containing offset (binary search over cum_ends)."""
    idx = bisect_right(cum_ends, offset)
    return min(idx + 1, len(cum_ends))


def build_chunks(paper_id: str, pages: list[tuple[int, str]]) -> list[Chunk]:
    """Glue parse output into ordered Chunk entities."""
    full_text, page_map = _build_page_map(pages)
    sections = detect_sections(full_text)
    chunks: list[Chunk] = []
    idx = 0
    for sec_start, sec_end, sec_label in sections:
        for local_start, local_end, chunk_str in chunk_text(full_text[sec_start:sec_end]):
            abs_start = sec_start + local_start
            abs_end = sec_start + local_end
            chunks.append(
                Chunk(
                    chunk_id=f"{paper_id}:{idx:04d}",
                    paper_id=paper_id,
                    text=chunk_str,
                    section=sec_label,
                    page=_page_for_offset(abs_start, page_map),
                    char_start=abs_start,
                    char_end=abs_end,
                )
            )
            idx += 1
    log.info(
        "pdf.chunks.built",
        paper_id=paper_id,
        n_chunks=len(chunks),
        n_sections=len(sections),
    )
    return chunks
