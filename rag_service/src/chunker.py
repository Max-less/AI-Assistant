"""
Document chunking for RAG pipeline.
Splits documents into overlapping chunks of target token size.
"""

import re
from dataclasses import dataclass, field
from loaders import Document


@dataclass
class Chunk:
    text: str
    source: str       # file path from Document.source
    chunk_id: str     # "{filename}::{index}"
    metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def split_by_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on double newlines.
    Fallback: if only 1 paragraph, split on single newlines.
    Strips empty results.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) <= 1 and "\n" in text:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    return paragraphs


def merge_to_target_size(
    paragraphs: list[str],
    target_tokens: int = 300,
    overlap: int = 1,
) -> list[str]:
    """
    Merge consecutive paragraphs until reaching target_tokens.
    overlap: number of paragraphs to repeat from the end of the previous chunk.
    Never splits a paragraph in the middle.
    """
    if not paragraphs:
        return []

    chunks = []
    current_parts = []
    current_tokens = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        para_tokens = estimate_tokens(para)

        # Single paragraph exceeds target — flush current, add it as own chunk
        if para_tokens > target_tokens:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(para)
            i += 1
            continue

        if current_tokens + para_tokens > target_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))

            # Overlap: keep last paragraphs only if they fit with the next one
            overlap_parts = current_parts[-overlap:] if overlap > 0 and len(current_parts) >= overlap else []
            overlap_tokens = sum(estimate_tokens(p) for p in overlap_parts)

            if overlap_parts and overlap_tokens + para_tokens <= target_tokens:
                current_parts = overlap_parts
                current_tokens = overlap_tokens
            else:
                current_parts = []
                current_tokens = 0
            continue  # re-evaluate this paragraph against the new chunk

        current_parts.append(para)
        current_tokens += para_tokens
        i += 1

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def chunk_document(doc: Document, target_tokens: int = 300, overlap: int = 1) -> list[Chunk]:
    """
    Split a Document into Chunks.
    Returns list of Chunk with IDs like "filename::0", "filename::1", etc.
    """
    paragraphs = split_by_paragraphs(doc.text)
    merged = merge_to_target_size(paragraphs, target_tokens, overlap)

    filename = doc.metadata.get("filename", doc.source)
    chunks = []

    for i, text in enumerate(merged):
        chunk = Chunk(
            text=text,
            source=doc.source,
            chunk_id=f"{filename}::{i}",
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "char_count": len(text),
                "estimated_tokens": estimate_tokens(text),
            },
        )
        chunks.append(chunk)

    return chunks
