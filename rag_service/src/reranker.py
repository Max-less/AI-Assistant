"""
Cross-encoder reranker: scores (query, chunk) pairs jointly and reorders candidates
by relevance. Default model: BAAI/bge-reranker-v2-m3 (multilingual, supports
Russian + English).
"""

from typing import Callable, Sequence

from sentence_transformers import CrossEncoder

from chunker import Chunk


def rerank_chunks(
    scorer: Callable[[list[tuple[str, str]]], Sequence[float]],
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[Chunk]:
    """Score (query, chunk.text) pairs with `scorer` and return chunks sorted
    by descending score, sliced to top_k. Pure helper — `scorer` is any callable
    so this can be unit-tested without loading a real model."""
    if not chunks:
        return []
    pairs = [(query, c.text) for c in chunks]
    scores = scorer(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
    return [c for c, _ in ranked[:top_k]]


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 5) -> list[Chunk]:
        return rerank_chunks(self.model.predict, query, chunks, top_k)
