import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunker import Chunk
from reranker import rerank_chunks


def _chunk(idx: int, text: str = None) -> Chunk:
    return Chunk(text=text or f"text {idx}", source="src", chunk_id=f"c::{idx}")


def test_rerank_empty_returns_empty():
    assert rerank_chunks(lambda pairs: [], "q", [], top_k=5) == []


def test_rerank_reorders_by_descending_score():
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    # scorer returns scores aligned with input pairs order: chunk0=0.1, chunk1=0.9, chunk2=0.5
    scorer = lambda pairs: [0.1, 0.9, 0.5]
    ranked = rerank_chunks(scorer, "q", chunks, top_k=3)
    assert [c.chunk_id for c in ranked] == ["c::1", "c::2", "c::0"]


def test_rerank_slices_to_top_k():
    chunks = [_chunk(i) for i in range(5)]
    scorer = lambda pairs: [0.5, 0.9, 0.1, 0.7, 0.3]
    ranked = rerank_chunks(scorer, "q", chunks, top_k=2)
    assert [c.chunk_id for c in ranked] == ["c::1", "c::3"]


def test_rerank_passes_query_chunk_text_pairs_to_scorer():
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]
    seen_pairs = []

    def scorer(pairs):
        seen_pairs.extend(pairs)
        return [0.0] * len(pairs)

    rerank_chunks(scorer, "the question", chunks, top_k=2)
    assert seen_pairs == [("the question", "alpha"), ("the question", "beta")]


def test_rerank_top_k_larger_than_chunks():
    chunks = [_chunk(0), _chunk(1)]
    scorer = lambda pairs: [0.2, 0.8]
    ranked = rerank_chunks(scorer, "q", chunks, top_k=10)
    assert [c.chunk_id for c in ranked] == ["c::1", "c::0"]
