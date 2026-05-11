import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunker import Chunk
from retriever import Retriever, fuse_rrf


class FakeEmbedder:
    def embed_query(self, query):
        return None  # FakeVectorStore.search ignores the vector


class FakeVectorStore:
    def __init__(self, results):
        self.results = list(results)
        self.chunks = [c for c, _ in self.results]

    def search(self, query_vec, top_k):
        return self.results[:top_k]


def _chunk(idx: int) -> Chunk:
    return Chunk(text=f"text {idx}", source="src", chunk_id=f"c::{idx}")


def test_fuse_doc_in_both_lists_sums_weights():
    fused = fuse_rrf(["a", "b"], ["a", "c"], alpha=0.5, k=60)
    scores = dict(fused)
    # "a" is rank 1 in both → 0.5/61 + 0.5/61
    assert scores["a"] == 1.0 / 61
    # "b" only in dense at rank 2 → 0.5/62
    assert scores["b"] == 0.5 / 62
    # "c" only in bm25 at rank 2 → 0.5/62
    assert scores["c"] == 0.5 / 62


def test_fuse_alpha_one_is_dense_only():
    """alpha=1.0 → BM25 ranking ignored, dense ranking preserved."""
    fused = fuse_rrf(["a", "b", "c"], ["c", "b", "a"], alpha=1.0, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b", "c"]
    # bm25-only doc would get score 0 — but here all docs appear in dense
    scores = dict(fused)
    assert scores["a"] == 1.0 / 61


def test_fuse_alpha_zero_is_bm25_only():
    """alpha=0.0 → dense ranking ignored, bm25 ranking preserved."""
    fused = fuse_rrf(["a", "b", "c"], ["c", "b", "a"], alpha=0.0, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["c", "b", "a"]


def test_fuse_output_sorted_descending():
    fused = fuse_rrf(["a", "b", "c"], ["b", "c", "a"], alpha=0.5, k=60)
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_fuse_doc_only_in_one_list():
    fused = fuse_rrf(["a"], ["b"], alpha=0.5, k=60)
    scores = dict(fused)
    assert set(scores.keys()) == {"a", "b"}
    # equal rank → equal score
    assert scores["a"] == scores["b"]


def test_fuse_empty_lists():
    assert fuse_rrf([], [], alpha=0.5, k=60) == []


def test_fuse_one_empty_list():
    fused = fuse_rrf(["a", "b"], [], alpha=0.5, k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b"]
    # only dense contribution, scaled by alpha
    scores = dict(fused)
    assert scores["a"] == 0.5 / 61


def test_threshold_returns_empty_when_top1_below():
    store = FakeVectorStore([(_chunk(0), 0.3), (_chunk(1), 0.2)])
    retriever = Retriever(store, FakeEmbedder(), score_threshold=0.5)
    assert retriever.retrieve("anything", top_k=2) == []


def test_threshold_returns_chunks_when_top1_above():
    store = FakeVectorStore([(_chunk(0), 0.7), (_chunk(1), 0.3)])
    retriever = Retriever(store, FakeEmbedder(), score_threshold=0.5)
    chunks = retriever.retrieve("anything", top_k=2)
    assert [c.chunk_id for c in chunks] == ["c::0", "c::1"]


def test_no_threshold_passes_through_low_scores():
    store = FakeVectorStore([(_chunk(0), 0.1)])
    retriever = Retriever(store, FakeEmbedder())  # threshold disabled
    chunks = retriever.retrieve("anything", top_k=1)
    assert [c.chunk_id for c in chunks] == ["c::0"]


def test_threshold_empty_store_returns_empty():
    store = FakeVectorStore([])
    retriever = Retriever(store, FakeEmbedder(), score_threshold=0.5)
    assert retriever.retrieve("anything", top_k=5) == []


class FakeReranker:
    def __init__(self):
        self.calls = []  # list of (query, candidate_chunk_ids, top_k)

    def rerank(self, query, chunks, top_k=5):
        self.calls.append((query, [c.chunk_id for c in chunks], top_k))
        # reverse the candidate order to prove rerank actually ran
        return list(reversed(chunks))[:top_k]


def test_reranker_receives_full_pool_returns_top_k():
    # 25 candidates above threshold; pool=20 → reranker sees 20 → returns top 5
    results = [(_chunk(i), 0.9 - i * 0.01) for i in range(25)]
    store = FakeVectorStore(results)
    reranker = FakeReranker()
    retriever = Retriever(
        store,
        FakeEmbedder(),
        reranker=reranker,
        rerank_pool=20,
    )

    out = retriever.retrieve("question", top_k=5)
    assert len(reranker.calls) == 1
    query, candidate_ids, top_k_arg = reranker.calls[0]
    assert query == "question"
    assert len(candidate_ids) == 20  # pool size
    assert top_k_arg == 5
    assert len(out) == 5


def test_reranker_not_called_when_threshold_blocks():
    store = FakeVectorStore([(_chunk(0), 0.2)])
    reranker = FakeReranker()
    retriever = Retriever(
        store,
        FakeEmbedder(),
        reranker=reranker,
        score_threshold=0.5,
    )
    assert retriever.retrieve("question", top_k=5) == []
    assert reranker.calls == []  # no point reranking nothing


def test_no_reranker_uses_top_k_pool():
    results = [(_chunk(i), 0.9 - i * 0.01) for i in range(25)]
    store = FakeVectorStore(results)
    retriever = Retriever(store, FakeEmbedder())  # no reranker
    out = retriever.retrieve("question", top_k=5)
    assert [c.chunk_id for c in out] == ["c::0", "c::1", "c::2", "c::3", "c::4"]
