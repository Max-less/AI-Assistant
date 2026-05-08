"""
Retriever: fetches top-k chunks for a question.
- Dense semantic search via VectorStore.
- Optional BM25 lexical search fused with dense via weighted Reciprocal Rank Fusion:
    score(d) = alpha * 1/(k + rank_dense) + (1 - alpha) * 1/(k + rank_bm25)
  alpha=1.0 → dense only; alpha=0.0 → BM25 only; alpha=0.5 → balanced.
- Optional QueryExpander splits compound questions into sub-queries; results across
  sub-queries are merged by best per-chunk score.
"""

from bm25 import BM25
from chunker import Chunk
from embedder import Embedder
from query_expander import QueryExpander
from vector_store import VectorStore


def fuse_rrf(
    dense_ids: list[str],
    bm25_ids: list[str],
    alpha: float = 0.5,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion of two ranked id lists.
    Returns (id, score) pairs sorted by descending fused score."""
    fused: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + alpha / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + (1.0 - alpha) / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        bm25: BM25 | None = None,
        expander: QueryExpander | None = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
        pool_size: int = 50,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25
        self.expander = expander
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.pool_size = pool_size

    def _search_one(self, query: str) -> list[tuple[Chunk, float]]:
        """Run dense + (optional) BM25 for a single query, fuse via RRF.
        Returns chunks ranked by fused score."""
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_vec, top_k=self.pool_size)

        if self.bm25 is None:
            return list(dense_results)

        bm25_results = self.bm25.search(query, top_k=self.pool_size)

        chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c, _ in dense_results}
        for doc_idx, _ in bm25_results:
            chunk = self.vector_store.chunks[doc_idx]
            chunks_by_id[chunk.chunk_id] = chunk

        dense_ids = [c.chunk_id for c, _ in dense_results]
        bm25_ids = [self.vector_store.chunks[i].chunk_id for i, _ in bm25_results]

        fused = fuse_rrf(dense_ids, bm25_ids, alpha=self.alpha, k=self.rrf_k)
        return [(chunks_by_id[doc_id], score) for doc_id, score in fused]

    def retrieve(self, question: str, top_k: int = 5) -> list[Chunk]:
        if self.expander is None:
            return [chunk for chunk, _ in self._search_one(question)[:top_k]]

        sub_queries = self.expander.expand(question)
        best_by_id: dict[str, tuple[Chunk, float]] = {}
        for sub in sub_queries:
            for chunk, score in self._search_one(sub):
                existing = best_by_id.get(chunk.chunk_id)
                if existing is None or score > existing[1]:
                    best_by_id[chunk.chunk_id] = (chunk, score)

        ranked = sorted(best_by_id.values(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_k]]
