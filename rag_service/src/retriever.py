"""
Retriever: fetches top-k chunks for a question.
- Dense semantic search via VectorStore.
- Optional BM25 lexical search fused with dense via weighted Reciprocal Rank Fusion:
    score(d) = alpha * 1/(k + rank_dense) + (1 - alpha) * 1/(k + rank_bm25)
  alpha=1.0 → dense only; alpha=0.0 → BM25 only; alpha=0.5 → balanced.
- Optional QueryExpander splits compound questions into sub-queries; results across
  sub-queries are merged by best per-chunk score.
- Optional score_threshold acts as a relevance gate on the dense top-1 cosine
  (RRF scores are not on an absolute scale; dense cosine on e5-base is). When the
  best dense match is below threshold, the sub-query contributes no chunks — if
  no sub-query passes, retrieve() returns []. Pipeline interprets [] as
  "база знаний не покрывает вопрос".
- Optional cross-encoder Reranker as the final stage: when set, retrieve() pulls
  rerank_pool candidates from fusion and rescores them jointly against the
  original question, returning top_k.
"""

from bm25 import BM25
from chunker import Chunk
from embedder import Embedder
from query_expander import QueryExpander
from reranker import Reranker
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
        score_threshold: float | None = None,
        reranker: Reranker | None = None,
        rerank_pool: int = 20,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = bm25
        self.expander = expander
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.score_threshold = score_threshold
        self.reranker = reranker
        self.rerank_pool = rerank_pool

    def _search_one(self, query: str) -> list[tuple[Chunk, float]]:
        """Run dense + (optional) BM25 for a single query, fuse via RRF.
        Returns chunks ranked by fused score, or [] if dense top-1 is below
        score_threshold (relevance gate)."""
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_vec, top_k=self.pool_size)

        if (
            self.score_threshold is not None
            and (not dense_results or dense_results[0][1] < self.score_threshold)
        ):
            return []

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
        # Pull a wider pool when reranking, then let the cross-encoder pick top_k.
        candidate_count = self.rerank_pool if self.reranker is not None else top_k

        if self.expander is None:
            candidates = [c for c, _ in self._search_one(question)[:candidate_count]]
        else:
            sub_queries = self.expander.expand(question)
            best_by_id: dict[str, tuple[Chunk, float]] = {}
            for sub in sub_queries:
                for chunk, score in self._search_one(sub):
                    existing = best_by_id.get(chunk.chunk_id)
                    if existing is None or score > existing[1]:
                        best_by_id[chunk.chunk_id] = (chunk, score)
            ranked = sorted(best_by_id.values(), key=lambda x: x[1], reverse=True)
            candidates = [chunk for chunk, _ in ranked[:candidate_count]]

        if self.reranker is None or not candidates:
            return candidates[:top_k]
        return self.reranker.rerank(question, candidates, top_k=top_k)
