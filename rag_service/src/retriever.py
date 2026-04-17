"""
Retriever: embeds a question and fetches top-k chunks from the vector store.
Optional QueryExpander splits compound questions into sub-queries
and merges retrieval results by best per-chunk score.
"""

from chunker import Chunk
from embedder import Embedder
from query_expander import QueryExpander
from vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        expander: QueryExpander | None = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.expander = expander

    def retrieve(self, question: str, top_k: int = 5) -> list[Chunk]:
        if self.expander is None:
            query_vec = self.embedder.embed_query(question)
            results = self.vector_store.search(query_vec, top_k=top_k)
            return [chunk for chunk, _score in results]

        sub_queries = self.expander.expand(question)

        best_by_id: dict[str, tuple[Chunk, float]] = {}
        for sub in sub_queries:
            query_vec = self.embedder.embed_query(sub)
            results = self.vector_store.search(query_vec, top_k=top_k)
            for chunk, score in results:
                existing = best_by_id.get(chunk.chunk_id)
                if existing is None or score > existing[1]:
                    best_by_id[chunk.chunk_id] = (chunk, score)

        ranked = sorted(best_by_id.values(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _score in ranked[:top_k]]
