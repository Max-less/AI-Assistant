"""
In-memory vector store for semantic search.
Uses dot product on L2-normalized vectors (= cosine similarity).
"""

import json
import numpy as np
from chunker import Chunk


class VectorStore:
    """Stores chunk vectors and performs top-k nearest neighbor search."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        self.vectors = vectors
        self.chunks = chunks

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """
        Find top_k most similar chunks to query_vec.
        Returns list of (Chunk, score) sorted by descending score.
        """
        scores = self.vectors @ query_vec
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    @classmethod
    def load(cls, vectors_path: str, meta_path: str) -> "VectorStore":
        """
        Load from vectors.npy and chunks_meta.json produced by build_index.py.
        """
        vectors = np.load(vectors_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        chunks = [
            Chunk(
                text="",  # text loaded lazily from chunks.jsonl if needed
                source=m["source"],
                chunk_id=m["chunk_id"],
                metadata=m["metadata"],
            )
            for m in meta_list
        ]

        return cls(vectors, chunks)

    @classmethod
    def load_with_texts(cls, vectors_path: str, meta_path: str, chunks_path: str) -> "VectorStore":
        """
        Load vectors + full chunk texts (for displaying search results).
        """
        vectors = np.load(vectors_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = [json.loads(line) for line in f]

        chunks = [
            Chunk(
                text=c["text"],
                source=c["source"],
                chunk_id=c["chunk_id"],
                metadata=c["metadata"],
            )
            for c in chunks_data
        ]

        return cls(vectors, chunks)
