"""
Text embedding using sentence-transformers.
Model: intfloat/multilingual-e5-base (good for Russian text).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Encode texts into normalized embedding vectors."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", batch_size: int = 32):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension = self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into a (N, dim) numpy array.
        Vectors are L2-normalized so cosine similarity = dot product.
        Processes in batches of self.batch_size.
        """
        # multilingual-e5 expects "query: " or "passage: " prefix
        prefixed = [f"passage: {t}" for t in texts]

        vectors = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > self.batch_size,
            normalize_embeddings=True,
        )
        return np.array(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query. Uses "query: " prefix per e5 spec.
        Returns a 1D vector of shape (dim,).
        """
        vector = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
        return np.array(vector)
