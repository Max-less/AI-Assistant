import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedder import Embedder

embedder = None


def get_embedder():
    global embedder
    if embedder is None:
        embedder = Embedder()
    return embedder


def test_embed_returns_correct_shape():
    emb = get_embedder()
    texts = ["Hello world", "Test sentence"]
    vectors = emb.embed(texts)
    assert vectors.shape == (2, emb.dimension)


def test_vectors_are_normalized():
    emb = get_embedder()
    vectors = emb.embed(["Some text here"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_similar_texts_have_higher_similarity():
    emb = get_embedder()

    similar_a = "Что такое Scrum и как он работает?"
    similar_b = "Расскажи про методологию Scrum"
    different = "Рецепт борща с говядиной"

    vectors = emb.embed([similar_a, similar_b, different])

    sim_ab = np.dot(vectors[0], vectors[1])
    sim_ac = np.dot(vectors[0], vectors[2])

    assert sim_ab > sim_ac, (
        f"Similar texts should score higher: sim(a,b)={sim_ab:.3f} vs sim(a,c)={sim_ac:.3f}"
    )


def test_embed_query_returns_1d():
    emb = get_embedder()
    vector = emb.embed_query("test query")
    assert vector.ndim == 1
    assert vector.shape == (emb.dimension,)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-5
