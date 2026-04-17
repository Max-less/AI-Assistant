import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedder import Embedder
from vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")


def test_search_returns_relevant_chunks():
    """Search for a known topic and print top-5 results for manual inspection."""
    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)
    embedder = Embedder()

    queries = [
        "Какие роли существуют в Scrum?",
        "Что такое DevOps?",
        "Как составить техническое задание?",
    ]

    for query in queries:
        query_vec = embedder.embed_query(query)
        results = store.search(query_vec, top_k=5)

        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        for i, (chunk, score) in enumerate(results):
            print(f"\n  #{i+1}  score={score:.4f}  [{chunk.chunk_id}]")
            preview = chunk.text[:200].replace("\n", " ")
            print(f"  {preview.encode('ascii', errors='replace').decode()}...")

        # Top result should have a reasonable score
        assert results[0][1] > 0.5, f"Top score too low: {results[0][1]:.4f}"


def test_search_top_k_ordering():
    """Verify results are sorted by descending score."""
    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)
    embedder = Embedder()

    query_vec = embedder.embed_query("Agile методология")
    results = store.search(query_vec, top_k=10)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert len(results) == 10
