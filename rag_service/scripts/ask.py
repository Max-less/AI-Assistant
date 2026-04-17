"""
Ask a single question against the RAG pipeline.
Run: python scripts/ask.py "Что такое Scrum?" (from rag_service/, active venv)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

from embedder import Embedder
from llm_client import LLMClient
from query_expander import QueryExpander
from rag_pipeline import RAGPipeline
from retriever import Retriever
from vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key or auth_key == "your-authorization-key-here":
        print("ERROR: Set GIGACHAT_AUTH_KEY in .env file")
        sys.exit(1)

    for path in (VECTORS_PATH, META_PATH, CHUNKS_PATH):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run build_chunks.py then build_index.py first.")
            sys.exit(1)

    print("Loading vector store...")
    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)

    print("Loading embedder...")
    embedder = Embedder()

    llm = LLMClient(auth_key)
    expander = QueryExpander(llm)
    retriever = Retriever(store, embedder, expander=expander)
    pipeline = RAGPipeline(retriever, llm)

    print(f"\nQuestion: {question}")
    print("-" * 70)
    result = pipeline.answer(question)

    print(f"\nAnswer:\n{result['answer']}")

    seen = set()
    unique_sources = []
    for src in result["sources"]:
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)

    print("\nSources:")
    for src in unique_sources:
        print(f"  - {src}")


if __name__ == "__main__":
    main()
