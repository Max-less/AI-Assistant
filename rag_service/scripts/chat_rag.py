"""
Interactive RAG chat — keeps dialog history and retrieves fresh context per turn.
Run: python scripts/chat_rag.py (from rag_service/, active venv)
Exit: quit, exit, or Ctrl+C.
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


def dedup_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main():
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

    history: list[dict] = []

    print("\nRAG chat (quit/exit to leave)")
    print("-" * 40)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            result = pipeline.answer(user_input, history=history)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"\nBot: {result['answer']}")

        sources = dedup_preserve_order(result["sources"])
        if sources:
            print("\nSources:")
            for src in sources:
                print(f"  - {src}")
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
