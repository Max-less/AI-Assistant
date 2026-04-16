"""
Build chunks from knowledge_base documents and save to data/chunks.jsonl.
Run: python scripts/build_chunks.py (from rag_service/, with active venv)
"""

import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from corpus import load_corpus
from chunker import chunk_document

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "chunks.jsonl")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading documents from {KB_DIR}...")
    docs = load_corpus(KB_DIR)
    print(f"Loaded {len(docs)} documents")

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, target_tokens=300, overlap=1)
        all_chunks.extend(chunks)
        print(f"  {doc.metadata.get('filename', '?'):40s} -> {len(chunks)} chunks")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            line = json.dumps(asdict(chunk), ensure_ascii=False)
            f.write(line + "\n")

    total_chars = sum(len(c.text) for c in all_chunks)
    avg_tokens = total_chars // 4 // max(len(all_chunks), 1)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Avg estimated tokens per chunk: {avg_tokens}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
