"""
Build vector index from chunks.jsonl.
Reads chunks, embeds them, saves vectors.npy and chunks_meta.json.
Run: python scripts/build_index.py (from rag_service/, with active venv)
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedder import Embedder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")


def main():
    if not os.path.exists(CHUNKS_PATH):
        print(f"ERROR: {CHUNKS_PATH} not found. Run build_chunks.py first.")
        sys.exit(1)

    print("Loading chunks...")
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    meta = [{"chunk_id": c["chunk_id"], "source": c["source"], "metadata": c["metadata"]} for c in chunks]

    print("Initializing embedder (downloading model on first run)...")
    embedder = Embedder()
    print(f"Model dimension: {embedder.dimension}")

    print(f"Embedding {len(texts)} chunks...")
    start = time.time()
    vectors = embedder.embed(texts)
    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s ({len(texts) / elapsed:.1f} chunks/sec)")

    np.save(VECTORS_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved vectors: {VECTORS_PATH}  shape={vectors.shape}")
    print(f"Saved metadata: {META_PATH}  ({len(meta)} entries)")


if __name__ == "__main__":
    main()
