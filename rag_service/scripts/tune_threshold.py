"""
Tune the relevance gate (RAG_SCORE_THRESHOLD) and top_k from a SINGLE retrieval
pass — no GigaChat, no repeated reranking.

The gate fires on the dense top-1 cosine (before reranking), and reranking does
not depend on the threshold. So we run the expensive pass once per question,
capturing (dense_top1_cosine, reranked_top5_sources), then evaluate every
(threshold, k) combination analytically.

Run (from rag_service/, venv active):
    python scripts/tune_threshold.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

import evaluator
from bm25 import BM25
from embedder import Embedder
from reranker import Reranker
from retriever import Retriever
from vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
QUESTIONS = os.path.join(ROOT, "tests", "eval_questions.json")

THRESHOLDS = [0.70, 0.78, 0.79, 0.80, 0.81, 0.815, 0.82, 0.835]
K_VALUES = [1, 3, 4, 5]


def main() -> None:
    with open(QUESTIONS, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading store / embedder / bm25 / reranker...")
    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)
    embedder = Embedder()
    bm25 = BM25().fit([c.text for c in store.chunks])
    reranker = Reranker()

    # No gate here (threshold=None): we capture the would-be results and apply
    # thresholds ourselves afterwards. Mirrors prod fusion/rerank settings.
    retriever = Retriever(
        store, embedder, bm25=bm25, alpha=0.5,
        score_threshold=None, reranker=reranker, rerank_pool=10,
    )

    rows = []
    print(f"Single retrieval pass over {len(questions)} questions...")
    for q in questions:
        question = q["question"]
        vec = embedder.embed_query(question)
        dense = store.search(vec, top_k=1)
        top1 = dense[0][1] if dense else 0.0
        reranked = [evaluator._basename(c.source) for c in retriever.retrieve(question, top_k=max(K_VALUES))]
        rows.append({
            "in_base": evaluator._is_in_base(q),
            "expected": {e.casefold() for e in evaluator.normalize_expected(q.get("expected_source"))},
            "top1": float(top1),
            "reranked": [s.casefold() for s in reranked],
        })

    in_base = [r for r in rows if r["in_base"]]
    out_base = [r for r in rows if not r["in_base"]]

    print(f"\nin-base={len(in_base)}  out-of-base={len(out_base)}\n")
    header = "thr   " + "  ".join(f"recall@{k}" for k in K_VALUES) + "   oob_gated  in_base_gated"
    print(header)
    print("-" * len(header))

    for thr in THRESHOLDS:
        recalls = []
        for k in K_VALUES:
            hits = 0
            for r in in_base:
                gated = r["top1"] < thr
                if not gated and (r["expected"] & set(r["reranked"][:k])):
                    hits += 1
            recalls.append(hits / len(in_base) if in_base else 0.0)

        oob_gated = sum(1 for r in out_base if r["top1"] < thr) / len(out_base) if out_base else 0.0
        in_gated = sum(1 for r in in_base if r["top1"] < thr) / len(in_base) if in_base else 0.0

        cells = "  ".join(f"{v:.3f}   " for v in recalls)
        print(f"{thr:.2f}  {cells}  {oob_gated:.3f}      {in_gated:.3f}")

    print("\nLegend: oob_gated = out-of-base questions correctly gated (higher=better); "
          "in_base_gated = in-base questions wrongly gated (lower=better).")
    print("Per-question dense top-1 cosine:")
    for r, q in zip(rows, questions):
        tag = "in " if r["in_base"] else "OUT"
        print(f"  [{tag}] top1={r['top1']:.3f}  {q.get('id')}  {q['question'][:50]}")


if __name__ == "__main__":
    main()
