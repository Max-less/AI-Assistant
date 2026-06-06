"""
Run the evaluation suite against the RAG pipeline with a given configuration,
print a metrics table, and save data/eval_results_<timestamp>.json.

Each run varies runtime knobs only (no re-indexing). Examples (from rag_service/,
active venv):

  # baseline: dense-only, no reranker, no expander
  python scripts/run_eval.py --label baseline --alpha 1.0 --no-reranker --no-expander

  # hybrid retrieval (add BM25)
  python scripts/run_eval.py --label hybrid --alpha 0.5 --no-reranker --no-expander

  # + cross-encoder reranker
  python scripts/run_eval.py --label rerank --alpha 0.5 --reranker --no-expander

  # tune top_k / threshold
  python scripts/run_eval.py --label topk8 --alpha 0.5 --reranker --top-k 8 --score-threshold 0.4

  # stricter prompt variant
  python scripts/run_eval.py --label strict --alpha 0.5 --reranker --top-k 8 --prompt-variant strict

  # quick smoke test on a few questions
  python scripts/run_eval.py --label smoke --limit 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

import evaluator
from bm25 import BM25
from embedder import Embedder
from llm_client import LLMClient
from prompt_builder import PROMPT_VARIANTS
from query_expander import QueryExpander
from rag_pipeline import RAGPipeline
from reranker import Reranker
from retriever import Retriever
from vector_store import VectorStore

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
DEFAULT_QUESTIONS = os.path.join(ROOT, "tests", "eval_questions.json")


def parse_args():
    p = argparse.ArgumentParser(description="Run RAG evaluation with a given config.")
    p.add_argument("--label", default="run", help="human label for this run (table row)")
    p.add_argument("--questions", default=DEFAULT_QUESTIONS, help="path to eval_questions.json")
    p.add_argument("--limit", type=int, default=0, help="evaluate only first N questions (0 = all)")
    p.add_argument("--top-k", type=int, default=5, help="chunks fed to the LLM")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="dense vs BM25 fusion weight; 1.0 = dense-only (BM25 disabled)")
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="dense top-1 cosine relevance gate (out-of-scope -> no answer)")
    p.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=True,
                   help="enable cross-encoder reranker")
    p.add_argument("--rerank-pool", type=int, default=20,
                   help="candidates pulled from fusion before reranking (smaller = faster on CPU)")
    p.add_argument("--expander", action=argparse.BooleanOptionalAction, default=False,
                   help="enable LLM query expansion (adds nondeterminism + LLM calls)")
    p.add_argument("--prompt-variant", choices=sorted(PROMPT_VARIANTS), default="base")
    p.add_argument("--retrieval-only", action="store_true",
                   help="only measure recall@k (no answers/judge); needs no GigaChat key")
    return p.parse_args()


def build_pipeline(args, auth_key):
    print("Loading vector store...")
    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)

    print("Loading embedder...")
    embedder = Embedder()

    use_bm25 = args.alpha < 1.0
    bm25 = None
    if use_bm25:
        print("Fitting BM25...")
        bm25 = BM25().fit([c.text for c in store.chunks])

    reranker = None
    if args.reranker:
        print("Loading reranker (first run downloads the model)...")
        reranker = Reranker()

    llm = LLMClient(auth_key) if auth_key else None
    # Query expansion needs the LLM, so it is forced off in retrieval-only mode.
    expander = QueryExpander(llm) if (args.expander and not args.retrieval_only and llm) else None

    retriever = Retriever(
        store,
        embedder,
        bm25=bm25,
        expander=expander,
        alpha=args.alpha,
        score_threshold=args.score_threshold,
        reranker=reranker,
        rerank_pool=args.rerank_pool,
    )

    # base variant -> None so the pipeline uses its built-in default prompt.
    system_prompt = None if args.prompt_variant == "base" else PROMPT_VARIANTS[args.prompt_variant]
    pipeline = RAGPipeline(retriever, llm, top_k=args.top_k, system_prompt=system_prompt)
    return pipeline


def config_dict(args):
    return {
        "label": args.label,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "bm25": args.alpha < 1.0,
        "score_threshold": args.score_threshold,
        "reranker": bool(args.reranker),
        "rerank_pool": args.rerank_pool if args.reranker else None,
        "expander": bool(args.expander) and not args.retrieval_only,
        "prompt_variant": args.prompt_variant,
        "retrieval_only": bool(args.retrieval_only),
    }


def trim_runs(runs):
    """Drop the bulky context_text before persisting; keep the rest for audit."""
    keep = ("id", "in_base", "expected_source", "expected_keywords",
            "retrieved_sources", "answer", "sources", "timings_ms", "error")
    return [{k: r.get(k) for k in keep} for r in runs]


def print_retrieval_table(cfg, questions, retrieval, wall_seconds):
    n_in = retrieval.get("n_in_base", 0)
    n_out = retrieval.get("n_out_of_base", 0)
    print("\n" + "=" * 64)
    print(f"Eval run: {cfg['label']}  (retrieval-only)")
    print("=" * 64)
    print(f"Config: top_k={cfg['top_k']} alpha={cfg['alpha']} bm25={cfg['bm25']} "
          f"reranker={cfg['reranker']} thr={cfg['score_threshold']}")
    print(f"Questions: {len(questions)} (in-base {n_in}, out-of-base {n_out})\n")
    print(f"  Retrieval      recall@1={retrieval['recall@1']:.3f}  "
          f"recall@3={retrieval['recall@3']:.3f}  recall@5={retrieval['recall@5']:.3f}  (n={n_in})")
    print(f"  Out-of-base    empty_retrieval_rate={retrieval['out_of_base_empty_retrieval_rate']:.3f}  (n={n_out})")
    print(f"  Timing         wall={wall_seconds:.1f}s")


def print_table(cfg, questions, results):
    s = results["summary"]
    n_in = sum(1 for q in questions if evaluator._is_in_base(q))
    n_out = len(questions) - n_in
    judged = sum(1 for x in results["faithfulness"]["per_question"] if x.get("judged"))

    print("\n" + "=" * 64)
    print(f"Eval run: {cfg['label']}")
    print("=" * 64)
    print(f"Config: top_k={cfg['top_k']} alpha={cfg['alpha']} bm25={cfg['bm25']} "
          f"reranker={cfg['reranker']} expander={cfg['expander']} "
          f"thr={cfg['score_threshold']} prompt={cfg['prompt_variant']}")
    print(f"Questions: {len(questions)} (in-base {n_in}, out-of-base {n_out})\n")

    print(f"  Retrieval      recall@1={s['recall@1']:.3f}  recall@3={s['recall@3']:.3f}  "
          f"recall@5={s['recall@5']:.3f}  (n={n_in})")
    print(f"  Keywords       keyword_recall={s['keyword_recall']:.3f}  "
          f"all={results['keywords']['all_keywords_rate']:.3f}  "
          f"any={results['keywords']['any_keyword_rate']:.3f}")
    print(f"  Faithfulness   {s['faithfulness']:.3f}  (judged {judged}/{len(questions)})")
    print(f"  Honest refusal {s['refusal_rate']:.3f}  (n={n_out})")
    print(f"  Timing         avg_llm={s['avg_llm_ms']:.0f}ms  wall={s['wall_seconds']:.1f}s  "
          f"errors={s['n_errors']}")
    if s["n_errors"]:
        for r in results["runs"]:
            if r.get("error"):
                print(f"    ! {r.get('id')}: {r['error']}")


def main():
    args = parse_args()

    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not args.retrieval_only and (not auth_key or auth_key == "your-authorization-key-here"):
        print("ERROR: Set GIGACHAT_AUTH_KEY in .env file (or use --retrieval-only)")
        sys.exit(1)

    for path in (VECTORS_PATH, META_PATH, CHUNKS_PATH):
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run build_chunks.py then build_index.py first.")
            sys.exit(1)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    pipeline = build_pipeline(args, auth_key)
    cfg = config_dict(args)

    print(f"\nEvaluating {len(questions)} questions (label='{args.label}')...")

    if args.retrieval_only:
        started = time.perf_counter()
        retrieval = evaluator.eval_retrieval(questions, pipeline)
        wall = time.perf_counter() - started
        print_retrieval_table(cfg, questions, retrieval, wall)
        summary = {
            "n_questions": len(questions),
            "recall@1": retrieval.get("recall@1"),
            "recall@3": retrieval.get("recall@3"),
            "recall@5": retrieval.get("recall@5"),
            "keyword_recall": None, "all_keywords_rate": None,
            "faithfulness": None, "refusal_rate": None,
            "avg_llm_ms": None, "n_errors": 0,
            "wall_seconds": round(wall, 1),
        }
        metrics = {"retrieval": retrieval, "keywords": None,
                   "faithfulness": None, "honest_refusal": None}
        runs_out = []
    else:
        results = evaluator.run_all(questions, pipeline)
        print_table(cfg, questions, results)
        summary = results["summary"]
        metrics = {
            "retrieval": results["retrieval"],
            "keywords": results["keywords"],
            "faithfulness": results["faithfulness"],
            "honest_refusal": results["honest_refusal"],
        }
        runs_out = trim_runs(results["runs"])

    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"eval_results_{timestamp}.json")
    payload = {
        "timestamp": timestamp,
        "label": args.label,
        "config": cfg,
        "questions_file": os.path.basename(args.questions),
        "n_questions": len(questions),
        "summary": summary,
        "metrics": metrics,
        "runs": runs_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {os.path.relpath(out_path, ROOT)}")

    if pipeline.llm is not None and hasattr(pipeline.llm, "close"):
        pipeline.llm.close()


if __name__ == "__main__":
    main()
