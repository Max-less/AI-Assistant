"""
Aggregate every data/eval_results_*.json into before/after markdown tables (the
headline artifact for the defense), print them, and inject them into EVALUATION.md
between the <!-- EVAL_TABLE_START/END --> markers.

Two tables are produced:
  * Retrieval iterations  (--retrieval-only runs): recall@k focus.
  * Full-pipeline runs    (with answers): all metrics.

Run (from rag_service/): python scripts/compare_eval.py
"""

import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
EVAL_MD = os.path.join(ROOT, "EVALUATION.md")

START = "<!-- EVAL_TABLE_START -->"
END = "<!-- EVAL_TABLE_END -->"

RETRIEVAL_HEADERS = ["Label", "alpha", "rerank", "pool", "thr", "R@1", "R@3", "R@5"]
FULL_HEADERS = ["Label", "top_k", "alpha", "rerank", "thr", "prompt",
                "R@1", "R@3", "R@5", "KW", "Faith", "Refuse", "llm ms"]


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _onoff(x):
    return "on" if x else "off"


def load_runs():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "eval_results_*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        s = data.get("summary", {})
        rows.append({
            "timestamp": data.get("timestamp", ""),
            "label": data.get("label", cfg.get("label", "?")),
            "retrieval_only": cfg.get("retrieval_only", False),
            "top_k": cfg.get("top_k"),
            "alpha": cfg.get("alpha"),
            "reranker": cfg.get("reranker"),
            "rerank_pool": cfg.get("rerank_pool"),
            "thr": cfg.get("score_threshold"),
            "prompt": cfg.get("prompt_variant", "base"),
            "recall@1": s.get("recall@1"),
            "recall@3": s.get("recall@3"),
            "recall@5": s.get("recall@5"),
            "keyword_recall": s.get("keyword_recall"),
            "faithfulness": s.get("faithfulness"),
            "refusal_rate": s.get("refusal_rate"),
            "avg_llm_ms": s.get("avg_llm_ms"),
        })
    return rows


def _table(headers, body_rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend(body_rows)
    return "\n".join(lines)


def _retrieval_row(r):
    cells = [
        str(r["label"]),
        _fmt(r["alpha"], 2),
        _onoff(r["reranker"]),
        str(r["rerank_pool"]) if r["rerank_pool"] else "—",
        _fmt(r["thr"], 2),
        _fmt(r["recall@1"]), _fmt(r["recall@3"]), _fmt(r["recall@5"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _full_row(r):
    cells = [
        str(r["label"]),
        str(r["top_k"]),
        _fmt(r["alpha"], 2),
        _onoff(r["reranker"]),
        _fmt(r["thr"], 2),
        str(r["prompt"]),
        _fmt(r["recall@1"]), _fmt(r["recall@3"]), _fmt(r["recall@5"]),
        _fmt(r["keyword_recall"]), _fmt(r["faithfulness"]), _fmt(r["refusal_rate"]),
        _fmt(r["avg_llm_ms"], 0),
    ]
    return "| " + " | ".join(cells) + " |"


def build_tables(rows):
    retrieval = [r for r in rows if r["retrieval_only"]]
    full = [r for r in rows if not r["retrieval_only"]]

    parts = []
    if retrieval:
        parts.append("### Итерации ретривера (recall@k, `--retrieval-only`)\n")
        parts.append(_table(RETRIEVAL_HEADERS, [_retrieval_row(r) for r in retrieval]))
        parts.append("")
    if full:
        parts.append("### Полный пайплайн (качество ответа)\n")
        parts.append(_table(FULL_HEADERS, [_full_row(r) for r in full]))
        parts.append("")
    return "\n".join(parts).rstrip()


def inject(table):
    if not os.path.exists(EVAL_MD):
        print(f"(EVALUATION.md not found at {EVAL_MD}; printed table only)")
        return
    with open(EVAL_MD, "r", encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        print("(EVALUATION.md is missing EVAL_TABLE markers; printed table only)")
        return
    pre = text.split(START)[0]
    post = text.split(END)[1]
    new_text = f"{pre}{START}\n\n{table}\n\n{END}{post}"
    with open(EVAL_MD, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"Updated {os.path.relpath(EVAL_MD, ROOT)}")


def main():
    rows = load_runs()
    if not rows:
        print("No data/eval_results_*.json found. Run scripts/run_eval.py first.")
        return
    tables = build_tables(rows)
    print(f"\nFound {len(rows)} run(s):\n")
    print(tables)
    print()
    inject(tables)


if __name__ == "__main__":
    main()
