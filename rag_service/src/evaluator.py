"""
Offline-friendly evaluation harness for the RAG pipeline.

Three required metrics + one extra (honest refusal), each callable standalone as
``eval_*(questions, pipeline)`` or with shared, precomputed ``runs=`` to avoid
paying the LLM cost several times over:

- eval_retrieval        — recall@k: did the expected source land in the top-k?
- eval_answer_keywords  — share of expected keywords present in the answer.
- eval_faithfulness_llm — LLM-as-a-judge: does the answer follow from the context (0/1)?
- eval_honest_refusal   — on out-of-base questions, did the system honestly refuse?

The module is intentionally dependency-light (stdlib only). ``pipeline``, its
``retriever`` and the chunk objects are duck-typed, so importing this module does
NOT pull in GigaChat / sentence-transformers — unit tests can exercise the metric
logic with fakes and no network.

Question schema (tests/eval_questions.json), per item:
    {
      "id": "q01",
      "question": "...",
      "expected_source": "file.pdf" | ["a.pdf", "b.pdf"] | null,
      "expected_keywords": ["...", ...],
      "in_base": true | false
    }

`expected_source` is matched on basename (the pipeline stores full absolute paths
in ``chunk.source``) and may be a single name or a list of acceptable names.
"""

import os
import re
import statistics
import time

# Refusal markers used to detect an honest "I don't know". Kept in sync with
# rag_pipeline.NO_CONTEXT_ANSWER ("В базе знаний нет информации по этому вопросу.")
# but defined locally so the evaluator stays import-light.
REFUSAL_MARKERS = (
    "в базе знаний нет",
    "нет информации",
    "не содержится",
    "не содержит информации",
    "отсутствует информация",
    "нет ответа на этот вопрос",
    "не могу ответить",
    "не нашёл",
    "не найдено",
    "не знаю",
    "не располагаю",
    "недостаточно информации",
    "в предоставленных материалах",
)

_DIGIT_RE = re.compile(r"[01]")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _basename(path: str) -> str:
    """Filename without directories, normalising both / and \\ separators."""
    return os.path.basename(str(path).replace("\\", "/"))


def normalize_expected(expected) -> list[str]:
    """Coerce expected_source (str | list | None) to a list of basenames."""
    if expected is None:
        return []
    if isinstance(expected, str):
        expected = [expected]
    return [_basename(e) for e in expected]


def _kw_hit(answer: str, keyword: str) -> bool:
    """Case-insensitive substring match."""
    return keyword.casefold() in (answer or "").casefold()


def _kw_group_hit(answer: str, group) -> bool:
    """A keyword may be a single string or a list of synonyms (RU/EN variants,
    declensions). The group counts as found if ANY of its variants appears."""
    if isinstance(group, (list, tuple)):
        return any(_kw_hit(answer, k) for k in group)
    return _kw_hit(answer, group)


def _kw_label(group) -> str:
    """Readable label for a keyword group (its first/canonical variant)."""
    if isinstance(group, (list, tuple)):
        return group[0] if group else ""
    return group


def is_refusal(answer: str) -> bool:
    """True if the answer reads as an honest 'no information' refusal."""
    low = (answer or "").casefold()
    return any(marker in low for marker in REFUSAL_MARKERS)


def _retrieve_sources(pipeline, question: str, k: int) -> list[str]:
    """Top-k retrieved source basenames (ordered), [] on gate/empty."""
    chunks = pipeline.retriever.retrieve(question, top_k=k)
    return [_basename(c.source) for c in chunks]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _is_in_base(q: dict) -> bool:
    """In-base if flagged, or (fallback) if an expected_source is given."""
    if "in_base" in q:
        return bool(q["in_base"])
    return q.get("expected_source") is not None


# --------------------------------------------------------------------------- #
# Shared pipeline pass (run each question once: retrieve + answer)
# --------------------------------------------------------------------------- #
def collect_runs(questions: list[dict], pipeline, k_values=(1, 3, 5)) -> list[dict]:
    """Run the pipeline once per question and cache everything the metrics need.

    For each question we both retrieve (for recall + judge context) and call the
    real ``pipeline.answer`` (the production answer path). Retrieval is done at
    ``max(max(k_values), pipeline.top_k)`` so recall@k prefixes and the answer's
    context window are both available from a single retrieval.
    """
    answer_k = getattr(pipeline, "top_k", max(k_values))
    fetch_k = max(max(k_values), answer_k)
    runs = []

    for q in questions:
        question = q["question"]
        run = {
            "id": q.get("id"),
            "question": question,
            "in_base": _is_in_base(q),
            "expected_source": normalize_expected(q.get("expected_source")),
            "expected_keywords": list(q.get("expected_keywords") or []),
            "retrieved_sources": [],
            "context_text": "",
            "answer": "",
            "sources": [],
            "timings_ms": {},
            "error": None,
        }
        try:
            chunks = pipeline.retriever.retrieve(question, top_k=fetch_k)
            run["retrieved_sources"] = [_basename(c.source) for c in chunks]
            run["context_text"] = "\n\n".join(
                f"[{i + 1}] {c.text}" for i, c in enumerate(chunks[:answer_k])
            )

            result = pipeline.answer(question)
            run["answer"] = result.get("answer", "")
            run["sources"] = result.get("sources", [])
            run["timings_ms"] = result.get("timings_ms", {})
        except Exception as exc:  # keep a long eval going; record the failure
            run["error"] = f"{type(exc).__name__}: {exc}"

        runs.append(run)

    return runs


def _runs_by_id(runs: list[dict]) -> dict:
    return {r.get("id"): r for r in runs}


def _resolve_runs(questions, pipeline, runs, k_values):
    return runs if runs is not None else collect_runs(questions, pipeline, k_values)


# --------------------------------------------------------------------------- #
# Metric 1 — retrieval recall@k
# --------------------------------------------------------------------------- #
def eval_retrieval(questions, pipeline, k_values=(1, 3, 5), runs=None) -> dict:
    """Recall@k over in-base questions: is an expected source in the top-k?

    Standalone (runs=None) this only retrieves — it does NOT generate answers, so
    it is cheap to call on its own. Out-of-base questions are excluded from recall
    but reported via ``empty_retrieval_rate`` (how often the relevance gate fired).
    """
    k_values = tuple(sorted(set(k_values)))
    k_max = max(k_values)
    by_id = _runs_by_id(runs) if runs is not None else {}

    per_question = []
    hits = {k: [] for k in k_values}
    out_of_base_empty = []

    for q in questions:
        qid = q.get("id")
        expected = normalize_expected(q.get("expected_source"))

        if runs is not None:
            retrieved = by_id.get(qid, {}).get("retrieved_sources", [])
        else:
            retrieved = _retrieve_sources(pipeline, q["question"], k_max)

        if not _is_in_base(q):
            out_of_base_empty.append(1.0 if len(retrieved) == 0 else 0.0)
            per_question.append({
                "id": qid, "in_base": False,
                "retrieved_top": retrieved[:k_max], "empty": len(retrieved) == 0,
            })
            continue

        expected_cf = {e.casefold() for e in expected}
        q_hits = {}
        for k in k_values:
            top_cf = {s.casefold() for s in retrieved[:k]}
            hit = bool(expected_cf & top_cf)
            q_hits[k] = hit
            hits[k].append(1.0 if hit else 0.0)

        per_question.append({
            "id": qid, "in_base": True,
            "expected_source": expected,
            "retrieved_top": retrieved[:k_max],
            "hit_at_k": q_hits,
        })

    metrics = {f"recall@{k}": round(_mean(hits[k]), 4) for k in k_values}
    metrics["n_in_base"] = len(hits[k_max])
    metrics["out_of_base_empty_retrieval_rate"] = round(_mean(out_of_base_empty), 4)
    metrics["n_out_of_base"] = len(out_of_base_empty)
    metrics["per_question"] = per_question
    return metrics


# --------------------------------------------------------------------------- #
# Metric 2 — answer keyword coverage
# --------------------------------------------------------------------------- #
def eval_answer_keywords(questions, pipeline, runs=None) -> dict:
    """Share of expected keywords present in the answer (in-base questions).

    Reports a soft mean coverage (``keyword_recall``) plus strict/lenient rates.
    """
    runs = _resolve_runs(questions, pipeline, runs, (1, 3, 5))
    by_id = _runs_by_id(runs)

    per_question = []
    fractions, all_present, any_present = [], [], []

    for q in questions:
        if not _is_in_base(q):
            continue
        keywords = list(q.get("expected_keywords") or [])
        if not keywords:
            continue

        run = by_id.get(q.get("id"), {})
        answer = run.get("answer", "")

        found, missing = [], []
        for group in keywords:
            (found if _kw_group_hit(answer, group) else missing).append(_kw_label(group))
        frac = len(found) / len(keywords)
        fractions.append(frac)
        all_present.append(1.0 if not missing else 0.0)
        any_present.append(1.0 if found else 0.0)

        per_question.append({
            "id": q.get("id"),
            "found": found,
            "missing": missing,
            "fraction": round(frac, 4),
        })

    return {
        "keyword_recall": round(_mean(fractions), 4),
        "all_keywords_rate": round(_mean(all_present), 4),
        "any_keyword_rate": round(_mean(any_present), 4),
        "n": len(fractions),
        "per_question": per_question,
    }


# --------------------------------------------------------------------------- #
# Metric 3 — faithfulness (LLM-as-a-judge)
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM_PROMPT = (
    "Ты — строгий эксперт по проверке фактической обоснованности ответов. "
    "Тебе дают КОНТЕКСТ и ОТВЕТ. Реши, следует ли ОТВЕТ строго из КОНТЕКСТА, "
    "не добавляя фактов, которых в контексте нет."
)


def _build_judge_messages(context: str, answer: str) -> list[dict]:
    user = (
        f"КОНТЕКСТ:\n{context}\n\n"
        f"ОТВЕТ:\n{answer}\n\n"
        "Все ли утверждения в ОТВЕТЕ подтверждаются КОНТЕКСТОМ "
        "(нет ли добавленных фактов, которых нет в контексте)?\n"
        "Ответь СТРОГО одной цифрой без пояснений: "
        "1 — да, ответ полностью обоснован контекстом; "
        "0 — нет, есть утверждения не из контекста."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_verdict(raw: str) -> int | None:
    """First standalone 0/1 in the judge's reply, or None if unparseable."""
    if not raw:
        return None
    m = _DIGIT_RE.search(raw)
    return int(m.group(0)) if m else None


def eval_faithfulness_llm(questions, pipeline, judge_llm=None, runs=None) -> dict:
    """LLM-as-a-judge faithfulness over ALL questions.

    A separate judge prompt asks whether the answer follows from the retrieved
    context (1/0). Honest refusals (and empty context) short-circuit to faithful=1
    without spending a judge call — declining to answer never invents facts.
    The judge defaults to ``pipeline.llm`` unless a ``judge_llm`` is supplied.
    """
    runs = _resolve_runs(questions, pipeline, runs, (1, 3, 5))
    by_id = _runs_by_id(runs)
    judge = judge_llm if judge_llm is not None else getattr(pipeline, "llm", None)

    per_question = []
    verdicts = []

    for q in questions:
        run = by_id.get(q.get("id"), {})
        answer = run.get("answer", "")
        context = run.get("context_text", "")

        if run.get("error"):
            per_question.append({"id": q.get("id"), "faithful": None,
                                 "judged": False, "note": "pipeline_error"})
            continue

        if is_refusal(answer) or not context.strip():
            verdicts.append(1.0)
            per_question.append({"id": q.get("id"), "faithful": 1,
                                 "judged": False, "note": "refusal_or_no_context"})
            continue

        raw = judge.complete(_build_judge_messages(context, answer))
        verdict = _parse_verdict(raw)
        faithful = 1 if verdict == 1 else 0  # unparseable -> conservative 0
        verdicts.append(float(faithful))
        per_question.append({
            "id": q.get("id"), "faithful": faithful, "judged": True,
            "verdict_raw": (raw or "").strip()[:40],
        })

    return {
        "faithfulness": round(_mean(verdicts), 4),
        "n": len(verdicts),
        "per_question": per_question,
    }


# --------------------------------------------------------------------------- #
# Metric 4 — honest refusal on out-of-base questions
# --------------------------------------------------------------------------- #
def eval_honest_refusal(questions, pipeline, runs=None) -> dict:
    """Share of out-of-base questions the system correctly refused to answer."""
    runs = _resolve_runs(questions, pipeline, runs, (1, 3, 5))
    by_id = _runs_by_id(runs)

    per_question = []
    refusals = []

    for q in questions:
        if _is_in_base(q):
            continue
        run = by_id.get(q.get("id"), {})
        answer = run.get("answer", "")
        refused = is_refusal(answer)
        refusals.append(1.0 if refused else 0.0)
        per_question.append({
            "id": q.get("id"),
            "refused": refused,
            "answer_preview": (answer or "").strip()[:80],
        })

    return {
        "refusal_rate": round(_mean(refusals), 4),
        "n": len(refusals),
        "per_question": per_question,
    }


# --------------------------------------------------------------------------- #
# Convenience: run everything off a single pipeline pass
# --------------------------------------------------------------------------- #
def run_all(questions, pipeline, k_values=(1, 3, 5), judge_llm=None) -> dict:
    """Single pipeline pass, then all four metrics. Returns merged results."""
    started = time.perf_counter()
    runs = collect_runs(questions, pipeline, k_values)

    retrieval = eval_retrieval(questions, pipeline, k_values, runs=runs)
    keywords = eval_answer_keywords(questions, pipeline, runs=runs)
    faithfulness = eval_faithfulness_llm(questions, pipeline, judge_llm, runs=runs)
    honest_refusal = eval_honest_refusal(questions, pipeline, runs=runs)

    llm_times = [r["timings_ms"].get("llm", 0) for r in runs if r.get("timings_ms")]
    errors = [r for r in runs if r.get("error")]

    return {
        "summary": {
            "n_questions": len(questions),
            "recall@1": retrieval.get("recall@1"),
            "recall@3": retrieval.get("recall@3"),
            "recall@5": retrieval.get("recall@5"),
            "keyword_recall": keywords.get("keyword_recall"),
            "all_keywords_rate": keywords.get("all_keywords_rate"),
            "faithfulness": faithfulness.get("faithfulness"),
            "refusal_rate": honest_refusal.get("refusal_rate"),
            "avg_llm_ms": round(_mean([float(t) for t in llm_times]), 1),
            "n_errors": len(errors),
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
        "retrieval": retrieval,
        "keywords": keywords,
        "faithfulness": faithfulness,
        "honest_refusal": honest_refusal,
        "runs": runs,
    }
