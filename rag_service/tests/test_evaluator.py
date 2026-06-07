"""
Offline tests for the evaluation harness — no GigaChat, no model downloads.
The pipeline, retriever and LLM judge are all faked, so this exercises pure
metric logic (basename matching, recall@k slicing, keyword coverage, refusal
detection, faithfulness parsing/short-circuit, error handling).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import evaluator


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeChunk:
    def __init__(self, source, text="some context text"):
        self.source = source
        self.text = text


class FakeRetriever:
    """Returns a preset, pre-ordered chunk list per question, sliced to top_k."""
    def __init__(self, per_question, raises_for=()):
        self.per_question = per_question
        self.raises_for = set(raises_for)

    def retrieve(self, question, top_k=5):
        if question in self.raises_for:
            raise RuntimeError("retriever boom")
        return list(self.per_question.get(question, []))[:top_k]


class FakeJudge:
    """Faithfulness judge stub; returns preset replies in order (or a constant)."""
    def __init__(self, replies="1"):
        self.replies = replies
        self.calls = 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        if isinstance(self.replies, str):
            return self.replies
        return self.replies[self.calls - 1]


class FakePipeline:
    def __init__(self, retriever, answers, llm=None, top_k=5):
        self.retriever = retriever
        self.answers = answers
        self.llm = llm
        self.top_k = top_k

    def answer(self, question, history=None, top_k=None):
        a = self.answers[question]
        return {
            "answer": a.get("answer", ""),
            "sources": a.get("sources", []),
            "timings_ms": a.get("timings_ms", {"reformulate": 0, "retrieve": 1, "llm": 10}),
        }


KB = "C:\\AI-Assistant\\rag_service\\scripts\\..\\knowledge_base\\"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_basename_handles_full_windows_path():
    assert evaluator._basename(KB + "2020-Scrum-Guide-Russian.pdf") == "2020-Scrum-Guide-Russian.pdf"


def test_normalize_expected_variants():
    assert evaluator.normalize_expected(None) == []
    assert evaluator.normalize_expected("a.pdf") == ["a.pdf"]
    assert evaluator.normalize_expected([KB + "a.pdf", "b.pdf"]) == ["a.pdf", "b.pdf"]


def test_is_refusal():
    assert evaluator.is_refusal("В базе знаний нет информации по этому вопросу.")
    assert evaluator.is_refusal("К сожалению, я не знаю ответа.")
    assert not evaluator.is_refusal("Владелец продукта отвечает за бэклог [1].")


def test_parse_verdict():
    assert evaluator._parse_verdict("1") == 1
    assert evaluator._parse_verdict("0") == 0
    assert evaluator._parse_verdict("Оценка: 1 — обоснован") == 1
    assert evaluator._parse_verdict("нет цифр") is None


# --------------------------------------------------------------------------- #
# Metric 1 — recall@k
# --------------------------------------------------------------------------- #
def test_recall_at_k_basename_and_slicing():
    q = "Какие роли в Scrum?"
    # expected source sits at rank 3 -> miss@1, hit@3, hit@5
    chunks = [
        FakeChunk(KB + "other1.pdf"),
        FakeChunk(KB + "other2.pdf"),
        FakeChunk(KB + "2020-Scrum-Guide-Russian.pdf"),
        FakeChunk(KB + "other3.pdf"),
        FakeChunk(KB + "other4.pdf"),
    ]
    questions = [{
        "id": "q01", "question": q, "in_base": True,
        "expected_source": "2020-Scrum-Guide-Russian.pdf",
        "expected_keywords": [],
    }]
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers={})
    r = evaluator.eval_retrieval(questions, pipe, k_values=(1, 3, 5))
    assert r["recall@1"] == 0.0
    assert r["recall@3"] == 1.0
    assert r["recall@5"] == 1.0
    assert r["n_in_base"] == 1


def test_recall_accepts_list_of_sources():
    q = "Что такое DevOps?"
    chunks = [FakeChunk(KB + "практик DevOps.pdf")]
    questions = [{
        "id": "q15", "question": q, "in_base": True,
        "expected_source": ["new_DevOps_read_stamped.pdf", "практик DevOps.pdf"],
        "expected_keywords": [],
    }]
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers={})
    r = evaluator.eval_retrieval(questions, pipe, k_values=(1, 3, 5))
    assert r["recall@1"] == 1.0  # any acceptable source counts


def test_out_of_base_empty_retrieval_rate():
    q = "Столица Австралии?"
    questions = [{"id": "q21", "question": q, "in_base": False,
                  "expected_source": None, "expected_keywords": []}]
    pipe = FakePipeline(FakeRetriever({q: []}), answers={})  # gate -> empty
    r = evaluator.eval_retrieval(questions, pipe, k_values=(1, 3, 5))
    assert r["out_of_base_empty_retrieval_rate"] == 1.0
    assert r["n_out_of_base"] == 1


# --------------------------------------------------------------------------- #
# Metric 2 — keyword coverage
# --------------------------------------------------------------------------- #
def test_keyword_recall_case_insensitive_fraction():
    q = "роли"
    questions = [{
        "id": "q01", "question": q, "in_base": True,
        "expected_source": "x.pdf",
        "expected_keywords": ["Владелец продукта", "Scrum-мастер", "Разработчики"],
    }]
    chunks = [FakeChunk(KB + "x.pdf")]
    answers = {q: {"answer": "владелец продукта и разработчики важны"}}  # 2 of 3, mixed case
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers=answers)
    runs = evaluator.collect_runs(questions, pipe)
    kw = evaluator.eval_answer_keywords(questions, pipe, runs=runs)
    assert kw["keyword_recall"] == round(2 / 3, 4)  # metric rounds to 4 dp
    assert kw["all_keywords_rate"] == 0.0
    assert kw["any_keyword_rate"] == 1.0
    assert kw["per_question"][0]["missing"] == ["Scrum-мастер"]


def test_keyword_synonym_groups_match_any_variant():
    q = "роли"
    questions = [{
        "id": "q01", "question": q, "in_base": True, "expected_source": "x.pdf",
        "expected_keywords": [
            ["Владелец продукта", "Product Owner"],   # EN variant present
            ["Scrum-мастер", "Scrum Master"],          # EN variant present
            ["Разработчики", "Developers"],            # neither present
        ],
    }]
    chunks = [FakeChunk(KB + "x.pdf")]
    answers = {q: {"answer": "Roles: Product Owner and Scrum Master coordinate work."}}
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers=answers)
    runs = evaluator.collect_runs(questions, pipe)
    kw = evaluator.eval_answer_keywords(questions, pipe, runs=runs)
    assert kw["keyword_recall"] == round(2 / 3, 4)
    # missing group is reported by its canonical (first) label
    assert kw["per_question"][0]["missing"] == ["Разработчики"]
    assert kw["per_question"][0]["found"] == ["Владелец продукта", "Scrum-мастер"]


# --------------------------------------------------------------------------- #
# Metric 3 — faithfulness (LLM judge)
# --------------------------------------------------------------------------- #
def test_faithfulness_judges_grounded_answer():
    q = "роли"
    questions = [{"id": "q01", "question": q, "in_base": True,
                  "expected_source": "x.pdf", "expected_keywords": []}]
    chunks = [FakeChunk(KB + "x.pdf", text="Scrum роли: владелец продукта...")]
    answers = {q: {"answer": "Владелец продукта отвечает за бэклог [1]."}}
    judge = FakeJudge("1")
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers=answers, llm=judge)
    runs = evaluator.collect_runs(questions, pipe)
    f = evaluator.eval_faithfulness_llm(questions, pipe, runs=runs)
    assert f["faithfulness"] == 1.0
    assert judge.calls == 1  # judge was actually consulted
    assert f["per_question"][0]["judged"] is True


def test_faithfulness_refusal_short_circuits_without_judge():
    q = "Столица Австралии?"
    questions = [{"id": "q21", "question": q, "in_base": False,
                  "expected_source": None, "expected_keywords": []}]
    chunks = []  # no context
    answers = {q: {"answer": "В базе знаний нет информации по этому вопросу."}}
    judge = FakeJudge("0")  # would say unfaithful — but must NOT be called
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers=answers, llm=judge)
    runs = evaluator.collect_runs(questions, pipe)
    f = evaluator.eval_faithfulness_llm(questions, pipe, runs=runs)
    assert f["faithfulness"] == 1.0          # refusal counts as faithful
    assert judge.calls == 0                  # short-circuited
    assert f["per_question"][0]["judged"] is False


def test_faithfulness_unparseable_verdict_is_conservative():
    q = "роли"
    questions = [{"id": "q01", "question": q, "in_base": True,
                  "expected_source": "x.pdf", "expected_keywords": []}]
    chunks = [FakeChunk(KB + "x.pdf", text="контекст")]
    answers = {q: {"answer": "Утверждение без опоры на контекст."}}
    judge = FakeJudge("не понял вопрос")  # no digit
    pipe = FakePipeline(FakeRetriever({q: chunks}), answers=answers, llm=judge)
    runs = evaluator.collect_runs(questions, pipe)
    f = evaluator.eval_faithfulness_llm(questions, pipe, runs=runs)
    assert f["faithfulness"] == 0.0


# --------------------------------------------------------------------------- #
# Metric 4 — honest refusal
# --------------------------------------------------------------------------- #
def test_honest_refusal_rate():
    q1, q2 = "Столица Австралии?", "Рецепт борща?"
    questions = [
        {"id": "q21", "question": q1, "in_base": False, "expected_source": None, "expected_keywords": []},
        {"id": "q22", "question": q2, "in_base": False, "expected_source": None, "expected_keywords": []},
    ]
    answers = {
        q1: {"answer": "В базе знаний нет информации по этому вопросу."},  # good refusal
        q2: {"answer": "Возьмите свёклу и капусту..."},                    # hallucinated
    }
    pipe = FakePipeline(FakeRetriever({q1: [], q2: []}), answers=answers)
    runs = evaluator.collect_runs(questions, pipe)
    ref = evaluator.eval_honest_refusal(questions, pipe, runs=runs)
    assert ref["refusal_rate"] == 0.5
    assert ref["n"] == 2


# --------------------------------------------------------------------------- #
# Integration / robustness
# --------------------------------------------------------------------------- #
def test_collect_runs_records_pipeline_errors():
    q = "взрыв"
    questions = [{"id": "qx", "question": q, "in_base": True,
                  "expected_source": "x.pdf", "expected_keywords": ["foo"]}]
    pipe = FakePipeline(FakeRetriever({}, raises_for=[q]), answers={})
    runs = evaluator.collect_runs(questions, pipe)
    assert runs[0]["error"] is not None
    # error question is skipped by the judge, not counted as faithful/unfaithful
    f = evaluator.eval_faithfulness_llm(questions, pipe, runs=runs)
    assert f["per_question"][0]["note"] == "pipeline_error"


def test_run_all_smoke():
    qa = "роли"
    qb = "Столица Австралии?"
    questions = [
        {"id": "q01", "question": qa, "in_base": True,
         "expected_source": "2020-Scrum-Guide-Russian.pdf",
         "expected_keywords": ["Владелец продукта"]},
        {"id": "q21", "question": qb, "in_base": False,
         "expected_source": None, "expected_keywords": []},
    ]
    retr = FakeRetriever({
        qa: [FakeChunk(KB + "2020-Scrum-Guide-Russian.pdf", text="владелец продукта")],
        qb: [],
    })
    answers = {
        qa: {"answer": "Владелец продукта отвечает за бэклог [1]."},
        qb: {"answer": "В базе знаний нет информации по этому вопросу."},
    }
    pipe = FakePipeline(retr, answers=answers, llm=FakeJudge("1"))
    out = evaluator.run_all(questions, pipe)
    s = out["summary"]
    assert s["recall@1"] == 1.0
    assert s["keyword_recall"] == 1.0
    assert s["faithfulness"] == 1.0
    assert s["refusal_rate"] == 1.0
    assert s["n_errors"] == 0
