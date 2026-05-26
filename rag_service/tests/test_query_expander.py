import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from query_expander import QueryExpander


class RecordingLLM:
    """Records every call to .complete() so we can assert it wasn't invoked."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        return self.response


def test_short_single_topic_question_skips_llm():
    """Fast-path: short question without conjunction markers must NOT hit the LLM."""
    llm = RecordingLLM()
    expander = QueryExpander(llm)
    out = expander.expand("Что такое Scrum?")
    assert out == ["Что такое Scrum?"]
    assert llm.calls == 0


def test_question_with_comma_goes_to_llm():
    """Question with a connective marker is treated as potentially multi-topic."""
    llm = RecordingLLM(response="часть один\nчасть два")
    expander = QueryExpander(llm)
    out = expander.expand("Расскажи про Scrum, а также про Kanban")
    assert llm.calls == 1
    assert out == ["часть один", "часть два"]


def test_long_question_goes_to_llm():
    """Even without obvious connectives, a long question goes through the LLM."""
    long_q = "Опиши очень подробно " + "детали " * 20 + "про процесс работы"
    assert len(long_q) >= 80
    llm = RecordingLLM(response=long_q)
    expander = QueryExpander(llm)
    expander.expand(long_q)
    assert llm.calls == 1


def test_llm_failure_falls_back_to_original():
    """If the LLM raises on a multi-topic question, expander returns [question]."""

    class FailingLLM:
        def complete(self, messages):
            raise RuntimeError("boom")

    expander = QueryExpander(FailingLLM())
    out = expander.expand("вопрос один, вопрос два")
    assert out == ["вопрос один, вопрос два"]
