import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dialog_manager import DialogManager


class RecordingLLM:
    """Records calls so we can assert the reformulation LLM was (not) invoked."""

    def __init__(self, response: str = "переформулированный вопрос"):
        self.response = response
        self.calls = 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        return self.response


HISTORY = [
    {"role": "user", "content": "Что такое Scrum?"},
    {"role": "assistant", "content": "Scrum — это фреймворк..."},
]


def test_no_history_skips_llm():
    llm = RecordingLLM()
    dm = DialogManager(llm)
    assert dm.reformulate("Какие роли в Scrum?", []) == "Какие роли в Scrum?"
    assert llm.calls == 0


def test_standalone_question_skips_llm():
    """Long, marker-free follow-up is self-contained -> no LLM call."""
    llm = RecordingLLM()
    dm = DialogManager(llm)
    q = "Какие основные роли существуют в методологии Scrum?"
    assert dm.reformulate(q, HISTORY) == q
    assert llm.calls == 0


def test_anaphoric_question_triggers_llm():
    """Pronoun reference -> must reformulate."""
    llm = RecordingLLM(response="Как внедрить Scrum?")
    dm = DialogManager(llm)
    out = dm.reformulate("А как его внедрить?", HISTORY)
    assert llm.calls == 1
    assert out == "Как внедрить Scrum?"


def test_short_followup_triggers_llm():
    """Short elliptical follow-up -> reformulate."""
    llm = RecordingLLM(response="Какие минусы у Scrum?")
    dm = DialogManager(llm)
    dm.reformulate("А минусы?", HISTORY)
    assert llm.calls == 1


def test_llm_failure_falls_back_to_original():
    class FailingLLM:
        def complete(self, messages, max_tokens=None):
            raise RuntimeError("boom")

    dm = DialogManager(FailingLLM())
    q = "А как его настроить под нашу команду правильно?"
    assert dm.reformulate(q, HISTORY) == q
