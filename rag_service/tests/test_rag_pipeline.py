import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunker import Chunk
from rag_pipeline import NO_CONTEXT_ANSWER, RAGPipeline


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k):
        return self.chunks


class FakeLLM:
    def __init__(self, response="some answer"):
        self.response = response
        self.calls = 0

    def complete(self, messages, max_tokens=None):
        self.calls += 1
        return self.response


def test_pipeline_returns_canned_answer_when_no_chunks():
    llm = FakeLLM()
    pipeline = RAGPipeline(FakeRetriever([]), llm)
    result = pipeline.answer("вопрос вне базы")

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["sources"] == []
    assert llm.calls == 0  # LLM is not called on empty retrieval


def test_pipeline_calls_llm_when_chunks_present():
    chunk = Chunk(text="контент", source="doc.md", chunk_id="doc.md::0")
    llm = FakeLLM(response="ответ [1]")  # cite the source so it survives _filter_cited
    pipeline = RAGPipeline(FakeRetriever([chunk]), llm)
    result = pipeline.answer("вопрос")

    assert result["answer"] == "ответ [1]"
    assert result["sources"] == ["doc.md"]
    assert llm.calls == 1


def test_pipeline_drops_uncited_sources():
    # _filter_cited keeps only sources actually cited as [N]; an answer with no
    # citation yields an empty sources list (the answer text is returned as-is).
    chunk = Chunk(text="контент", source="doc.md", chunk_id="doc.md::0")
    llm = FakeLLM(response="ответ без ссылки")
    pipeline = RAGPipeline(FakeRetriever([chunk]), llm)
    result = pipeline.answer("вопрос")

    assert result["answer"] == "ответ без ссылки"
    assert result["sources"] == []
    assert llm.calls == 1
