"""
RAG pipeline orchestrator: retrieve -> build prompt -> call LLM.
Components are injected so they can be swapped in tests or future variants.
Supports optional dialog history for multi-turn chat.
If the retriever returns no chunks (e.g. relevance threshold not met), the
pipeline short-circuits with a canned "no info" answer instead of calling the
LLM — guards against hallucination on out-of-scope questions.
The result includes per-stage timings for diagnostic visibility.
"""

import re
import time

from chunker import Chunk
from dialog_manager import DialogManager
from llm_client import LLMClient
from prompt_builder import build_messages
from retriever import Retriever


NO_CONTEXT_ANSWER = "В базе знаний нет информации по этому вопросу."

_CITATION_RE = re.compile(r"\[(\d+)\]")


def _filter_cited(answer: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    """Keep only sources actually cited as [N] in the answer.
    Dedupes by source path and renumbers citations sequentially so answer
    indices match the final sources list. If no valid citation is found,
    returns the answer unchanged and an empty sources list."""
    source_to_new_num: dict[str, int] = {}
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if not (1 <= n <= len(chunks)):
            continue
        src = chunks[n - 1].source
        if src not in source_to_new_num:
            source_to_new_num[src] = len(source_to_new_num) + 1

    if not source_to_new_num:
        return answer, []

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            src = chunks[n - 1].source
            if src in source_to_new_num:
                return f"[{source_to_new_num[src]}]"
        return m.group(0)

    new_answer = _CITATION_RE.sub(_repl, answer)
    sources = sorted(source_to_new_num, key=lambda s: source_to_new_num[s])
    return new_answer, sources


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        dialog_manager: DialogManager | None = None,
        top_k: int = 8,
        system_prompt: str | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.dialog_manager = dialog_manager
        self.top_k = top_k
        self.system_prompt = system_prompt

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> dict:
        effective_top_k = top_k if top_k is not None else self.top_k

        reformulate_ms = 0
        retrieval_query = question
        if history and self.dialog_manager is not None:
            t = time.perf_counter()
            retrieval_query = self.dialog_manager.reformulate(question, history)
            reformulate_ms = int((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        chunks = self.retriever.retrieve(retrieval_query, top_k=effective_top_k)
        retrieve_ms = int((time.perf_counter() - t) * 1000)

        if not chunks:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "timings_ms": {
                    "reformulate": reformulate_ms,
                    "retrieve": retrieve_ms,
                    "llm": 0,
                },
            }

        messages = build_messages(
            question, chunks, history=history, system_prompt=self.system_prompt
        )
        t = time.perf_counter()
        answer = self.llm.complete(messages)
        llm_ms = int((time.perf_counter() - t) * 1000)

        answer, cited_sources = _filter_cited(answer, chunks)

        return {
            "answer": answer,
            "sources": cited_sources,
            "timings_ms": {
                "reformulate": reformulate_ms,
                "retrieve": retrieve_ms,
                "llm": llm_ms,
            },
        }
