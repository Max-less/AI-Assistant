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


def _basename(path: str) -> str:
    """Filename without directories, normalising both / and \\ separators."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _clean_snippet(text: str) -> str:
    """PDF extraction often yields one word per line; rejoin soft line breaks
    into running prose for display while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate split words
    text = re.sub(r"[ \t]*\n{2,}[ \t]*", "\n\n", text)  # keep paragraph breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)  # soft break -> space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _filter_cited(answer: str, chunks: list[Chunk]) -> tuple[str, list[dict]]:
    """Keep only chunks actually cited as [N] in the answer.

    Dedupes by the cited chunk (so two distinct passages from the same file
    keep separate numbers — needed for "show the exact passage") and renumbers
    citations sequentially so answer indices match the final sources list.
    Each source carries the document ``filename`` and the ``snippet`` (the
    retrieved chunk text) it was drawn from. If no valid citation is found,
    returns the answer unchanged and an empty sources list."""
    # Key on chunk identity (source + text) so repeated cites collapse but
    # different passages of the same document don't.
    key_to_num: dict[tuple[str, str], int] = {}
    key_to_chunk: dict[tuple[str, str], Chunk] = {}
    for m in _CITATION_RE.finditer(answer):
        n = int(m.group(1))
        if not (1 <= n <= len(chunks)):
            continue
        chunk = chunks[n - 1]
        key = (chunk.source, chunk.text)
        if key not in key_to_num:
            key_to_num[key] = len(key_to_num) + 1
            key_to_chunk[key] = chunk

    if not key_to_num:
        return answer, []

    def _repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            key = (chunk.source, chunk.text)
            if key in key_to_num:
                return f"[{key_to_num[key]}]"
        return m.group(0)

    new_answer = _CITATION_RE.sub(_repl, answer)
    ordered = sorted(key_to_num, key=lambda k: key_to_num[k])
    sources = [
        {
            "filename": _basename(key_to_chunk[k].source),
            "snippet": _clean_snippet(key_to_chunk[k].text),
        }
        for k in ordered
    ]
    return new_answer, sources


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        dialog_manager: DialogManager | None = None,
        top_k: int = 8,
        system_prompt: str | None = None,
        answer_max_tokens: int | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.dialog_manager = dialog_manager
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.answer_max_tokens = answer_max_tokens

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
        answer = self.llm.complete(messages, max_tokens=self.answer_max_tokens)
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
