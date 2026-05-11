"""
RAG pipeline orchestrator: retrieve -> build prompt -> call LLM.
Components are injected so they can be swapped in tests or future variants.
Supports optional dialog history for multi-turn chat.
If the retriever returns no chunks (e.g. relevance threshold not met), the
pipeline short-circuits with a canned "no info" answer instead of calling the
LLM — guards against hallucination on out-of-scope questions.
"""

from llm_client import LLMClient
from prompt_builder import build_messages
from retriever import Retriever


NO_CONTEXT_ANSWER = "В базе знаний нет информации по этому вопросу."


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 8):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> dict:
        effective_top_k = top_k if top_k is not None else self.top_k
        chunks = self.retriever.retrieve(question, top_k=effective_top_k)
        if not chunks:
            return {"answer": NO_CONTEXT_ANSWER, "sources": []}

        messages = build_messages(question, chunks, history=history)
        answer = self.llm.complete(messages)
        return {
            "answer": answer,
            "sources": [c.source for c in chunks],
        }
