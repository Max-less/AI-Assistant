"""
RAG pipeline orchestrator: retrieve -> build prompt -> call LLM.
Components are injected so they can be swapped in tests or future variants.
Supports optional dialog history for multi-turn chat.
"""

from llm_client import LLMClient
from prompt_builder import build_messages
from retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient, top_k: int = 8):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        messages = build_messages(question, chunks, history=history)
        answer = self.llm.complete(messages)
        return {
            "answer": answer,
            "sources": [c.source for c in chunks],
        }
