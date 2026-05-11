"""
HTTP API for the RAG service.

Run locally:
    uvicorn rag_service.api:app --host 0.0.0.0 --port 8000

Heavy components (vector store, embedder, BM25, reranker, LLM client) are built
once during the FastAPI lifespan and reused across requests.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

from bm25 import BM25
from embedder import Embedder
from llm_client import LLMClient
from query_expander import QueryExpander
from rag_pipeline import RAGPipeline
from reranker import Reranker
from retriever import Retriever
from vector_store import VectorStore


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npy")
META_PATH = os.path.join(DATA_DIR, "chunks_meta.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "")


class HistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[HistoryItem] | None = None
    top_k: int | None = Field(None, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int


class HealthResponse(BaseModel):
    status: Literal["ok", "loading"]
    index_loaded: bool
    chunk_count: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    auth_key = os.getenv("GIGACHAT_AUTH_KEY")
    if not auth_key or auth_key == "your-authorization-key-here":
        raise RuntimeError("GIGACHAT_AUTH_KEY is not set")

    for path in (VECTORS_PATH, META_PATH, CHUNKS_PATH):
        if not os.path.exists(path):
            raise RuntimeError(
                f"Index file missing: {path}. "
                "Run scripts/build_chunks.py then scripts/build_index.py first."
            )

    store = VectorStore.load_with_texts(VECTORS_PATH, META_PATH, CHUNKS_PATH)
    embedder = Embedder()
    bm25 = BM25().fit([c.text for c in store.chunks])

    reranker = Reranker() if _env_flag("RAG_USE_RERANKER", default=True) else None

    llm = LLMClient(auth_key)
    expander = QueryExpander(llm)
    retriever = Retriever(
        store,
        embedder,
        bm25=bm25,
        expander=expander,
        alpha=float(os.getenv("BM25_ALPHA", "0.5")),
        score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.5")),
        reranker=reranker,
    )
    app.state.pipeline = RAGPipeline(
        retriever, llm, top_k=int(os.getenv("RAG_TOP_K", "5"))
    )
    app.state.chunk_count = len(store.chunks)

    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    pipeline: RAGPipeline = request.app.state.pipeline
    history_dicts = (
        [h.model_dump() for h in req.history] if req.history else None
    )

    t0 = time.perf_counter()
    try:
        result = pipeline.answer(
            req.question, history=history_dicts, top_k=req.top_k
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")
    latency_ms = int((time.perf_counter() - t0) * 1000)

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return HealthResponse(status="loading", index_loaded=False)
    return HealthResponse(
        status="ok",
        index_loaded=True,
        chunk_count=request.app.state.chunk_count,
    )
