"""
HTTP API for the RAG service.

Run locally:
    uvicorn rag_service.api:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

from bm25 import BM25
from dialog_manager import DialogManager
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
    use_expander: bool | None = None
    use_reranker: bool | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int
    latency_breakdown: dict[str, int]


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
    dialog_manager = DialogManager(llm)

    # Warm up models so the first real request doesn't pay first-inference cost.
    embedder.embed_query("warmup")
    if reranker is not None:
        reranker.model.predict([("warmup", "warmup")])

    app.state.store = store
    app.state.embedder = embedder
    app.state.bm25 = bm25
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.expander = expander
    app.state.dialog_manager = dialog_manager
    app.state.default_use_expander = _env_flag("RAG_USE_EXPANDER", default=True)
    app.state.default_use_reranker = _env_flag("RAG_USE_RERANKER", default=True)
    app.state.default_top_k = int(os.getenv("RAG_TOP_K", "5"))
    app.state.rerank_pool = int(os.getenv("RAG_RERANK_POOL", "10"))
    app.state.alpha = float(os.getenv("BM25_ALPHA", "0.5"))
    app.state.score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
    app.state.chunk_count = len(store.chunks)

    yield

    llm.close()


app = FastAPI(title="RAG Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    s = request.app.state

    use_expander = (
        req.use_expander if req.use_expander is not None else s.default_use_expander
    )
    use_reranker = (
        req.use_reranker if req.use_reranker is not None else s.default_use_reranker
    )

    retriever = Retriever(
        s.store,
        s.embedder,
        bm25=s.bm25,
        expander=s.expander if use_expander else None,
        reranker=s.reranker if (use_reranker and s.reranker is not None) else None,
        rerank_pool=s.rerank_pool,
        alpha=s.alpha,
        score_threshold=s.score_threshold,
    )
    pipeline = RAGPipeline(
        retriever,
        s.llm,
        dialog_manager=s.dialog_manager,
        top_k=s.default_top_k,
    )

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
        latency_breakdown=result["timings_ms"],
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    chunk_count = getattr(request.app.state, "chunk_count", None)
    if chunk_count is None:
        return HealthResponse(status="loading", index_loaded=False)
    return HealthResponse(
        status="ok",
        index_loaded=True,
        chunk_count=chunk_count,
    )
