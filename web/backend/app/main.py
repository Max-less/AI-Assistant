"""FastAPI entrypoint for the web backend (BFF in front of the RAG service)."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import Base, engine
from .rag_client import RagClient
from .routes import chat, feedback, health, history


def _cors_origins() -> list[str]:
    raw = os.getenv("WEB_CORS_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Import models so SQLAlchemy registers the tables, then create them.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8000")
    rag_timeout = float(os.getenv("RAG_TIMEOUT", "60"))
    app.state.rag = RagClient(rag_url, timeout=rag_timeout)

    yield

    app.state.rag.close()


app = FastAPI(title="Svod Web Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
app.include_router(health.router, prefix="/api", tags=["health"])
