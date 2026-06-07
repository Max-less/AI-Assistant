"""Test fixtures: isolated SQLite DB, env config, and a stubbed RAG client."""

import os
import tempfile

# Configure the environment BEFORE importing the app (modules read env at import).
_tmp_dir = tempfile.mkdtemp(prefix="svod-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_dir}/test.db"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["GUEST_QUERY_LIMIT"] = "5"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class StubRag:
    """Stand-in for RagClient so chat works without the upstream service."""

    def ask(self, question: str, history: list[dict[str, str]]) -> dict:
        return {
            "answer": f"stub answer to: {question}",
            "sources": ["stub-source.pdf"],
            "latency_ms": 1,
            "latency_breakdown": {},
        }

    def health(self) -> dict:
        return {"status": "ok", "chunk_count": 0}

    def close(self) -> None:
        pass


@pytest.fixture
def client():
    from app.db import Base, engine
    from app.main import app

    # Fresh schema per test.
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    with TestClient(app) as c:
        app.state.rag = StubRag()  # replace the real client set up in lifespan
        yield c
