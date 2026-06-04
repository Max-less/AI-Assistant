"""Thin HTTP client to the RAG service."""

from typing import Any

import httpx


class RagError(Exception):
    """Raised when the upstream RAG service returns an error or is unreachable."""


class RagClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def ask(self, question: str, history: list[dict[str, str]]) -> dict[str, Any]:
        try:
            r = self._client.post("/ask", json={"question": question, "history": history})
        except httpx.RequestError as e:
            raise RagError(f"unreachable: {e}") from e
        if r.status_code >= 400:
            detail = r.text
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            raise RagError(f"{r.status_code}: {detail}")
        return r.json()

    def health(self) -> dict[str, Any] | None:
        try:
            r = self._client.get("/health", timeout=5.0)
            if r.status_code != 200:
                return None
            return r.json()
        except httpx.RequestError:
            return None

    def close(self) -> None:
        self._client.close()
