"""Pydantic DTOs for /api endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: int | None = None


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    sources: list[str] | None = None
    latency_ms: int | None = None
    latency_breakdown: dict[str, Any] | None = None
    created_at: datetime
    feedback: Literal[-1, 1] | None = None


class ChatResponse(BaseModel):
    session_id: int
    message: MessageOut


class SessionSummary(BaseModel):
    id: int
    title: str
    created_at: datetime
    message_count: int


class SessionDetail(BaseModel):
    id: int
    title: str
    created_at: datetime


class HistoryResponse(BaseModel):
    session: SessionDetail
    messages: list[MessageOut]


class FeedbackRequest(BaseModel):
    message_id: int
    value: Literal[-1, 0, 1]


class FeedbackResponse(BaseModel):
    ok: bool = True


class HealthResponse(BaseModel):
    web: Literal["ok"] = "ok"
    rag: Literal["ok", "loading", "down"]
    chunk_count: int | None = None
