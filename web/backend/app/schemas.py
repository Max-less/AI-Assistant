"""Pydantic DTOs for /api endpoints."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=6, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class GuestRequest(BaseModel):
    # Stable browser-generated id so the same device reuses its guest account.
    client_id: str | None = Field(default=None, max_length=64)


class UserOut(BaseModel):
    id: int
    email: EmailStr | None = None
    name: str | None = None
    is_guest: bool
    created_at: datetime
    # Remaining guest quota; None for registered users.
    guest_remaining: int | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut


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
