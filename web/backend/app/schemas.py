"""Pydantic DTOs for /api endpoints."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, EmailStr, Field, PlainSerializer, field_validator


def _to_utc_iso(dt: datetime) -> str:
    # DB timestamps (SQLite CURRENT_TIMESTAMP) are naive UTC. Tag them as UTC so
    # the JSON carries an explicit offset; otherwise JS new Date() treats the
    # string as local time and shifts it (e.g. early-morning rows land in "Вчера").
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# datetime that always serialises with an explicit UTC offset.
UTCDateTime = Annotated[datetime, PlainSerializer(_to_utc_iso, return_type=str)]


def _basename(path: str) -> str:
    """Filename without directories, normalising both / and \\ separators."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


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
    created_at: UTCDateTime
    # Remaining guest quota; None for registered users.
    guest_remaining: int | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserOut


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: int | None = None


class Source(BaseModel):
    filename: str
    # Empty for messages persisted before snippets were stored (legacy rows).
    snippet: str = ""


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] | None = None
    latency_ms: int | None = None
    latency_breakdown: dict[str, Any] | None = None
    created_at: UTCDateTime
    feedback: Literal[-1, 1] | None = None

    @field_validator("sources", mode="before")
    @classmethod
    def _coerce_sources(cls, v: Any) -> Any:
        # Legacy rows stored sources as bare path strings; new rows store
        # {filename, snippet} dicts. Normalise both to the Source shape.
        if not isinstance(v, list):
            return v
        return [
            {"filename": _basename(item), "snippet": ""} if isinstance(item, str) else item
            for item in v
        ]


class ChatResponse(BaseModel):
    session_id: int
    message: MessageOut


class SessionSummary(BaseModel):
    id: int
    title: str
    created_at: UTCDateTime
    message_count: int


class SessionDetail(BaseModel):
    id: int
    title: str
    created_at: UTCDateTime


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


class DocumentsResponse(BaseModel):
    documents: list[str]
