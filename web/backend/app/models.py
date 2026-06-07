"""ORM models: sessions, messages, feedback."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Email/password/name are null for guest accounts. SQLite allows multiple
    # NULLs under a UNIQUE constraint, so guests don't collide on email.
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_guest: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    # Stable per-browser id for guests, so logging out and back in as a guest
    # reuses the same account (and its consumed quota) instead of resetting it.
    guest_id: Mapped[str | None] = mapped_column(
        String(64), unique=True, nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class ChatSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.id",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped[ChatSession] = relationship(back_populates="messages")
    feedback: Mapped["Feedback | None"] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        unique=True,
        index=True,
    )
    value: Mapped[int]  # +1 or -1
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    message: Mapped[Message] = relationship(back_populates="feedback")
