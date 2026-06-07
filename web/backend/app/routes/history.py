"""GET /api/sessions and GET /api/history/{session_id}."""

from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import ChatSession, Message, User
from ..schemas import HistoryResponse, MessageOut, SessionDetail, SessionSummary

router = APIRouter()


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[SessionSummary]:
    rows = db.execute(
        select(ChatSession, func.count(Message.id))
        .outerjoin(Message, Message.session_id == ChatSession.id)
        .where(ChatSession.user_id == user.id)
        .group_by(ChatSession.id)
        .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
    ).all()

    return [
        SessionSummary(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            message_count=count or 0,
        )
        for s, count in rows
    ]


@router.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> HistoryResponse:
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        MessageOut(
            id=m.id,
            role=cast(str, m.role),  # "user" | "assistant"
            content=m.content,
            sources=m.sources,
            latency_ms=m.latency_ms,
            latency_breakdown=m.latency_breakdown,
            created_at=m.created_at,
            feedback=m.feedback.value if m.feedback else None,
        )
        for m in session.messages
    ]

    return HistoryResponse(
        session=SessionDetail(
            id=session.id, title=session.title, created_at=session.created_at
        ),
        messages=messages,
    )
