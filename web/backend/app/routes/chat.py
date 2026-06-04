"""POST /api/chat — accept a question, call RAG, persist both turns."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import ChatSession, Message
from ..rag_client import RagClient, RagError
from ..schemas import ChatRequest, ChatResponse, MessageOut

router = APIRouter()

HISTORY_TURNS = 6


def _title_from(question: str) -> str:
    title = question.strip().splitlines()[0] if question.strip() else "Новая беседа"
    return title[:80] or "Новая беседа"


@router.post("/chat", response_model=ChatResponse)
def post_chat(req: ChatRequest, request: Request, db: Session = Depends(get_db)) -> ChatResponse:
    rag: RagClient = request.app.state.rag

    session = db.get(ChatSession, req.session_id) if req.session_id is not None else None
    if session is None:
        session = ChatSession(title=_title_from(req.question))
        db.add(session)
        db.flush()

    history = [
        {"role": m.role, "content": m.content}
        for m in session.messages[-HISTORY_TURNS:]
    ]

    user_msg = Message(session_id=session.id, role="user", content=req.question)
    db.add(user_msg)
    db.flush()

    try:
        result = rag.ask(req.question, history)
    except RagError as e:
        # Persist the user's turn so it shows up in history, then surface the error.
        db.commit()
        raise HTTPException(status_code=502, detail=f"RAG service error: {e}")

    asst = Message(
        session_id=session.id,
        role="assistant",
        content=result.get("answer", ""),
        sources=result.get("sources") or None,
        latency_ms=result.get("latency_ms"),
        latency_breakdown=result.get("latency_breakdown") or None,
    )
    db.add(asst)
    db.commit()
    db.refresh(asst)

    return ChatResponse(
        session_id=session.id,
        message=MessageOut(
            id=asst.id,
            role="assistant",
            content=asst.content,
            sources=asst.sources,
            latency_ms=asst.latency_ms,
            latency_breakdown=asst.latency_breakdown,
            created_at=asst.created_at,
            feedback=None,
        ),
    )
