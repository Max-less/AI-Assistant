"""POST /api/feedback — upsert (or delete) a thumbs up/down on an assistant message."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Feedback, Message, User
from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def post_feedback(
    req: FeedbackRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> FeedbackResponse:
    message = db.get(Message, req.message_id)
    if message is None or message.session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(status_code=400, detail="Feedback only applies to assistant messages")

    existing = message.feedback

    if req.value == 0:
        if existing is not None:
            db.delete(existing)
            db.commit()
        return FeedbackResponse()

    if existing is None:
        db.add(Feedback(message_id=message.id, value=req.value))
    else:
        existing.value = req.value
    db.commit()
    return FeedbackResponse()
