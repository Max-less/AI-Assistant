"""Small shared helpers around users and the guest quota."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import GUEST_QUERY_LIMIT
from .models import ChatSession, Message, User
from .schemas import UserOut


def count_user_queries(db: Session, user_id: int) -> int:
    """Number of user-role messages the principal has sent (its 'queries')."""
    return db.execute(
        select(func.count(Message.id))
        .join(ChatSession, ChatSession.id == Message.session_id)
        .where(ChatSession.user_id == user_id, Message.role == "user")
    ).scalar_one()


def guest_remaining(db: Session, user: User) -> int | None:
    if not user.is_guest:
        return None
    return max(0, GUEST_QUERY_LIMIT - count_user_queries(db, user.id))


def build_user_out(db: Session, user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        name=user.name,
        is_guest=user.is_guest,
        created_at=user.created_at,
        guest_remaining=guest_remaining(db, user),
    )
