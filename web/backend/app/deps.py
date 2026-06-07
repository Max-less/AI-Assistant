"""Shared FastAPI dependencies — current-user resolution from the Bearer token."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import decode_token

_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise _UNAUTHORIZED

    payload = decode_token(creds.credentials)
    if payload is None:
        raise _UNAUTHORIZED

    sub = payload.get("sub")
    if sub is None:
        raise _UNAUTHORIZED

    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        raise _UNAUTHORIZED

    user = db.get(User, user_id)
    if user is None:
        raise _UNAUTHORIZED

    return user
