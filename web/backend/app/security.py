"""Password hashing (bcrypt) and JWT access tokens (PyJWT)."""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

_DEFAULT_DEV_SECRET = "dev-insecure-change-me"
JWT_SECRET = os.getenv("JWT_SECRET", _DEFAULT_DEV_SECRET)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(7 * 24 * 60)))

if JWT_SECRET == _DEFAULT_DEV_SECRET:
    warnings.warn(
        "JWT_SECRET is unset — using an insecure dev default. "
        "Set JWT_SECRET (e.g. `openssl rand -hex 32`) in production.",
        stacklevel=2,
    )

# bcrypt rejects inputs longer than 72 bytes; the password length is also bounded
# in the request schema, but we truncate defensively here too.
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    pw = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    try:
        return bcrypt.checkpw(pw, password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, is_guest: bool) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "is_guest": is_guest,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
