"""Auth endpoints: register, login, guest, me."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import User
from ..schemas import GuestRequest, LoginRequest, RegisterRequest, TokenResponse, UserOut
from ..security import create_access_token, hash_password, verify_password
from ..services import build_user_out

router = APIRouter()


def _token_response(db: Session, user: User) -> TokenResponse:
    token = create_access_token(subject=str(user.id), is_guest=user.is_guest)
    return TokenResponse(access_token=token, user=build_user_out(db, user))


@router.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = req.email.lower()
    exists = db.execute(select(User.id).where(User.email == email)).first()
    if exists is not None:
        raise HTTPException(status_code=400, detail="Этот email уже зарегистрирован")

    user = User(
        email=email,
        name=req.name.strip(),
        password_hash=hash_password(req.password),
        is_guest=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(db, user)


@router.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    email = req.email.lower()
    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None or user.password_hash is None or not verify_password(
        req.password, user.password_hash
    ):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    return _token_response(db, user)


@router.post("/auth/guest", response_model=TokenResponse)
def guest(
    req: GuestRequest | None = None, db: Session = Depends(get_db)
) -> TokenResponse:
    client_id = req.client_id if req else None

    # Reuse the existing guest account for this browser so its quota persists
    # across logout/login. Only fall through to creating a new one if unknown.
    if client_id:
        existing = db.execute(
            select(User).where(User.guest_id == client_id, User.is_guest.is_(True))
        ).scalar_one_or_none()
        if existing is not None:
            return _token_response(db, existing)

    user = User(is_guest=True, guest_id=client_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return _token_response(db, user)


@router.get("/auth/me", response_model=UserOut)
def me(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> UserOut:
    return build_user_out(db, user)
