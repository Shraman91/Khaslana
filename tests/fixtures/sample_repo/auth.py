"""Authentication and session management module."""

import hashlib
import time

from .database import DatabaseClient
from .models import SessionToken, User


class AuthService:
    """Handles user authentication, token issuance, and password hashing."""

    def __init__(self, db: DatabaseClient, secret_key: str = "khaslana-test-secret") -> None:
        self.db = db
        self.secret_key = secret_key

    def hash_password(self, password: str, salt: str = "s@lt") -> str:
        """Hash a plaintext password using SHA256 with salt."""
        payload = f"{password}:{salt}:{self.secret_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Verify username and password against database records."""
        user = self.db.find_user_by_username(username)
        if not user:
            return None
        pwd_hash = self.hash_password(password, salt=user.salt)
        if user.password_hash == pwd_hash:
            return user
        return None

    def issue_token(self, user: User, expires_in_seconds: int = 3600) -> SessionToken:
        """Issue a new session token for an authenticated user."""
        raw = f"{user.id}:{time.time()}:{self.secret_key}"
        token_str = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        session = SessionToken(
            token=token_str,
            user_id=user.id,
            expires_at=time.time() + expires_in_seconds,
        )
        self.db.save_session(session)
        return session

    def validate_token(self, token_str: str) -> User | None:
        """Validate an active session token and return the associated User."""
        session = self.db.get_session(token_str)
        if not session or session.is_expired():
            return None
        return self.db.get_user(session.user_id)
