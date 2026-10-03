"""API router endpoints for sample repo."""

from typing import Any

from .auth import AuthService
from .database import DatabaseClient


class AppRouter:
    """HTTP endpoint router dispatching to AuthService and DatabaseClient."""

    def __init__(self, auth_service: AuthService, db: DatabaseClient) -> None:
        self.auth = auth_service
        self.db = db

    def post_login(self, body: dict[str, str]) -> dict[str, Any]:
        """Authenticate user and return session token."""
        username = body.get("username", "")
        password = body.get("password", "")
        user = self.auth.authenticate_user(username, password)
        if not user:
            return {"error": "Invalid credentials", "status": 401}
        session = self.auth.issue_token(user)
        return {"token": session.token, "expires_at": session.expires_at}

    def get_me(self, token: str) -> dict[str, Any]:
        """Get profile for authenticated token."""
        user = self.auth.validate_token(token)
        if not user:
            return {"error": "Unauthorized", "status": 401}
        return {"id": user.id, "username": user.username, "email": user.email}
