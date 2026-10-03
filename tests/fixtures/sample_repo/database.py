"""In-memory mock database client for sample repo."""

from .models import Product, SessionToken, User


class DatabaseClient:
    """Mock in-memory database storing users, sessions, and products."""

    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.sessions: dict[str, SessionToken] = {}
        self.products: dict[str, Product] = {}

    def get_user(self, user_id: str) -> User | None:
        """Fetch user by ID."""
        return self.users.get(user_id)

    def find_user_by_username(self, username: str) -> User | None:
        """Query user by unique username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def save_user(self, user: User) -> None:
        """Upsert a user record."""
        self.users[user.id] = user

    def get_session(self, token: str) -> SessionToken | None:
        """Fetch active session by token."""
        return self.sessions.get(token)

    def save_session(self, session: SessionToken) -> None:
        """Store session token."""
        self.sessions[session.token] = session

    def list_products(self) -> list[Product]:
        """Return all products in inventory."""
        return list(self.products.values())
