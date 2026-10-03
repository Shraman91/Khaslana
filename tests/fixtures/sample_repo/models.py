"""Data models for sample repository."""

import time

from pydantic import BaseModel, Field


class User(BaseModel):
    """User account entity."""

    id: str
    username: str
    email: str
    password_hash: str
    salt: str
    created_at: float = Field(default_factory=time.time)


class SessionToken(BaseModel):
    """Active user session representation."""

    token: str
    user_id: str
    expires_at: float

    def is_expired(self) -> bool:
        """Check if token lifetime has passed."""
        return time.time() > self.expires_at


class Product(BaseModel):
    """E-commerce product entity."""

    id: str
    title: str
    price: float
    inventory_count: int
