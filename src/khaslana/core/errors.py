"""Custom exception hierarchy for Khaslana."""


class KhaslanaError(Exception):
    """Base class for all Khaslana exceptions."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class ConfigurationError(KhaslanaError):
    """Raised when configuration is invalid or missing."""


class LLMError(KhaslanaError):
    """Raised when an LLM provider fails."""


class LLMConnectionError(LLMError):
    """Raised when connecting to an LLM provider fails."""


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""


class LLMAuthError(LLMError):
    """Raised when API key or credentials are invalid."""


class ToolError(KhaslanaError):
    """Raised when tool execution fails."""


class DependencyMissingError(KhaslanaError):
    """Raised when a required external binary or tool is missing."""

    def __init__(self, name: str, install_hint: str) -> None:
        super().__init__(
            message=f"Required dependency '{name}' is missing.",
            hint=install_hint,
        )
        self.name = name


class RedactionError(KhaslanaError):
    """Raised when secret redaction encounters an issue."""
