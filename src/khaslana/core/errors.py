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


class SandboxError(KhaslanaError):
    """Base error for sandbox container operations."""


class SandboxUnavailableError(SandboxError):
    """Raised when Docker daemon or container runtime is unavailable."""

    def __init__(
        self,
        message: str = "Docker daemon is not running or Docker is not installed.",
        hint: str = (
            "Please install Docker or start Docker Desktop to use sandbox execution."
        ),
    ) -> None:
        super().__init__(message=message, hint=hint)


class ExecutionTimeoutError(SandboxError):
    """Raised when sandbox execution exceeds allowed timeout."""


class MemoryStoreError(KhaslanaError):
    """Raised when memory persistence or recall encounters an error."""


class WatcherError(KhaslanaError):
    """Raised when file watching or incremental indexing encounters an issue."""


class DaemonError(KhaslanaError):
    """Raised when daemon lifecycle operations (start/stop/status) fail."""


class PluginError(KhaslanaError):
    """Raised when plugin loading, registration, or validation fails."""


