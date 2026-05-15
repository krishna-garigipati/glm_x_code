"""Custom exceptions for Walker component."""


class WalkerError(Exception):
    """Base exception for Walker component."""


class ValidationError(WalkerError):
    """Raised when configuration or input validation fails."""


class EmbeddingLookupError(WalkerError):
    """Raised when node embeddings cannot be resolved."""
