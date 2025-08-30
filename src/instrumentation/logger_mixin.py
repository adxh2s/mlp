from __future__ import annotations

from typing import Optional, Protocol


class SupportsGetLogger(Protocol):
    """Protocol for objects exposing get_logger(name)->logger."""

    def get_logger(self, name: Optional[str] = None):
        """Return a logger instance (stdlib or structlog-compatible)."""
        ...


class LoggerMixin:
    """Provide a self.log attribute using a provided LoggerManager-like object."""

    LOGGER_NAME = __name__

    def _init_logger(self, lm: SupportsGetLogger) -> None:
        """Initialize self.log with a named logger from the logger manager.

        Args:
            lm: Object exposing get_logger(name) (LoggerManager/StructlogLoggerManager).
        """
        self.log = lm.get_logger(self.LOGGER_NAME)
