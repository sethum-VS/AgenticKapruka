"""FastAPI middleware and exception handlers."""

from app.middleware.errors import register_exception_handlers

__all__ = ["register_exception_handlers"]
