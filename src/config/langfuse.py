"""
=============================================================================
Langfuse Observability Configuration
=============================================================================

Langfuse v3 integration for LangChain observability.

IMPORTANT: Langfuse CallbackHandler reads credentials from environment variables:
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_BASE_URL (standardized from settings)

Make sure these are set in your .env file.
=============================================================================
"""

import logging
import os
from functools import lru_cache

from langfuse import Langfuse, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def _ensure_langfuse_env_vars():
    """
    Ensure Langfuse environment variables are set from settings.

    The Langfuse CallbackHandler reads from env vars directly,
    so we need to set them before creating handlers.
    """
    settings = get_settings()

    # Set env vars if not already set (Langfuse expects uppercase names)
    if settings.langfuse_public_key and not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key

    if settings.langfuse_secret_key and not os.environ.get("LANGFUSE_SECRET_KEY"):
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key

    # AGENTS.MD: Use LANGFUSE_BASE_URL instead of LANGFUSE_HOST
    if settings.langfuse_base_url and not os.environ.get("LANGFUSE_BASE_URL"):
        os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url
        # Also set LANGFUSE_HOST for SDK backward compatibility if needed
        os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url


@lru_cache
def get_langfuse_client() -> Langfuse:
    """Get cached Langfuse client instance."""
    settings = get_settings()

    # Check if credentials are configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "[LANGFUSE] Missing credentials. Set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY in .env file."
        )

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )


def get_langfuse_handler() -> CallbackHandler:
    """
    Create a Langfuse callback handler for LangChain.

    The CallbackHandler reads credentials from environment variables.
    We ensure env vars are set from our settings before creating the handler.

    NOTE: session_id and user_id should be passed in the metadata
    of the LangChain invoke() call config.
    """
    # Ensure env vars are set from settings
    _ensure_langfuse_env_vars()

    settings = get_settings()

    # Check if credentials are configured
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "[LANGFUSE] Missing credentials. Tracing will be disabled. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env file."
        )

    # Create handler - it will read from env vars
    return CallbackHandler()


__all__ = [
    "get_langfuse_client",
    "get_langfuse_handler",
    "observe",
    "propagate_attributes",
]
