r"""Provide a resolution utility for creating LangChain embedding
models."""

from __future__ import annotations

__all__ = ["resolve_embeddings"]

import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from objectory import factory

logger: logging.Logger = logging.getLogger(__name__)


def resolve_embeddings(embeddings: Embeddings | dict[str, Any]) -> Embeddings:
    """Resolve a LangChain
    :class:`~langchain_core.embeddings.Embeddings` instance from an
    existing object or a configuration dictionary.

    If ``embeddings`` is already an
    :class:`~langchain_core.embeddings.Embeddings` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        embeddings: Either a fully configured
            :class:`~langchain_core.embeddings.Embeddings` instance, or
            a :class:`dict` containing an ``objectory`` factory
            specification (must include a ``"_target_"`` key pointing to
            the fully-qualified class name).

    Returns:
        A configured :class:`~langchain_core.embeddings.Embeddings`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_core.embeddings.Embeddings` instance.

    Example:
        ```pycon
        >>> from zenpyre.embeddings import resolve_embeddings
        >>> # From an existing instance:
        >>> from langchain_core.embeddings.fake import FakeEmbeddings
        >>> embeddings = resolve_embeddings(FakeEmbeddings(size=128))
        >>> # From a configuration dictionary:
        >>> embeddings = resolve_embeddings(
        ...     {"_target_": "langchain_ollama.OllamaEmbeddings", "model": "nomic-embed-text"}
        ... )

        ```
    """
    if isinstance(embeddings, dict):
        logger.info("Initializing an Embeddings instance from its configuration...")
        embeddings = factory(**embeddings)
    if not isinstance(embeddings, Embeddings):
        msg = f"Received object is not an Embeddings instance (received: {type(embeddings)})"
        raise TypeError(msg)
    return embeddings
