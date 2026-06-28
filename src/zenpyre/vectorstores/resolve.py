r"""Provide a resolution utility for creating LangChain vector store
models."""

from __future__ import annotations

__all__ = ["resolve_vector_store"]

import logging
from typing import Any

from langchain_core.vectorstores import VectorStore
from objectory import factory

logger: logging.Logger = logging.getLogger(__name__)


def resolve_vector_store(vector_store: VectorStore | dict[str, Any]) -> VectorStore:
    """Resolve a LangChain
    :class:`~langchain_core.vectorstores.VectorStore` instance from an
    existing object or a configuration dictionary.

    If ``vector_store`` is already a
    :class:`~langchain_core.vectorstores.VectorStore` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        vector_store: Either a fully configured
            :class:`~langchain_core.vectorstores.VectorStore` instance,
            or a :class:`dict` containing an ``objectory`` factory
            specification (must include a ``"_target_"`` key pointing to
            the fully-qualified class name).

    Returns:
        A configured :class:`~langchain_core.vectorstores.VectorStore`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_core.vectorstores.VectorStore` instance.

    Example:
        ```pycon
        >>> from zenpyre.vectorstores.resolve import resolve_vector_store
        >>> from langchain_core.vectorstores import InMemoryVectorStore
        >>> from langchain_core.embeddings.fake import FakeEmbeddings
        >>> # From an existing instance:
        >>> vector_store = resolve_vector_store(InMemoryVectorStore(FakeEmbeddings(size=128)))
        >>> # From a configuration dictionary:
        >>> vector_store = resolve_vector_store(  # doctest: +SKIP
        ...     {
        ...         "_target_": "langchain_core.vectorstores.InMemoryVectorStore",
        ...         "embedding": FakeEmbeddings(size=128),
        ...     }
        ... )

        ```
    """
    if isinstance(vector_store, dict):
        logger.info("Initializing a VectorStore instance from its configuration...")
        vector_store = factory(**vector_store)
    if not isinstance(vector_store, VectorStore):
        msg = f"Received object is not a VectorStore instance (received: {type(vector_store)})"
        raise TypeError(msg)
    return vector_store
