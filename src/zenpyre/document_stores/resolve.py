r"""Provide a resolution utility for creating zenpyre BaseDocumentStore
models."""

from __future__ import annotations

__all__ = ["resolve_document_store"]

import logging
from typing import Any

from objectory import factory

from zenpyre.document_stores.base import BaseDocumentStore

logger: logging.Logger = logging.getLogger(__name__)


def resolve_document_store(
    document_store: BaseDocumentStore | dict[str, Any],
) -> BaseDocumentStore:
    """Resolve a
    :class:`~zenpyre.document_stores.base.BaseDocumentStore` instance
    from an existing object or a configuration dictionary.

    If ``document_store`` is already a
    :class:`~zenpyre.document_stores.base.BaseDocumentStore` instance
    it is returned as-is.  If it is a :class:`dict`, it is treated as
    an ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        document_store: Either a fully configured
            :class:`~zenpyre.document_stores.base.BaseDocumentStore`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured
        :class:`~zenpyre.document_stores.base.BaseDocumentStore`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~zenpyre.document_stores.base.BaseDocumentStore`
            instance.

    Example:
        ```pycon
        >>> from zenpyre.document_stores import InMemoryDocumentStore, resolve_document_store
        >>> # From an existing instance:
        >>> document_store = resolve_document_store(InMemoryDocumentStore())
        >>> # From a configuration dictionary:
        >>> document_store = resolve_document_store(
        ...     {"_target_": "zenpyre.document_stores.InMemoryDocumentStore"}
        ... )

        ```
    """
    if isinstance(document_store, dict):
        logger.info("Initializing a BaseDocumentStore instance from its configuration...")
        document_store = factory(**document_store)
    if not isinstance(document_store, BaseDocumentStore):
        msg = (
            f"Received object is not a BaseDocumentStore instance "
            f"(received: {type(document_store)})"
        )
        raise TypeError(msg)
    return document_store
