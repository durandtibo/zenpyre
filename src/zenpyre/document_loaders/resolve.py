r"""Provide a resolution utility for creating LangChain document loader
models."""

from __future__ import annotations

__all__ = ["resolve_document_loader"]

import logging
from typing import Any

from langchain_core.document_loaders import BaseLoader
from objectory import factory

logger: logging.Logger = logging.getLogger(__name__)


def resolve_document_loader(
    document_loader: BaseLoader | dict[str, Any],
) -> BaseLoader:
    """Resolve a LangChain
    :class:`~langchain_core.document_loaders.BaseLoader` instance from
    an existing object or a configuration dictionary.

    If ``document_loader`` is already a
    :class:`~langchain_core.document_loaders.BaseLoader` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        document_loader: Either a fully configured
            :class:`~langchain_core.document_loaders.BaseLoader`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured :class:`~langchain_core.document_loaders.BaseLoader`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_core.document_loaders.BaseLoader`
            instance.

    Example:
        ```pycon
        >>> from langchain_core.document_loaders import BaseLoader
        >>> from zenpyre.document_loaders import resolve_document_loader
        >>> class MyLoader(BaseLoader):
        ...     def lazy_load(self):
        ...         return iter([])
        ...
        >>> # From an existing instance:
        >>> document_loader = resolve_document_loader(MyLoader())
        >>> # From a configuration dictionary:
        >>> document_loader = resolve_document_loader(  # doctest: +SKIP
        ...     {
        ...         "_target_": "langchain_core.document_loaders.LangSmithLoader",
        ...         "project_name": "my-project",
        ...     }
        ... )

        ```
    """
    if isinstance(document_loader, dict):
        logger.info("Initializing a BaseLoader instance from its configuration...")
        document_loader = factory(**document_loader)
    if not isinstance(document_loader, BaseLoader):
        msg = f"Received object is not a BaseLoader instance (received: {type(document_loader)})"
        raise TypeError(msg)
    return document_loader
