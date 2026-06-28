r"""Provide a resolution utility for creating LangChain text splitter
models."""

from __future__ import annotations

__all__ = ["resolve_text_splitter"]

import logging
from typing import TYPE_CHECKING, Any

from objectory import factory

from zenpyre.utils.imports import is_langchain_text_splitters_available

if TYPE_CHECKING or is_langchain_text_splitters_available():
    from langchain_text_splitters import TextSplitter
else:  # pragma: no cover
    from zenpyre.utils.fallback.langchain_text_splitters import TextSplitter

logger: logging.Logger = logging.getLogger(__name__)


def resolve_text_splitter(text_splitter: TextSplitter | dict[str, Any]) -> TextSplitter:
    """Resolve a LangChain
    :class:`~langchain_text_splitters.TextSplitter` instance from an
    existing object or a configuration dictionary.

    If ``text_splitter`` is already a
    :class:`~langchain_text_splitters.TextSplitter` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        text_splitter: Either a fully configured
            :class:`~langchain_text_splitters.TextSplitter` instance, or
            a :class:`dict` containing an ``objectory`` factory
            specification (must include a ``"_target_"`` key pointing to
            the fully-qualified class name).

    Returns:
        A configured :class:`~langchain_text_splitters.TextSplitter`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_text_splitters.TextSplitter` instance.

    Example:
        ```pycon
        >>> from zenpyre.text_splitters import resolve_text_splitter
        >>> from langchain_text_splitters import CharacterTextSplitter
        >>> # From an existing instance:
        >>> text_splitter = resolve_text_splitter(CharacterTextSplitter())
        >>> # From a configuration dictionary:
        >>> text_splitter = resolve_text_splitter(
        ...     {"_target_": "langchain_text_splitters.CharacterTextSplitter"}
        ... )

        ```
    """
    if isinstance(text_splitter, dict):
        logger.info("Initializing a TextSplitter instance from its configuration...")
        text_splitter = factory(**text_splitter)
    if not isinstance(text_splitter, TextSplitter):
        msg = f"Received object is not a TextSplitter instance (received: {type(text_splitter)})"
        raise TypeError(msg)
    return text_splitter
