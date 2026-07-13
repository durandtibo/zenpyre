r"""Provide a resolution utility for creating zenpyre BaseProcessor
models."""

from __future__ import annotations

__all__ = ["resolve_data_processor"]

import logging
from typing import Any, TypeVar

from objectory import factory

from zenpyre.data_processors.base import BaseProcessor

logger: logging.Logger = logging.getLogger(__name__)


T = TypeVar("T")
U = TypeVar("U")


def resolve_data_processor(
    processor: BaseProcessor[U, T] | dict[str, Any],
) -> BaseProcessor[U, T]:
    """Resolve a :class:`~zenpyre.data_processors.base.BaseProcessor`
    instance from an existing object or a configuration dictionary.

    If ``processor`` is already a
    :class:`~zenpyre.data_processors.base.BaseProcessor` instance
    it is returned as-is.  If it is a :class:`dict`, it is treated as
    an ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        processor: Either a fully configured
            :class:`~zenpyre.data_processors.base.BaseProcessor`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured
        :class:`~zenpyre.data_processors.base.BaseProcessor`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~zenpyre.data_processors.base.BaseProcessor`
            instance.

    Example:
        ```pycon
        >>> from zenpyre.data_processors import resolve_data_processor, FirstNProcessor
        >>> # From an existing instance:
        >>> processor = resolve_data_processor(FirstNProcessor(n=5))
        >>> # From a configuration dictionary:
        >>> processor = resolve_data_processor(  # doctest: +SKIP
        ...     {"_target_": "zenpyre.data_processors.FirstNProcessor", "n": 5}
        ... )

        ```
    """
    if isinstance(processor, dict):
        logger.info("Initializing a BaseProcessor instance from its configuration...")
        processor = factory(**processor)
    if not isinstance(processor, BaseProcessor):
        msg = f"Received object is not a BaseProcessor instance (received: {type(processor)})"
        raise TypeError(msg)
    return processor
