r"""Provide a resolution utility for creating zenpyre BaseIngestor
models."""

from __future__ import annotations

__all__ = ["resolve_ingestor"]

import logging
from typing import Any, TypeVar

from objectory import factory

from zenpyre.ingestors.base import BaseIngestor

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def resolve_ingestor(
    ingestor: BaseIngestor[T] | dict[str, Any],
) -> BaseIngestor[T]:
    """Resolve a :class:`~zenpyre.ingestors.base.BaseIngestor` instance
    from an existing object or a configuration dictionary.

    If ``ingestor`` is already a
    :class:`~zenpyre.ingestors.base.BaseIngestor` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        ingestor: Either a fully configured
            :class:`~zenpyre.ingestors.base.BaseIngestor`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured :class:`~zenpyre.ingestors.base.BaseIngestor`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~zenpyre.ingestors.base.BaseIngestor`
            instance.

    Example:
        ```pycon
        >>> from zenpyre.ingestors import resolve_ingestor
        >>> from zenpyre.ingestors.base import BaseIngestor
        >>> class MyIngestor(BaseIngestor):
        ...     def ingest(self) -> Any:
        ...         return {"hello": "world"}
        ...
        >>> # From an existing instance:
        >>> ingestor = resolve_ingestor(MyIngestor())
        >>> # From a configuration dictionary:
        >>> ingestor = resolve_ingestor(  # doctest: +SKIP
        ...     {"_target_": "my_package.ingestors.MyIngestor"}
        ... )

        ```
    """
    if isinstance(ingestor, dict):
        logger.info("Initializing a BaseIngestor instance from its configuration...")
        ingestor = factory(**ingestor)
    if not isinstance(ingestor, BaseIngestor):
        msg = f"Received object is not a BaseIngestor instance (received: {type(ingestor)})"
        raise TypeError(msg)
    return ingestor
