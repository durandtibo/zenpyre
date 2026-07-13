r"""Provide a resolution utility for creating zenpyre BaseRecordStore
models."""

from __future__ import annotations

__all__ = ["resolve_record_store"]

import logging
from typing import Any

from objectory import factory

from zenpyre.record_stores.base import BaseRecordStore

logger: logging.Logger = logging.getLogger(__name__)


def resolve_record_store(
    record_store: BaseRecordStore | dict[str, Any],
) -> BaseRecordStore:
    """Resolve a :class:`~zenpyre.record_stores.base.BaseRecordStore`
    instance from an existing object or a configuration dictionary.

    If ``record_store`` is already a
    :class:`~zenpyre.record_stores.base.BaseRecordStore` instance
    it is returned as-is.  If it is a :class:`dict`, it is treated as
    an ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        record_store: Either a fully configured
            :class:`~zenpyre.record_stores.base.BaseRecordStore`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured
        :class:`~zenpyre.record_stores.base.BaseRecordStore`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~zenpyre.record_stores.base.BaseRecordStore`
            instance.

    Example:
        ```pycon
        >>> from zenpyre.record_stores import InMemoryRecordStore, resolve_record_store
        >>> # From an existing instance:
        >>> record_store = resolve_record_store(InMemoryRecordStore())
        >>> # From a configuration dictionary:
        >>> record_store = resolve_record_store(
        ...     {"_target_": "zenpyre.record_stores.InMemoryRecordStore"}
        ... )

        ```
    """
    if isinstance(record_store, dict):
        logger.info("Initializing a BaseRecordStore instance from its configuration...")
        record_store = factory(**record_store)
    if not isinstance(record_store, BaseRecordStore):
        msg = f"Received object is not a BaseRecordStore instance (received: {type(record_store)})"
        raise TypeError(msg)
    return record_store
