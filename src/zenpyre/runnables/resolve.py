r"""Provide a resolution utility for creating LangChain Runnable
models."""

from __future__ import annotations

__all__ = ["resolve_runnable"]

import logging
from typing import Any

from langchain_core.runnables import Runnable
from objectory import factory

logger: logging.Logger = logging.getLogger(__name__)


def resolve_runnable(
    runnable: Runnable[Any, Any] | dict[str, Any],
) -> Runnable[Any, Any]:
    """Resolve a LangChain :class:`~langchain_core.runnables.Runnable`
    instance from an existing object or a configuration dictionary.

    If ``runnable`` is already a
    :class:`~langchain_core.runnables.Runnable` instance it is
    returned as-is.  If it is a :class:`dict`, it is treated as an
    ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        runnable: Either a fully configured
            :class:`~langchain_core.runnables.Runnable`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured :class:`~langchain_core.runnables.Runnable`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_core.runnables.Runnable`
            instance.

    Example:
        ```pycon
        >>> from langchain_core.runnables import Runnable
        >>> from zenpyre.runnables import resolve_runnable
        >>> class MyRunnable(Runnable):
        ...     def invoke(self, input, config=None):
        ...         return input
        ...
        >>> # From an existing instance:
        >>> runnable = resolve_runnable(MyRunnable())
        >>> # From a configuration dictionary:
        >>> runnable = resolve_runnable(
        ...     {"_target_": "langchain_core.runnables.RunnablePassthrough"}
        ... )

        ```
    """
    if isinstance(runnable, dict):
        logger.info("Initializing a Runnable instance from its configuration...")
        runnable = factory(**runnable)
    if not isinstance(runnable, Runnable):
        msg = f"Received object is not a Runnable instance (received: {type(runnable)})"
        raise TypeError(msg)
    return runnable
