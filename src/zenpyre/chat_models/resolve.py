r"""Provide a resolution utility for creating LangChain BaseChatModel
models."""

from __future__ import annotations

__all__ = ["resolve_chat_model"]

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from objectory import factory

logger: logging.Logger = logging.getLogger(__name__)


def resolve_chat_model(
    chat_model: BaseChatModel | dict[str, Any],
) -> BaseChatModel:
    """Resolve a LangChain
    :class:`~langchain_core.language_models.BaseChatModel` instance from
    an existing object or a configuration dictionary.

    If ``chat_model`` is already a
    :class:`~langchain_core.language_models.BaseChatModel` instance
    it is returned as-is.  If it is a :class:`dict`, it is treated as
    an ``objectory`` factory configuration and instantiated via
    :func:`objectory.factory`.

    Args:
        chat_model: Either a fully configured
            :class:`~langchain_core.language_models.BaseChatModel`
            instance, or a :class:`dict` containing an ``objectory``
            factory specification (must include a ``"_target_"`` key
            pointing to the fully-qualified class name).

    Returns:
        A configured
        :class:`~langchain_core.language_models.BaseChatModel`
        instance.

    Raises:
        TypeError: If the resolved object is not a
            :class:`~langchain_core.language_models.BaseChatModel`
            instance.

    Example:
        ```pycon
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from zenpyre.chat_models import resolve_chat_model
        >>> # From an existing instance:
        >>> chat_model = resolve_chat_model(FakeListChatModel(responses=["hello"]))
        >>> # From a configuration dictionary:
        >>> chat_model = resolve_chat_model(  # doctest: +SKIP
        ...     {
        ...         "_target_": "langchain_core.language_models.FakeListChatModel",
        ...         "responses": ["hello"],
        ...     }
        ... )

        ```
    """
    if isinstance(chat_model, dict):
        logger.info("Initializing a BaseChatModel instance from its configuration...")
        chat_model = factory(**chat_model)
    if not isinstance(chat_model, BaseChatModel):
        msg = f"Received object is not a BaseChatModel instance (received: {type(chat_model)})"
        raise TypeError(msg)
    return chat_model
