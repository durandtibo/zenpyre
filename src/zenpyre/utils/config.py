"""Contain configurations."""

from __future__ import annotations

__all__ = ["BaseConfig"]

from abc import ABC, abstractmethod
from typing import Any

from coola.hashing import hash_object


class BaseConfig(ABC):
    """Define the interface for chat model configurations.

    Concrete subclasses only need to implement :meth:`to_kwargs`;
    :meth:`cache_key` is derived from it automatically.

    Example:
        ```pycon
        >>> from zenpyre.utils.config import BaseConfig
        >>> class ChatModelConfig(BaseConfig):
        ...     def __init__(self, model, extra=None):
        ...         self.model = model
        ...         self.extra = extra or {}
        ...     def to_kwargs(self):
        ...         return {"model": self.model, **self.extra}
        ...
        >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
        >>> isinstance(cfg, BaseConfig)
        True

        ```
    """

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """Return the configuration as a flat dict of keyword arguments.

        This is the single source of truth for the configuration's
        content: it is used both to construct/invoke the chat model
        and, via :meth:`cache_key`, to derive a stable hash. Any field
        that should affect caching or equality must be included here.

        Returns:
            A dict of keyword arguments representing this
                configuration.

        Example:
            ```pycon
            >>> from zenpyre.utils.config import BaseConfig
            >>> class ChatModelConfig(BaseConfig):
            ...     def __init__(self, model, extra=None):
            ...         self.model = model
            ...         self.extra = extra or {}
            ...     def to_kwargs(self):
            ...         return {"model": self.model, **self.extra}
            ...
            >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
            >>> cfg.to_kwargs()
            {'model': 'gpt-4', 'temperature': 0.2}

            ```
        """

    def cache_key(self, length: int = 64) -> str:
        """Return a stable hash string derived from the current
        configuration.

        Delegates to :func:`coola.hashing.hash_object`, which
        canonicalizes the input (e.g. via sorted keys) before hashing,
        so two configs with identical ``to_kwargs()`` output always
        produce the same hash regardless of field ordering. Note that
        this only covers whatever ``to_kwargs()`` returns. A subclass
        that adds a field must also include it in ``to_kwargs()`` for
        it to affect the cache key.

        Useful for cache keys, output filenames, or detecting
        configuration changes between runs without comparing each
        field manually.

        Args:
            length: The desired length of the returned hash string.

        Returns:
            A stable hash string, ``length`` characters long.

        Example:
            ```pycon
            >>> from zenpyre.utils.config import BaseConfig
            >>> class ChatModelConfig(BaseConfig):
            ...     def __init__(self, model, extra=None):
            ...         self.model = model
            ...         self.extra = extra or {}
            ...     def to_kwargs(self):
            ...         return {"model": self.model, **self.extra}
            ...
            >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
            >>> key = cfg.cache_key()
            >>> len(key)
            64
            >>> cfg.cache_key() == cfg.cache_key()
            True

            ```
        """
        return hash_object(self.to_kwargs(), length=length)
