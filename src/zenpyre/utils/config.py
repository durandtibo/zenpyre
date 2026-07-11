"""Contain configurations."""

from __future__ import annotations

__all__ = ["BaseConfig", "ExtraFieldsConfig"]

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from coola.hashing import hash_object
from coola.recursive import recursive_apply


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

        Walks the ``to_kwargs()`` output and replaces any nested
        :class:`BaseConfig` instance with its own ``cache_key()`` string,
        then hashes the result via :func:`coola.hashing.hash_object`. The
        substitution step means two structurally-equal-but-distinct nested
        configs (e.g. two separately constructed ``ChatModelConfig``
        objects with the same fields) yield the same key, since the nested
        config contributes its content rather than its object identity.

        :func:`hash_object` canonicalizes its input (e.g. via sorted keys)
        before hashing, so two configs with identical (post-substitution)
        ``to_kwargs()`` output always produce the same hash regardless of
        field ordering.

        Note that this only covers whatever ``to_kwargs()`` returns. A
        subclass that adds a field must also include it in ``to_kwargs()``
        for it to affect the cache key.

        Useful for cache keys, output filenames, or detecting
        configuration changes between runs without comparing each field
        manually.

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

            Nested configs are resolved by content, not identity:

            ```pycon
            >>> class AgentConfig(BaseConfig):
            ...     def __init__(self, chat_model):
            ...         self.chat_model = chat_model
            ...     def to_kwargs(self):
            ...         return {"chat_model": self.chat_model}
            ...
            >>> a = AgentConfig(ChatModelConfig(model="gpt-4"))
            >>> b = AgentConfig(ChatModelConfig(model="gpt-4"))
            >>> a.cache_key() == b.cache_key()
            True

            ```
        """
        resolved_kwargs = recursive_apply(
            self.to_kwargs(), lambda x: x.cache_key() if isinstance(x, BaseConfig) else x
        )
        return hash_object(resolved_kwargs, length=length)


@dataclass(frozen=True, kw_only=True)
class ExtraFieldsConfig(BaseConfig):
    """Base for configs that merge arbitrary keyword arguments ("extra")
    into :meth:`to_kwargs`, alongside whatever typed fields a subclass
    declares.

    Subclasses just need to be frozen dataclasses with their own typed
    fields; :meth:`to_kwargs`, the extra/field-name collision check, and
    ``__hash__`` are all inherited from here and work automatically via
    introspection (:func:`dataclasses.fields`) — no per-subclass
    overriding required.

    ``extra`` is declared keyword-only (``kw_only=True`` on this class)
    specifically so that subclasses are free to add non-defaulted,
    positional-or-keyword fields of their own without hitting
    dataclass's "non-default argument follows default argument" error,
    which would otherwise apply because ``extra`` (with its default)
    is defined here in the base class.

    Attributes:
        extra: Additional keyword arguments merged into
            :meth:`to_kwargs`. Must not contain a key that collides
            with any of this config's own field names (including ones
            declared by a subclass).
    """

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = {f.name for f in dataclasses.fields(self) if f.name != "extra"}
        collisions = reserved & self.extra.keys()
        if collisions:
            msg = (
                f"'extra' must not contain any of this config's own field names "
                f"{sorted(reserved)}; got overlapping key(s) {sorted(collisions)} "
                f"in extra={dict(self.extra)!r}."
            )
            raise ValueError(msg)
        # Wrap `extra` in a read-only view so mutating it after
        # construction raises, matching the `frozen=True` contract
        # (which otherwise only prevents reassigning the attribute
        # itself, not mutating the dict it points to).
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_kwargs(self) -> dict[str, Any]:
        """Return every dataclass field (including ones declared by a
        subclass) merged with ``extra``, as a flat dict.

        Fields are collected via introspection
        (:func:`dataclasses.fields`) rather than hardcoded by name, so
        a subclass that adds a new typed field gets it included here
        automatically, with no need to override this method.

        Returns:
            A dict of keyword arguments representing this
                configuration.
        """
        kwargs = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"
        }
        kwargs.update(self.extra)
        return kwargs

    def __hash__(self) -> int:
        # The auto-generated dataclass __hash__ would hash `extra`
        # (and any nested BaseConfig field, unless it's independently
        # hashable) directly, which fails for an unhashable
        # dict/MappingProxyType. Hashing the (string) cache key
        # sidesteps that while keeping equal configs hash-equal.
        return hash(self.cache_key())
