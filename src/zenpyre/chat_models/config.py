r"""Contain chat model configurations."""

from __future__ import annotations

__all__ = ["BaseChatModelConfig", "ChatModelConfig"]

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from zenpyre.utils.config import BaseConfig

if TYPE_CHECKING:
    from typing import Self


class BaseChatModelConfig(BaseConfig):
    """Define the interface for chat model configurations.

    Concrete subclasses only need to implement :meth:`to_kwargs`;
    :meth:`cache_key` is derived from it automatically.

    Example:
        ```pycon
        >>> from zenpyre.chat_models import ChatModelConfig
        >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
        >>> isinstance(cfg, BaseChatModelConfig)
        True

        ```
    """

    model: str


@dataclass(frozen=True)
class ChatModelConfig(BaseChatModelConfig):
    """A generic chat model configuration.

    Args:
        model: The model identifier.
        extra: Additional keyword arguments merged into
            :meth:`to_kwargs`. Must not contain a ``"model"`` key;
            use the ``model`` field for that.

    Example:
        ```pycon
        >>> from zenpyre.chat_models import ChatModelConfig
        >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
        >>> cfg.model
        'gpt-4'
        >>> cfg.to_kwargs()
        {'model': 'gpt-4', 'temperature': 0.2}

        ```
    """

    model: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "model" in self.extra:
            msg = (
                "'extra' must not contain a 'model' key: the 'model' field already "
                f"provides it (got extra={self.extra!r})."
            )
            raise ValueError(msg)
        # Wrap `extra` in a read-only view so mutating it after
        # construction raises, matching the `frozen=True` contract
        # (which otherwise only prevents reassigning the attribute
        # itself, not mutating the dict it points to).
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def to_kwargs(self) -> dict[str, Any]:
        """Return ``model`` merged with ``extra`` as a flat dict.

        Example:
            ```pycon
            >>> from zenpyre.chat_models import ChatModelConfig
            >>> cfg = ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})
            >>> cfg.to_kwargs()
            {'model': 'gpt-4', 'temperature': 0.2}

            ```
        """
        kwargs = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"
        }
        kwargs.update(self.extra)
        return kwargs

    @classmethod
    def from_kwargs(cls, model: str, **kwargs: Any) -> Self:
        """Construct a :class:`ChatModelConfig` from a model identifier
        and arbitrary keyword arguments.

        A convenience alternative to the regular constructor's
        ``extra={...}`` dict, letting callers pass extra fields
        directly as keyword arguments instead:
        ``ChatModelConfig.from_kwargs("gpt-4", temperature=0.2)`` is
        equivalent to
        ``ChatModelConfig(model="gpt-4", extra={"temperature": 0.2})``.

        Args:
            model: The model identifier.
            **kwargs: Additional keyword arguments, stored as
                ``extra``.

        Returns:
            A new :class:`ChatModelConfig`.

        Raises:
            TypeError: If ``kwargs`` contains a ``"model"`` key, since
                Python's own argument binding intercepts it as a
                duplicate value for the explicit ``model`` parameter
                before this method's body ever runs. For example,
                ``from_kwargs("gpt-4", **{"model": "x"})`` raises
                ``TypeError: got multiple values for argument
                'model'``. (This is distinct from the ``ValueError``
                the regular constructor raises for the same
                conceptual conflict when ``extra`` is passed directly
                as an already-built dict; that path isn't reachable
                through this method.)

        Example:
            ```pycon
            >>> from zenpyre.chat_models import ChatModelConfig
            >>> cfg = ChatModelConfig.from_kwargs("gpt-4", temperature=0.2)
            >>> cfg.to_kwargs()
            {'model': 'gpt-4', 'temperature': 0.2}

            ```
        """
        return cls(model=model, extra=kwargs)

    def __hash__(self) -> int:
        # The auto-generated dataclass __hash__ would hash `extra`
        # directly, which fails since dicts (and MappingProxyType) are
        # unhashable. Hashing the (string) cache key sidesteps that
        # while keeping equal configs hash-equal.
        return hash(self.cache_key())
