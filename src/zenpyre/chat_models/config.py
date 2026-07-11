r"""Contain chat model configurations."""

from __future__ import annotations

__all__ = ["BaseChatModelConfig", "ChatModelConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zenpyre.utils.config import BaseConfig, ExtraFieldsConfig

if TYPE_CHECKING:
    from collections.abc import Callable
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
class ChatModelConfig(BaseChatModelConfig, ExtraFieldsConfig):
    """A generic chat model configuration.

    ``to_kwargs()``, the ``extra``/field-name collision check, and
    ``__hash__`` are all inherited from
    :class:`~zenpyre.utils.config.ExtraFieldsConfig`; this class only
    needs to declare its own typed field(s).

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

    # @dataclass(frozen=True) auto-generates its own __hash__ for THIS
    # class unless __hash__ is present in this class's own body — merely
    # inheriting one from ExtraFieldsConfig does not stop the override.
    # Restating it here (rather than repeating its logic) keeps the
    # single implementation in ExtraFieldsConfig authoritative.
    __hash__: Callable[[object], int] = ExtraFieldsConfig.__hash__

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
