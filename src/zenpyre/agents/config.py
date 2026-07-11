r"""Contain a generic LLM agent configuration."""

from __future__ import annotations

__all__ = ["AgentConfig"]

import dataclasses
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from zenpyre.utils.config import BaseConfig

if TYPE_CHECKING:
    from typing import Self

    from zenpyre.chat_models import BaseChatModelConfig


@dataclass(frozen=True)
class AgentConfig(BaseConfig):
    r"""A generic LLM agent configuration.

    Subclass this to add provider- or agent-specific parameters as
    typed fields (e.g. ``max_tokens`` for OpenAI, ``top_k`` for
    Anthropic). Since this class is frozen, subclasses must also be
    frozen dataclasses, and any additional fields must declare a
    default value (dataclass field-ordering rules require defaulted
    fields to follow other defaulted fields).

    Fields added by subclasses are picked up automatically by
    :meth:`to_kwargs` (and therefore :meth:`cache_key`) via
    introspection; you don't need to override either method just to
    add a field.

    Attributes:
        chat_model: The chat model configuration (see
            :class:`~zenpyre.chat_models.BaseChatModelConfig`) used by
            this agent.
        system_prompt: The system prompt that instructs the LLM on its
            role and task.
        extra: Additional keyword arguments merged into
            :meth:`to_kwargs`. Must not contain a key that collides
            with any of this config's own field names (including ones
            declared by a subclass).

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from zenpyre.agents import AgentConfig
        >>> from zenpyre.chat_models import ChatModelConfig
        >>> config = AgentConfig.from_kwargs(
        ...     chat_model=ChatModelConfig(model="openai:gpt-4o"),
        ...     system_prompt="You are helpful.",
        ...     max_tokens=1024,
        ... )
        >>> config
        AgentConfig(chat_model=ChatModelConfig(model='openai:gpt-4o', extra=mappingproxy({})), system_prompt='You are helpful.', extra=mappingproxy({'max_tokens': 1024}))

        ```
    """

    chat_model: BaseChatModelConfig
    system_prompt: str
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
        a subclass that adds a new typed field (e.g. ``max_tokens``)
        gets it included here automatically, with no need to override
        this method.

        Note that ``chat_model`` is included as the actual
        :class:`~zenpyre.chat_models.BaseChatModelConfig` instance,
        not a hash string; :meth:`cache_key` is the one that converts
        it to a stable string for hashing purposes.

        Example:
            ```pycon
            >>> from zenpyre.agents import AgentConfig
            >>> from zenpyre.chat_models import ChatModelConfig
            >>> cfg = AgentConfig(
            ...     chat_model=ChatModelConfig(model="gpt-4"),
            ...     system_prompt="You are helpful.",
            ...     extra={"max_retries": 3},
            ... )
            >>> kwargs = cfg.to_kwargs()
            >>> kwargs["system_prompt"]
            'You are helpful.'
            >>> kwargs["max_retries"]
            3

            ```
        """
        kwargs = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"
        }
        kwargs.update(self.extra)
        return kwargs

    @classmethod
    def from_kwargs(
        cls, chat_model: BaseChatModelConfig, system_prompt: str, **kwargs: Any
    ) -> Self:
        """Construct an :class:`AgentConfig` from a chat model
        configuration, a system prompt, and arbitrary keyword arguments.

        A convenience alternative to the regular constructor's
        ``extra={...}`` dict, letting callers pass extra fields
        directly as keyword arguments instead:
        ``AgentConfig.from_kwargs(chat_model, "You are helpful.", max_retries=3)``
        is equivalent to
        ``AgentConfig(chat_model=chat_model, system_prompt="You are helpful.", extra={"max_retries": 3})``.

        Args:
            chat_model: The chat model configuration used by this
                agent.
            system_prompt: The system prompt that instructs the LLM on
                its role and task.
            **kwargs: Additional keyword arguments, stored as
                ``extra``.

        Returns:
            A new :class:`AgentConfig`.

        Raises:
            TypeError: If ``kwargs`` contains a ``"chat_model"`` or
                ``"system_prompt"`` key, since Python's own argument
                binding intercepts it as a duplicate value for the
                corresponding explicit parameter before this method's
                body ever runs. For example,
                ``from_kwargs(cm, "prompt", **{"chat_model": other})``
                raises ``TypeError: got multiple values for argument
                'chat_model'``. (This is distinct from the
                ``ValueError`` the regular constructor raises for the
                same conceptual conflict when ``extra`` is passed
                directly as an already-built dict; that path isn't
                reachable through this method.)

        Example:
            ```pycon
            >>> from zenpyre.agents import AgentConfig
            >>> from zenpyre.chat_models import ChatModelConfig
            >>> cfg = AgentConfig.from_kwargs(
            ...     ChatModelConfig(model="gpt-4"), "You are helpful.", max_retries=3
            ... )
            >>> cfg.to_kwargs()["max_retries"]
            3

            ```
        """
        return cls(chat_model=chat_model, system_prompt=system_prompt, extra=kwargs)

    def __hash__(self) -> int:
        # The auto-generated dataclass __hash__ would hash `extra`
        # (and `chat_model`, unless it's independently hashable)
        # directly, which fails for an unhashable dict/MappingProxyType.
        # Hashing the (string) cache key sidesteps that while keeping
        # equal configs hash-equal.
        return hash(self.cache_key())
