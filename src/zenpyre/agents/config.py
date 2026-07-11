r"""Contain a generic LLM agent configuration."""

from __future__ import annotations

__all__ = ["AgentConfig"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zenpyre.utils.config import ExtraFieldsConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from zenpyre.chat_models import BaseChatModelConfig


@dataclass(frozen=True)
class AgentConfig(ExtraFieldsConfig):
    r"""A generic LLM agent configuration.

    Subclass this to add provider- or agent-specific parameters as
    typed fields (e.g. ``max_tokens`` for OpenAI, ``top_k`` for
    Anthropic). Since this class is frozen, subclasses must also be
    frozen dataclasses. Additional fields do not need a default value:
    ``extra`` is declared keyword-only on
    :class:`~zenpyre.utils.config.ExtraFieldsConfig`, so it doesn't
    force subclass fields into a particular ordering.

    Fields added by subclasses are picked up automatically by
    :meth:`to_kwargs` (and therefore :meth:`cache_key`) via
    introspection; you don't need to override either method just to
    add a field.

    One thing subclasses *do* need to restate: ``@dataclass(frozen=True)``
    auto-generates a fresh ``__hash__`` for every dataclass-decorated
    class unless that class's own body defines ``__hash__`` — merely
    inheriting one does not suppress the override, and the
    auto-generated version would try to hash the unhashable ``extra``
    field. Any further subclass should include
    ``__hash__ = AgentConfig.__hash__`` in its own body, the same way
    this class restates ``ExtraFieldsConfig.__hash__``.

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
        AgentConfig(extra=mappingproxy({'max_tokens': 1024}), chat_model=ChatModelConfig(extra=mappingproxy({}), model='openai:gpt-4o'), system_prompt='You are helpful.')

        ```
    """

    chat_model: BaseChatModelConfig
    system_prompt: str

    # @dataclass(frozen=True) auto-generates its own __hash__ for THIS
    # class unless __hash__ is present in this class's own body — merely
    # inheriting one from ExtraFieldsConfig does not stop the override.
    # Restating it here (rather than repeating its logic) keeps the
    # single implementation in ExtraFieldsConfig authoritative.
    __hash__: Callable[[object], int] = ExtraFieldsConfig.__hash__

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
