r"""Contain a generic LLM agent configuration."""

from __future__ import annotations

__all__ = ["AgentConfig"]

from dataclasses import asdict, dataclass

from coola.hashing import hash_object


@dataclass(frozen=True)
class AgentConfig:
    r"""A generic LLM configuration.

    Subclass this to add provider-specific parameters as typed fields
    (e.g. ``max_tokens`` for OpenAI, ``top_k`` for Anthropic). Since
    this class is frozen, subclasses must also be frozen dataclasses,
    and any additional fields must declare a default value (dataclass
    field-ordering rules require defaulted fields to follow other
    defaulted fields).

    Attributes:
        model: The model identifier string passed to ``init_chat_model``
            (e.g. ``"openai:gpt-4o"`` or ``"ollama:gemma3:1b"``).
        system_prompt: The system prompt that instructs the LLM on its
            role and task.
        temperature: Sampling temperature passed to the LLM. Set to
            ``0.0`` for deterministic outputs. Defaults to ``0.0``.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from zenpyre.agents import AgentConfig
        >>> @dataclass(frozen=True)
        ... class OpenAIAgentConfig(AgentConfig):
        ...     max_tokens: int | None = None
        ...
        >>> config = OpenAIAgentConfig(
        ...     model="openai:gpt-4o",
        ...     system_prompt="You are helpful.",
        ...     max_tokens=1024,
        ... )
        >>> config.max_tokens
        1024

        ```
    """

    model: str
    system_prompt: str
    temperature: float = 0.0

    def cache_key(self) -> str:
        """Return a stable hash string derived from the current
        configuration.

        Serializes all attributes (including any fields added by
        subclasses) to a canonical form with sorted keys before
        hashing, ensuring that two configs with identical values always
        produce the same hash regardless of field ordering.

        Useful for cache keys, output filenames, or detecting
        configuration changes between runs without comparing each field
        manually.

        Returns:
            A stable hash string, 64 characters long.
        """
        return hash_object(asdict(self), length=64)
