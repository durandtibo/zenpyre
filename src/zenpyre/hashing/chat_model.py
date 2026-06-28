r"""Provide deterministic hashing utility functions for chat models."""

from __future__ import annotations

__all__ = ["hash_chat_model"]

from typing import TYPE_CHECKING, Any

from coola.hashing import hash_object
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_CHAT_MODEL_EXCLUDED_FIELDS: frozenset[str] = frozenset({"default_headers", "default_query"})


def hash_chat_model(model: BaseChatModel, length: int = 64) -> str:
    """Compute a stable, reproducible hash of a LangChain chat model's
    configuration.

    Serialises the model's full Pydantic configuration via
    :meth:`~pydantic.BaseModel.model_dump`, then removes fields that
    would make the hash environment-dependent or non-reproducible:

    - :class:`~pydantic.SecretStr` fields (API keys, tokens) — replaced
      with the placeholder ``"<secret>"`` so the hash is independent of
      credentials while still tracking whether the field is present.
    - Fields in :data:`_CHAT_MODEL_EXCLUDED_FIELDS` (``default_headers``,
      ``default_query``) — transport-layer overrides that do not affect
      model behaviour.

    The model's fully-qualified class name (including module) is included
    in the hash to distinguish providers that share field names (e.g.
    ``ChatAnthropic`` and ``ChatOpenAI`` both expose a ``model`` field).

    Args:
        model: Any LangChain :class:`~langchain_core.language_models.BaseChatModel`
            instance.
        length: The desired length of the returned hex string. Must be an
            even number between 2 and 128 inclusive. Defaults to 64.

    Returns:
        A lowercase hexadecimal string of exactly ``length`` characters
        that uniquely identifies the model's configuration.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and 128
            inclusive.

    Example:
        ```pycon
        >>> from zenpyre.hashing import hash_chat_model
        >>> from langchain_ollama import ChatOllama
        >>> model = ChatOllama(model="gemma3:4b", temperature=0)
        >>> hash_chat_model(model)  # doctest: +SKIP
        'a3f2...'

        ```
    """
    config: dict[str, Any] = {
        k: "<secret>" if isinstance(v, SecretStr) else v
        for k, v in model.model_dump().items()
        if k not in _CHAT_MODEL_EXCLUDED_FIELDS
    }
    config["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
    return hash_object(config, length=length)
