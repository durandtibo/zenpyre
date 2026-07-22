r"""Provide a caching chat model wrapper."""

from __future__ import annotations

__all__ = ["CachingChatModel"]

import asyncio
import logging
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Any

from coola.hashing import hash_object
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage  # noqa: TC002
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable  # noqa: TC002
from langchain_core.tools import BaseTool  # noqa: TC002
from persista.cache import Cache
from pydantic import ConfigDict

from zenpyre.runnables import hashing as _hashing  # noqa: F401
from zenpyre.utils.imports import check_persista

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

logger: logging.Logger = logging.getLogger(__name__)


class CachingChatModel(BaseChatModel):
    r"""Wrap a chat model to cache its output, keyed by a hash of the
    messages, stop words, and call kwargs.

    Unlike :class:`~zenpyre.runnables.CachingRunnable`, this class is a
    genuine :class:`~langchain_core.language_models.BaseChatModel`
    subclass -- it can be used anywhere a chat model is expected, and
    ``bind_tools`` returns another ``CachingChatModel`` wrapping the
    bound inner model, so caching keeps working after tools are bound.

    On each call, a cache key is derived from ``(messages, stop,
    kwargs)`` and used to look up a previously cached ``ChatResult`` in
    ``result_cache``. On a cache hit, the cached result is returned
    without calling the wrapped chat model. On a cache miss, the
    wrapped chat model is invoked and its result is stored in
    ``result_cache`` before being returned. If ``result_cache`` is
    ``None``, caching is disabled entirely and every call goes straight
    to the wrapped chat model.

    A failure to read or write a cache entry (backing store error,
    serialization error, etc.) is logged as a warning and treated as a
    cache miss; it never raises and never prevents the wrapped chat
    model from producing a result.

    Args:
        chat_model: The chat model whose output should be cached.
        result_cache: The :class:`persista.cache.Cache` used to store
            cached results. If ``None``, caching is disabled. Typed as
            ``Any`` (rather than importing ``persista.cache.Cache``
            unconditionally) since ``persista`` is an optional
            dependency; :func:`~zenpyre.utils.imports.check_persista`
            is invoked instead whenever ``result_cache`` is used. Named
            ``result_cache`` rather than ``cache`` to avoid shadowing
            :class:`~langchain_core.language_models.BaseChatModel`'s
            own built-in ``cache`` field, which controls an unrelated
            LangChain caching mechanism.
        key_fn: A function that derives a cache key from a
            ``(messages, stop, kwargs)`` tuple. The returned string is
            used as the key under ``result_cache``. Defaults to
            ``hash_object``, which dispatches through ``coola``'s
            hasher registry (e.g. using ``SerializableHasher`` for
            LangChain messages).
        ignore_none: If ``True``, a ``None`` result is never written to
            the cache. This is useful when the wrapped chat model can
            return ``None`` to mean "no result yet" or "failed
            silently," and you don't want that non-result to be cached
            as if it were a real answer. Note that
            :meth:`persista.cache.Cache.get` cannot distinguish a
            cached ``None`` from a missing key, so on the read side a
            ``None`` is always treated as a cache miss regardless of
            this flag.

    Example:
        ```pycon
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from persista.cache import Cache
        >>> from zenpyre.chat_models import CachingChatModel
        >>> chat_model = CachingChatModel(
        ...     chat_model=FakeListChatModel(responses=["hello"]),
        ...     result_cache=Cache(),
        ... )
        >>> chat_model.invoke("hi").content
        'hello'

        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat_model: Runnable[Any, BaseMessage]
    cache: Cache
    key_fn: Callable[[Any], str] | None = None

    @property
    def _llm_type(self) -> str:
        inner = getattr(self.chat_model, "_llm_type", self.chat_model.__class__.__qualname__)
        return f"caching-{inner}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        inner = getattr(self.chat_model, "_identifying_params", {})
        return {"chat_model": dict(inner), "result_cache": self.result_cache is not None}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.result_cache is None:
            return self._call_chat_model(messages, stop=stop, run_manager=run_manager, **kwargs)

        check_persista()
        key = self._cache_key(messages, stop, kwargs)
        hit, result = self._load_cached(key)
        if hit:
            return result

        logger.info("Cache miss: %s", key)
        result = self._call_chat_model(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._save_cache(result, key)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.result_cache is None:
            return await self._acall_chat_model(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        check_persista()
        key = self._cache_key(messages, stop, kwargs)
        hit, result = await self._aload_cached(key)
        if hit:
            return result

        logger.info("Cache miss: %s", key)
        result = await self._acall_chat_model(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        await self._asave_cache(result, key)
        return result

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, BaseMessage]:
        bound = self.chat_model.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return self.model_copy(update={"chat_model": bound})

    def _call_chat_model(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        run_manager: CallbackManagerForLLMRun | None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        message = self.chat_model.invoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _acall_chat_model(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        run_manager: AsyncCallbackManagerForLLMRun | None,  # noqa: ARG002
        **kwargs: Any,
    ) -> ChatResult:
        message = await self.chat_model.ainvoke(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _cache_key(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> str:
        key_fn = self.key_fn if self.key_fn is not None else hash_object
        return key_fn((messages, stop, kwargs))

    def _load_cached(self, key: str) -> tuple[bool, ChatResult | None]:
        try:
            result = self.result_cache.get(key)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", key, exc_info=True)
            return False, None
        if result is None:
            return False, None
        logger.info("Cache hit: %s", key)
        return True, result

    async def _aload_cached(self, key: str) -> tuple[bool, ChatResult | None]:
        try:
            result = await asyncio.to_thread(self.result_cache.get, key)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", key, exc_info=True)
            return False, None
        if result is None:
            return False, None
        logger.info("Cache hit: %s", key)
        return True, result

    def _save_cache(self, result: ChatResult | None, key: str) -> None:
        if self.ignore_none and result is None:
            logger.info("Not caching None result: %s", key)
            return
        try:
            self.result_cache.set(key, result)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", key, exc_info=True)

    async def _asave_cache(self, result: ChatResult | None, key: str) -> None:
        if self.ignore_none and result is None:
            logger.info("Not caching None result: %s", key)
            return
        try:
            await asyncio.to_thread(self.result_cache.set, key, result)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", key, exc_info=True)
