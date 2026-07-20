r"""Provide a caching chat model wrapper."""

from __future__ import annotations

__all__ = ["CachingChatModel"]

import asyncio
import logging
from collections.abc import Callable  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any

from coola.hashing import hash_object
from coola.utils.path import sanitize_path
from iden.io import load_pickle, save_pickle
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage  # noqa: TC002
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable  # noqa: TC002
from langchain_core.tools import BaseTool  # noqa: TC002
from pydantic import ConfigDict

from zenpyre.runnables import hashing as _hashing  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )

logger: logging.Logger = logging.getLogger(__name__)


class CachingChatModel(BaseChatModel):
    r"""Wrap a chat model to cache its output to disk, keyed by a hash of
    the messages, stop words, and call kwargs.

    Unlike :class:`~zenpyre.runnables.CachingRunnable`, this class is a
    genuine :class:`~langchain_core.language_models.BaseChatModel`
    subclass -- it can be used anywhere a chat model is expected, and
    ``bind_tools`` returns another ``CachingChatModel`` wrapping the
    bound inner model, so caching keeps working after tools are bound.

    On each call, a cache key is derived from ``(messages, stop,
    kwargs)`` and used to look up a previously pickled ``ChatResult``
    under ``cache_dir``. On a cache hit, the pickled result is returned
    without calling the wrapped chat model. On a cache miss, the
    wrapped chat model is invoked and its result is pickled to disk
    before being returned. If ``cache_dir`` is ``None``, caching is
    disabled entirely and every call goes straight to the wrapped chat
    model.

    A failure to read or write a cache file (corrupt pickle, permission
    error, disk full, etc.) is logged as a warning and treated as a
    cache miss; it never raises and never prevents the wrapped chat
    model from producing a result.

    Args:
        chat_model: The chat model whose output should be cached.
        cache_dir: The directory used to store cached results. If
            ``None``, caching is disabled.
        key_fn: A function that derives a cache key from a
            ``(messages, stop, kwargs)`` tuple. The returned string is
            used as a filename (with a ``.pkl`` extension appended)
            under ``cache_dir``, so it must be filesystem-safe -- in
            particular, it must not contain path separators. Defaults
            to ``hash_object``, which dispatches through ``coola``'s
            hasher registry (e.g. using ``SerializableHasher`` for
            LangChain messages).
        ignore_none: If ``False`` (the default), ``None`` is treated as
            an ordinary cacheable value. If ``True``, ``None`` is
            treated as if there were no cache entry at all: a cached
            ``None`` is treated as a cache miss, and a ``None`` result
            is never written to disk. This is useful when the wrapped
            chat model can return ``None`` to mean "no result yet" or
            "failed silently," and you don't want that non-result to be
            cached as if it were a real answer.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from zenpyre.chat_models import CachingChatModel
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     chat_model = CachingChatModel(
        ...         chat_model=FakeListChatModel(responses=["hello"]),
        ...         cache_dir=Path(tmpdir),
        ...     )
        ...     chat_model.invoke("hi").content
        ...
        'hello'

        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat_model: Runnable[Any, BaseMessage]
    cache_dir: Path | str | None = None
    key_fn: Callable[[Any], str] | None = None
    ignore_none: bool = False

    @property
    def _llm_type(self) -> str:
        inner = getattr(self.chat_model, "_llm_type", self.chat_model.__class__.__qualname__)
        return f"caching-{inner}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        inner = getattr(self.chat_model, "_identifying_params", {})
        return {"chat_model": dict(inner), "cache_dir": self.cache_dir}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cache_dir = sanitize_path(self.cache_dir) if self.cache_dir is not None else None
        if cache_dir is None:
            return self._call_chat_model(messages, stop=stop, run_manager=run_manager, **kwargs)

        filepath = self._cache_path(cache_dir, messages, stop, kwargs)
        hit, result = self._load_cached(filepath)
        if hit:
            return result

        logger.info("Cache miss: %s", filepath)
        result = self._call_chat_model(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._save_cache(result, filepath)
        return result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        cache_dir = sanitize_path(self.cache_dir) if self.cache_dir is not None else None
        if cache_dir is None:
            return await self._acall_chat_model(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        filepath = self._cache_path(cache_dir, messages, stop, kwargs)
        hit, result = await self._aload_cached(filepath)
        if hit:
            return result

        logger.info("Cache miss: %s", filepath)
        result = await self._acall_chat_model(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        await self._asave_cache(result, filepath)
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

    def _cache_path(
        self,
        cache_dir: Path,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> Path:
        key_fn = self.key_fn if self.key_fn is not None else hash_object
        key = key_fn((messages, stop, kwargs))
        return cache_dir / f"{key}.pkl"

    def _load_cached(self, filepath: Path) -> tuple[bool, ChatResult | None]:
        if not filepath.is_file():
            return False, None
        try:
            result = load_pickle(filepath)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", filepath, exc_info=True)
            return False, None
        if self.ignore_none and result is None:
            logger.info("Ignoring cached None: %s", filepath)
            return False, None
        logger.info("Cache hit: %s", filepath)
        return True, result

    async def _aload_cached(self, filepath: Path) -> tuple[bool, ChatResult | None]:
        if not filepath.is_file():
            return False, None
        try:
            result = await asyncio.to_thread(load_pickle, filepath)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", filepath, exc_info=True)
            return False, None
        if self.ignore_none and result is None:
            logger.info("Ignoring cached None: %s", filepath)
            return False, None
        logger.info("Cache hit: %s", filepath)
        return True, result

    def _save_cache(self, result: ChatResult | None, filepath: Path) -> None:
        if self.ignore_none and result is None:
            logger.info("Not caching None result: %s", filepath)
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(result, filepath, exist_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", filepath, exc_info=True)

    async def _asave_cache(self, result: ChatResult | None, filepath: Path) -> None:
        if self.ignore_none and result is None:
            logger.info("Not caching None result: %s", filepath)
            return

        def _write() -> None:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(result, filepath, exist_ok=True)

        try:
            await asyncio.to_thread(_write)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", filepath, exc_info=True)
