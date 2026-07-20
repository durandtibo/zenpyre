r"""Provide a concrete chat model factory that wraps another chat model
factory with caching."""

from __future__ import annotations

__all__ = ["CachingChatModelFactory"]

from typing import TYPE_CHECKING, Any, cast

from coola.display import MultilineDisplayMixin

from zenpyre.chat_models.factory.base import BaseChatModelFactory
from zenpyre.runnables import CachingRunnable
from zenpyre.utils.resolve import resolve_object

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

    from zenpyre.utils.config import BaseConfig


class CachingChatModelFactory(BaseChatModelFactory, MultilineDisplayMixin):
    """A concrete chat model factory that wraps another chat model
    factory and caches the resulting model's outputs via
    :class:`~zenpyre.runnables.CachingRunnable`.

    The wrapped chat model is built by delegating to the inner
    factory, then wrapped in a
    :class:`~zenpyre.runnables.CachingRunnable` so every call to the
    model transparently reads from and writes to ``cache_dir``. Note
    that the returned object is a ``CachingRunnable``, not a
    :class:`~langchain_core.language_models.BaseChatModel` -- it
    exposes the ``Runnable`` interface (``invoke``, ``batch``, and
    their async counterparts) used to call the chat model, but not
    ``BaseChatModel``-specific methods such as ``bind_tools`` or
    ``with_structured_output``. Bind tools / structured output on the
    inner chat model before wrapping it with this factory if needed.

    Args:
        chat_model_factory: The wrapped chat model factory, an
            objectory configuration, or a :class:`~BaseConfig` that
            resolves to a :class:`~BaseChatModelFactory`.
        cache_dir: The directory used to store cached results. If
            ``None``, caching is disabled.
        key_fn: A function that derives a cache key from an input. See
            :class:`~zenpyre.runnables.CachingRunnable` for details.
        ignore_none: If ``True``, ``None`` results are treated as if
            there were no cache entry at all. See
            :class:`~zenpyre.runnables.CachingRunnable` for details.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from zenpyre.chat_models.factory import CachingChatModelFactory, ChatModelFactory
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     factory = CachingChatModelFactory(
        ...         chat_model_factory=ChatModelFactory(FakeListChatModel(responses=["hello"])),
        ...         cache_dir=Path(tmpdir),
        ...     )
        ...     chat_model = factory.make_chat_model()
        ...

        ```
    """

    def __init__(
        self,
        chat_model_factory: BaseChatModelFactory | dict[str, Any] | BaseConfig,
        cache_dir: Path | str | None = None,
        key_fn: Callable[[Any], str] | None = None,
        ignore_none: bool = False,
    ) -> None:
        self._chat_model_factory = resolve_object(chat_model_factory, cls=BaseChatModelFactory)
        self._cache_dir = cache_dir
        self._key_fn = key_fn
        self._ignore_none = ignore_none

    def make_chat_model(self) -> BaseChatModel:
        chat_model = self._chat_model_factory.make_chat_model()
        # CachingRunnable exposes the Runnable interface (invoke/batch/...) used to
        # call a chat model, but is not literally a BaseChatModel instance -- cast to
        # satisfy the base factory's declared return type.
        return cast(
            "BaseChatModel",
            CachingRunnable(
                runnable=chat_model,
                cache_dir=self._cache_dir,
                key_fn=self._key_fn,
                ignore_none=self._ignore_none,
            ),
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "chat_model_factory": self._chat_model_factory,
            "cache_dir": self._cache_dir,
            "key_fn": self._key_fn,
            "ignore_none": self._ignore_none,
        }
