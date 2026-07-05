r"""Contain a runnable with cache."""

from __future__ import annotations

__all__ = ["RunnableWithCache"]

import logging
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.hashing import hash_object
from coola.utils.path import sanitize_path
from iden.io import load_pickle, save_pickle
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


class RunnableWithCache(Runnable[Input, Output], MultilineDisplayMixin):
    r"""Wrap a ``Runnable`` to cache its output to disk, keyed by a hash
    of the input.

    On each call to ``invoke``, ``key_fn(input)`` is used to look up a
    previously pickled result under ``cache_dir``. On a cache hit, the
    pickled result is returned without calling the wrapped runnable. On
    a cache miss, the wrapped runnable is invoked and its result is
    pickled to disk before being returned. If ``cache_dir`` is ``None``,
    caching is disabled entirely and every call goes straight to the
    wrapped runnable.

    Unlike subclassing a caching base class, this wrapper works with any
    ``Runnable`` — including third-party ones you don't control — since
    caching is composed around the runnable rather than baked into its
    class hierarchy.

    Args:
        runnable: The runnable whose output should be cached.
        cache_dir: The directory used to store cached results. If
            ``None``, caching is disabled.
        key_fn: A function that derives a cache key (filename, without
            extension) from an input. Defaults to ``hash_object``, which
            dispatches through ``coola``'s hasher registry (e.g. using
            ``DocumentHasher`` for ``Document`` inputs). Override this
            if you need input-specific hashing behavior that differs
            from the registered default.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from langchain_core.runnables import RunnableLambda
        >>> from zenpyre.agents import RunnableWithCache
        >>> runnable = RunnableLambda(lambda x: x.upper())
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cached = RunnableWithCache(runnable=runnable, cache_dir=Path(tmpdir))
        ...     cached.invoke("hello")
        ...
        'HELLO'

        ```
    """

    def __init__(
        self,
        runnable: Runnable[Input, Output],
        cache_dir: Path | str | None = None,
        key_fn: Callable[[Input], str] | None = None,
    ) -> None:
        self._runnable = runnable
        self._cache_dir = sanitize_path(cache_dir) if cache_dir is not None else None
        self._key_fn = key_fn if key_fn is not None else hash_object

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._cache_dir is None:
            return self._runnable.invoke(input, config, **kwargs)

        filepath = (self._cache_dir / self._key_fn(input)).with_suffix(".pkl")

        if filepath.is_file():
            try:
                result = load_pickle(filepath)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load cache: %s", filepath, exc_info=True)
            else:
                logger.info("Cache hit: %s", filepath)
                return result

        logger.info("Cache miss: %s", filepath)
        result = self._runnable.invoke(input, config, **kwargs)

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(result, filepath, exist_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", filepath, exc_info=True)

        return result

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"runnable": self._runnable, "cache_dir": self._cache_dir}
