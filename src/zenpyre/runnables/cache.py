r"""Contain a caching runnable wrapper."""

from __future__ import annotations

__all__ = ["CachingRunnable"]

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.hashing import hash_object
from coola.utils.path import sanitize_path
from iden.io import load_pickle, save_pickle
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_config_list
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


class CachingRunnable(Runnable[Input, Output], MultilineDisplayMixin):
    r"""Wrap a ``Runnable`` to cache its output to disk, keyed by a hash
    of the input.

    On each call, ``key_fn(input)`` is used to look up a previously
    pickled result under ``cache_dir``. On a cache hit, the pickled
    result is returned without calling the wrapped runnable. On a cache
    miss, the wrapped runnable is invoked and its result is pickled to
    disk before being returned. If ``cache_dir`` is ``None``, caching is
    disabled entirely and every call goes straight to the wrapped
    runnable.

    ``batch``/``abatch`` look up each input's cache entry individually,
    then call the wrapped runnable's own ``batch``/``abatch`` for only
    the inputs that missed -- so a partially-cached batch still benefits
    from the wrapped runnable's batching, rather than falling back to
    one call per miss.

    A failure to read or write a cache file (corrupt pickle, permission
    error, disk full, etc.) is logged as a warning and treated as a
    cache miss; it never raises and never prevents the wrapped runnable
    from producing a result.

    Unlike subclassing a caching base class, this wrapper works with any
    ``Runnable`` — including third-party ones you don't control — since
    caching is composed around the runnable rather than baked into its
    class hierarchy.

    Args:
        runnable: The runnable whose output should be cached.
        cache_dir: The directory used to store cached results. If
            ``None``, caching is disabled.
        key_fn: A function that derives a cache key from an input. The
            returned string is used as a filename (with a ``.pkl``
            extension appended) under ``cache_dir``, so it must be
            filesystem-safe -- in particular, it must not contain path
            separators. Defaults to ``hash_object``, which dispatches
            through ``coola``'s hasher registry (e.g. using
            ``DocumentHasher`` for ``Document`` inputs). Override this
            if you need input-specific hashing behavior that differs
            from the registered default.
        ignore_none: If ``False`` (the default), ``None`` is treated as
            an ordinary cacheable value: a cached ``None`` counts as a
            cache hit, and a wrapped-runnable result of ``None`` is
            written to disk like any other result. If ``True``, ``None``
            is treated as if there were no cache entry at all: a cached
            ``None`` is treated as a cache miss (the wrapped runnable is
            called instead of returning ``None``), and a wrapped-runnable
            result of ``None`` is never written to disk. This is useful
            when the wrapped runnable can return ``None`` to mean
            "no result yet" or "failed silently," and you don't want
            that non-result to be cached as if it were a real answer.

    Example:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> from langchain_core.runnables import RunnableLambda
        >>> from zenpyre.runnables import CachingRunnable
        >>> runnable = RunnableLambda(lambda x: x.upper())
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cached = CachingRunnable(runnable=runnable, cache_dir=Path(tmpdir))
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
        ignore_none: bool = False,
    ) -> None:
        self._runnable = runnable
        self._cache_dir = sanitize_path(cache_dir) if cache_dir is not None else None
        self._key_fn = key_fn if key_fn is not None else hash_object
        self._ignore_none = ignore_none

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._cache_dir is None:
            return self._runnable.invoke(input, config=config, **kwargs)

        filepath = self._cache_path(self._key_fn(input))
        hit, result = self._load_cached(filepath)
        if hit:
            return result

        logger.info("Cache miss: %s", filepath)
        result = self._runnable.invoke(input, config=config, **kwargs)
        self._save_cache(result, filepath)
        return result

    async def ainvoke(
        self,
        input: Input,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._cache_dir is None:
            return await self._runnable.ainvoke(input, config=config, **kwargs)

        filepath = self._cache_path(self._key_fn(input))
        hit, result = await self._aload_cached(filepath)
        if hit:
            return result

        logger.info("Cache miss: %s", filepath)
        result = await self._runnable.ainvoke(input, config=config, **kwargs)
        await self._asave_cache(result, filepath)
        return result

    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        if not inputs:
            return []
        if self._cache_dir is None:
            return self._runnable.batch(
                inputs, config=config, return_exceptions=return_exceptions, **kwargs
            )

        configs = get_config_list(config, len(inputs))
        filepaths = [self._cache_path(self._key_fn(inp)) for inp in inputs]

        results: list[Any] = [None] * len(inputs)
        miss_indices: list[int] = []
        for i, filepath in enumerate(filepaths):
            hit, result = self._load_cached(filepath)
            if hit:
                results[i] = result
            else:
                miss_indices.append(i)

        if miss_indices:
            logger.info("Cache miss: %d/%d input(s)", len(miss_indices), len(inputs))
            miss_outputs = self._runnable.batch(
                [inputs[i] for i in miss_indices],
                config=[configs[i] for i in miss_indices],
                return_exceptions=return_exceptions,
                **kwargs,
            )
            for idx, output in zip(miss_indices, miss_outputs, strict=True):
                results[idx] = output
                if return_exceptions and isinstance(output, BaseException):
                    continue
                self._save_cache(output, filepaths[idx])

        return results

    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        if not inputs:
            return []
        if self._cache_dir is None:
            return await self._runnable.abatch(
                inputs, config=config, return_exceptions=return_exceptions, **kwargs
            )

        configs = get_config_list(config, len(inputs))
        filepaths = [self._cache_path(self._key_fn(inp)) for inp in inputs]

        results: list[Any] = [None] * len(inputs)
        miss_indices: list[int] = []
        for i, filepath in enumerate(filepaths):
            hit, result = await self._aload_cached(filepath)
            if hit:
                results[i] = result
            else:
                miss_indices.append(i)

        if miss_indices:
            logger.info("Cache miss: %d/%d input(s)", len(miss_indices), len(inputs))
            miss_outputs = await self._runnable.abatch(
                [inputs[i] for i in miss_indices],
                config=[configs[i] for i in miss_indices],
                return_exceptions=return_exceptions,
                **kwargs,
            )
            for idx, output in zip(miss_indices, miss_outputs, strict=True):
                results[idx] = output
                if return_exceptions and isinstance(output, BaseException):
                    continue
                await self._asave_cache(output, filepaths[idx])

        return results

    def _cache_path(self, key: str) -> Path:
        """Build the cache file path for ``key``.

        Uses plain string concatenation rather than
        ``Path.with_suffix(".pkl")``, since ``with_suffix`` treats
        anything after the last ``.`` in ``key`` as an existing suffix
        and replaces it -- silently truncating keys that legitimately
        contain a period (e.g. ``"3.14"`` would become ``"3.pkl"``).

        Args:
            key: The cache key, as returned by ``self._key_fn``.

        Returns:
            The full path to the cache file for ``key``.
        """
        return self._cache_dir / f"{key}.pkl"

    def _load_cached(self, filepath: Path) -> tuple[bool, Output | None]:
        """Attempt to load a cached result from ``filepath``.

        Args:
            filepath: The cache file to read.

        Returns:
            A ``(hit, result)`` tuple. ``hit`` is ``True`` only if
                ``filepath`` exists and was loaded successfully, and
                the loaded value is not ``None`` when
                ``self._ignore_none`` is ``True`` (in which case a
                cached ``None`` is treated the same as a missing cache
                file). In every other case (missing file, corrupt
                pickle, read error, ignored ``None``) it is ``False``
                and ``result`` is ``None``.
        """
        if not filepath.is_file():
            return False, None
        try:
            result = load_pickle(filepath)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", filepath, exc_info=True)
            return False, None
        if self._ignore_none and result is None:
            logger.info("Ignoring cached None: %s", filepath)
            return False, None
        logger.info("Cache hit: %s", filepath)
        return True, result

    async def _aload_cached(self, filepath: Path) -> tuple[bool, Output | None]:
        """Async equivalent of :meth:`_load_cached`, run in a thread so
        the blocking file read doesn't block the event loop."""
        if not filepath.is_file():
            return False, None
        try:
            result = await asyncio.to_thread(load_pickle, filepath)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to load cache: %s", filepath, exc_info=True)
            return False, None
        if self._ignore_none and result is None:
            logger.info("Ignoring cached None: %s", filepath)
            return False, None
        logger.info("Cache hit: %s", filepath)
        return True, result

    def _save_cache(self, result: Output, filepath: Path) -> None:
        """Best-effort write of ``result`` to ``filepath``.

        If ``self._ignore_none`` is ``True`` and ``result`` is ``None``,
        nothing is written -- a ``None`` result is treated as "not a
        real result to cache."

        Failures are logged as a warning rather than raised, since a
        cache write failure should not prevent a successfully computed
        result from being returned to the caller.

        Args:
            result: The result to cache.
            filepath: The cache file to write.
        """
        if self._ignore_none and result is None:
            logger.info("Not caching None result: %s", filepath)
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(result, filepath, exist_ok=True)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", filepath, exc_info=True)

    async def _asave_cache(self, result: Output, filepath: Path) -> None:
        """Async equivalent of :meth:`_save_cache`, run in a thread so
        the blocking file write doesn't block the event loop."""
        if self._ignore_none and result is None:
            logger.info("Not caching None result: %s", filepath)
            return

        def _write() -> None:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            save_pickle(result, filepath, exist_ok=True)

        try:
            await asyncio.to_thread(_write)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache: %s", filepath, exc_info=True)

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "runnable": self._runnable,
            "cache_dir": self._cache_dir,
            "key_fn": self._key_fn,
            "ignore_none": self._ignore_none,
        }
