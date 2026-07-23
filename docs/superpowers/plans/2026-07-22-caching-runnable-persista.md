# CachingRunnable persista.cache.Cache Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `CachingRunnable`'s disk-pickle cache with a caller-supplied `persista.cache.Cache` instance, and update `CachingAgentFactory` (which wraps `CachingRunnable`) to match, without making `persista` a hard import-time dependency of `zenpyre.runnables`.

**Architecture:** `CachingRunnable` takes `cache: Cache | None` instead of `cache_dir`/`ignore_none`. Internally, disk-pickle I/O (`load_pickle`/`save_pickle`/`_cache_path`) is replaced by `cache.get(key)` / `cache.set(key, value)`. `Cache` is only imported under `TYPE_CHECKING`; at runtime, `zenpyre.utils.imports.check_persista()` is called in `__init__` when a real `Cache` is passed, so `persista`'s absence surfaces as a clear `RuntimeError` at construction time rather than an import-time failure for users who never use caching. `CachingAgentFactory` mirrors the same constructor signature change since it forwards its args straight to `CachingRunnable`.

**Tech Stack:** Python, `persista.cache.cache.Cache`, `langchain_core.runnables.Runnable`, `pytest`.

## Global Constraints

- `persista` stays an optional dependency (`pyproject.toml` extra `persista = [ "persista >=0.0.3a1", ]`) — do not move it to core `dependencies`.
- `Cache` must only be imported under `TYPE_CHECKING` in `src/zenpyre/runnables/cache.py` and `src/zenpyre/agents/factory/cache.py`.
- `CachingRunnable.__init__` calls `zenpyre.utils.imports.check_persista()` when `cache is not None`, before storing it.
- `CachingRunnable` drops its own `ignore_none` parameter entirely; `ignore_none` behavior is delegated to the `Cache` instance the caller passes in.
- `key_fn` keeps its existing signature and default (`hash_object`); its return value is used directly as the `Cache` key (no `.pkl` suffix, no filesystem-safety constraint).
- `ainvoke`/`abatch` call `Cache`'s sync methods directly — no `asyncio.to_thread` wrapping.
- `CachingChatModel` (`src/zenpyre/chat_models/cache.py`) and `CachingChatModelFactory` (`src/zenpyre/chat_models/factory/cache.py`) are out of scope — leave them unchanged.

---

### Task 1: Rewrite `CachingRunnable` to use `persista.cache.Cache`

**Files:**
- Modify: `src/zenpyre/runnables/cache.py`
- Test: `tests/unit/runnables/test_cache.py`

**Interfaces:**
- Consumes: `persista.cache.cache.Cache` (methods used: `cache.get(key: str) -> Any | None`, `cache.set(key: str, value: Any) -> None`; both synchronous). `zenpyre.utils.imports.check_persista() -> None` (raises `RuntimeError` if `persista` is not installed).
- Produces: `CachingRunnable(runnable: Runnable[Input, Output], cache: Cache | None = None, key_fn: Callable[[Input], str] | None = None)`. Public attributes used elsewhere via `_get_repr_kwargs`: `self._cache`, `self._key_fn`. `invoke`, `ainvoke`, `batch`, `abatch` keep their existing public signatures (only internal caching mechanics change). This is consumed by Task 2 (`CachingAgentFactory`).

- [ ] **Step 1: Write the failing tests (full rewrite of `tests/unit/runnables/test_cache.py`)**

Replace the entire file contents with:

```python
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from persista.cache.cache import Cache

from tests.unit.runnables.helpers import TrackingRunnable
from zenpyre.runnables import CachingRunnable

MODULE = "zenpyre.runnables.cache"


def _identity_key(x: str) -> str:
    return x


def _failing_lambda(fail_on: str) -> RunnableLambda:
    def fn(x: str) -> str:
        if x == fail_on:
            msg = f"failed for {x}"
            raise ValueError(msg)
        return x.upper()

    return RunnableLambda(fn)


##################################
#     Tests for CachingRunnable   #
##################################


# --- constructor ---


def test_caching_runnable_default_key_fn_is_hash_object() -> None:
    from coola.hashing import hash_object

    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._key_fn is hash_object


def test_caching_runnable_cache_none_disables_caching() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._cache is None


def test_caching_runnable_stores_cache() -> None:
    cache = Cache()
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=cache)
    assert cached._cache is cache


def test_caching_runnable_missing_persista_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raising_check() -> None:
        msg = "persista is not installed"
        raise RuntimeError(msg)

    monkeypatch.setattr(f"{MODULE}.check_persista", raising_check)
    with pytest.raises(RuntimeError, match="persista is not installed"):
        CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=Cache())


# --- repr ---


def test_caching_runnable_repr_contains_class_name() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=Cache())
    assert "CachingRunnable" in repr(cached)


# --- invoke: caching disabled ---


def test_caching_runnable_invoke_no_cache_always_calls_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    assert cached.invoke("a") == "A"
    assert cached.invoke("a") == "A"
    assert inner.invoke_calls == ["a", "a"]


# --- invoke: caching enabled ---


def test_caching_runnable_invoke_returns_correct_result() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.invoke("hello") == "HELLO"


def test_caching_runnable_invoke_cache_miss_calls_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.invoke("a")
    assert inner.invoke_calls == ["a"]


def test_caching_runnable_invoke_cache_hit_does_not_call_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.invoke("a")
    inner.invoke_calls.clear()
    result = cached.invoke("a")
    assert result == "A"
    assert inner.invoke_calls == []


def test_caching_runnable_invoke_writes_cache_entry() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=cache, key_fn=_identity_key
    )
    cached.invoke("a")
    assert cache.get("a") == "A"


def test_caching_runnable_invoke_key_with_dot_not_truncated() -> None:
    # Regression test: the old pickle-file backend used
    # Path.with_suffix(".pkl"), which would turn a key like "3.14" into
    # "3.pkl", silently truncating the key. Cache keys are plain strings,
    # so this must not happen.
    cache = Cache()
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=cache, key_fn=_identity_key
    )
    cached.invoke("3.14")
    assert cache.get("3.14") == "3.14".upper()
    assert cache.get("3") is None


def test_caching_runnable_invoke_propagates_inner_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        cached.invoke("a")


# --- ainvoke ---


def test_caching_runnable_ainvoke_no_cache_uses_inner_ainvoke() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    result = asyncio.run(cached.ainvoke("a"))
    assert result == "A"
    assert inner.ainvoke_calls == ["a"]
    assert inner.invoke_calls == []


def test_caching_runnable_ainvoke_cache_miss_then_hit() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)

    result1 = asyncio.run(cached.ainvoke("a"))
    assert result1 == "A"
    assert inner.ainvoke_calls == ["a"]

    inner.ainvoke_calls.clear()
    result2 = asyncio.run(cached.ainvoke("a"))
    assert result2 == "A"
    assert inner.ainvoke_calls == []


def test_caching_runnable_ainvoke_propagates_inner_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        asyncio.run(cached.ainvoke("a"))


# --- batch ---


def test_caching_runnable_batch_empty_list_returns_empty_list() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.batch([]) == []


def test_caching_runnable_batch_no_cache_delegates_to_inner_batch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    results = cached.batch(["a", "b"])
    assert results == ["A", "B"]
    assert inner.batch_calls == [["a", "b"]]


def test_caching_runnable_batch_returns_correct_results() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.batch(["a", "b", "c"]) == ["A", "B", "C"]


def test_caching_runnable_batch_calls_inner_batch_only_for_misses() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.invoke("a")  # pre-populate the cache for "a"
    inner.invoke_calls.clear()
    inner.batch_calls.clear()

    results = cached.batch(["a", "b", "c"])

    assert results == ["A", "B", "C"]
    # Only the misses ("b", "c") should be sent to inner.batch, as one
    # call -- not one call per miss, and "a" should not appear at all.
    assert inner.batch_calls == [["b", "c"]]


def test_caching_runnable_batch_writes_cache_for_new_misses() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=cache, key_fn=_identity_key
    )
    cached.batch(["a", "b"])
    assert cache.get("a") == "A"
    assert cache.get("b") == "B"


def test_caching_runnable_batch_second_call_all_hits() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.batch(["a", "b"])
    inner.batch_calls.clear()

    results = cached.batch(["a", "b"])

    assert results == ["A", "B"]
    assert inner.batch_calls == []


def test_caching_runnable_batch_raises_by_default_on_failure() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for b"):
        cached.batch(["a", "b", "c"])


def test_caching_runnable_batch_return_exceptions_keeps_raw_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    results = cached.batch(["a", "b", "c"], return_exceptions=True)
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


def test_caching_runnable_batch_return_exceptions_does_not_cache_failure() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=cache, key_fn=_identity_key
    )
    cached.batch(["a", "b", "c"], return_exceptions=True)
    assert cache.get("b") is None
    assert cache.get("a") == "A"
    assert cache.get("c") == "C"


# --- abatch ---


def test_caching_runnable_abatch_empty_list_returns_empty_list() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert asyncio.run(cached.abatch([])) == []


def test_caching_runnable_abatch_no_cache_delegates_to_inner_abatch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    results = asyncio.run(cached.abatch(["a", "b"]))
    assert results == ["A", "B"]
    assert inner.abatch_calls == [["a", "b"]]


def test_caching_runnable_abatch_calls_inner_abatch_only_for_misses() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    asyncio.run(cached.ainvoke("a"))  # pre-populate the cache for "a"
    inner.ainvoke_calls.clear()
    inner.abatch_calls.clear()

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", "B", "C"]
    assert inner.abatch_calls == [["b", "c"]]


def test_caching_runnable_abatch_second_call_all_hits() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    asyncio.run(cached.abatch(["a", "b"]))
    inner.abatch_calls.clear()

    results = asyncio.run(cached.abatch(["a", "b"]))

    assert results == ["A", "B"]
    # No misses at all this time, so the "if miss_indices:" branch in
    # abatch must be skipped entirely -- inner.abatch must not be called.
    assert inner.abatch_calls == []


def test_caching_runnable_abatch_return_exceptions_keeps_raw_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    results = asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


def test_caching_runnable_abatch_return_exceptions_does_not_cache_failure() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=cache, key_fn=_identity_key
    )
    asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert cache.get("b") is None


# --- ignore_none delegated to the Cache instance ---


def test_caching_runnable_cache_ignore_none_false_caches_none_result() -> None:
    cache = Cache()  # ignore_none defaults to False
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    result1 = cached.invoke("a")
    assert result1 is None
    assert cache.contains("a")

    inner.invoke_calls.clear()
    result2 = cached.invoke("a")
    assert result2 is None
    assert inner.invoke_calls == []  # cache hit, no call to inner


def test_caching_runnable_cache_ignore_none_true_does_not_cache_none_result() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    result = cached.invoke("a")

    assert result is None
    assert not cache.contains("a")


def test_caching_runnable_cache_ignore_none_true_none_result_is_a_miss_next_call() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    cached.invoke("a")
    inner.invoke_calls.clear()
    cached.invoke("a")

    # Nothing was ever cached, so the second call must also hit the inner
    # runnable.
    assert inner.invoke_calls == ["a"]


def test_caching_runnable_cache_ignore_none_true_non_none_results_still_cached() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="skip-me")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    cached.invoke("a")
    assert cache.get("a") == "A"

    inner.invoke_calls.clear()
    result = cached.invoke("a")
    assert result == "A"
    assert inner.invoke_calls == []  # cache hit


def test_caching_runnable_cache_ignore_none_true_batch_mixed_results() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    results = cached.batch(["a", "b", "c"])

    assert results == ["A", None, "C"]
    assert cache.get("a") == "A"
    assert not cache.contains("b")
    assert cache.get("c") == "C"


def test_caching_runnable_cache_ignore_none_true_abatch_mixed_results() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", None, "C"]
    assert cache.get("a") == "A"
    assert not cache.contains("b")
    assert cache.get("c") == "C"


def test_caching_runnable_cache_ignore_none_true_ainvoke_does_not_cache_none() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    result = asyncio.run(cached.ainvoke("a"))

    assert result is None
    assert not cache.contains("a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/runnables/test_cache.py -v`
Expected: FAIL (many tests) — `CachingRunnable.__init__() got an unexpected keyword argument 'cache'` and similar, since the implementation still uses `cache_dir`/`ignore_none`.

- [ ] **Step 3: Rewrite `src/zenpyre/runnables/cache.py`**

Replace the entire file contents with:

```python
r"""Contain a caching runnable wrapper."""

from __future__ import annotations

__all__ = ["CachingRunnable"]

import logging
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.hashing import hash_object
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_config_list
from langchain_core.runnables.utils import Input, Output

from zenpyre.utils.imports import check_persista

if TYPE_CHECKING:
    from collections.abc import Callable

    from persista.cache.cache import Cache

logger: logging.Logger = logging.getLogger(__name__)


class CachingRunnable(Runnable[Input, Output], MultilineDisplayMixin):
    r"""Wrap a ``Runnable`` to cache its output, keyed by a hash of the
    input.

    On each call, ``key_fn(input)`` is used to look up a previously
    cached result in ``cache``. On a cache hit, the cached result is
    returned without calling the wrapped runnable. On a cache miss,
    the wrapped runnable is invoked and its result is stored in
    ``cache`` before being returned. If ``cache`` is ``None``, caching
    is disabled entirely and every call goes straight to the wrapped
    runnable.

    ``batch``/``abatch`` look up each input's cache entry individually,
    then call the wrapped runnable's own ``batch``/``abatch`` for only
    the inputs that missed -- so a partially-cached batch still benefits
    from the wrapped runnable's batching, rather than falling back to
    one call per miss.

    Unlike subclassing a caching base class, this wrapper works with any
    ``Runnable`` — including third-party ones you don't control — since
    caching is composed around the runnable rather than baked into its
    class hierarchy.

    Args:
        runnable: The runnable whose output should be cached.
        cache: The :class:`~persista.cache.cache.Cache` instance used
            to store cached results. If ``None``, caching is disabled.
            The caller configures the cache's backing store, TTL, and
            ``ignore_none`` behavior; ``CachingRunnable`` has no
            caching policy of its own beyond what ``cache`` provides.
        key_fn: A function that derives a cache key from an input. The
            returned string is used directly as the ``cache`` key.
            Defaults to ``hash_object``, which dispatches through
            ``coola``'s hasher registry (e.g. using ``DocumentHasher``
            for ``Document`` inputs). Override this if you need
            input-specific hashing behavior that differs from the
            registered default.

    Raises:
        RuntimeError: If ``cache`` is not ``None`` and the ``persista``
            package is not installed.

    Example:
        ```pycon
        >>> from langchain_core.runnables import RunnableLambda
        >>> from persista.cache.cache import Cache
        >>> from zenpyre.runnables import CachingRunnable
        >>> runnable = RunnableLambda(lambda x: x.upper())
        >>> cached = CachingRunnable(runnable=runnable, cache=Cache())
        >>> cached.invoke("hello")
        'HELLO'

        ```
    """

    def __init__(
        self,
        runnable: Runnable[Input, Output],
        cache: Cache | None = None,
        key_fn: Callable[[Input], str] | None = None,
    ) -> None:
        self._runnable = runnable
        if cache is not None:
            check_persista()
        self._cache = cache
        self._key_fn = key_fn if key_fn is not None else hash_object

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._cache is None:
            return self._runnable.invoke(input, config=config, **kwargs)

        key = self._key_fn(input)
        if self._cache.contains(key):
            logger.info("Cache hit: %s", key)
            return self._cache.get(key)

        logger.info("Cache miss: %s", key)
        result = self._runnable.invoke(input, config=config, **kwargs)
        self._cache.set(key, result)
        return result

    async def ainvoke(
        self,
        input: Input,  # noqa: A002
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._cache is None:
            return await self._runnable.ainvoke(input, config=config, **kwargs)

        key = self._key_fn(input)
        if self._cache.contains(key):
            logger.info("Cache hit: %s", key)
            return self._cache.get(key)

        logger.info("Cache miss: %s", key)
        result = await self._runnable.ainvoke(input, config=config, **kwargs)
        self._cache.set(key, result)
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
        if self._cache is None:
            return self._runnable.batch(
                inputs, config=config, return_exceptions=return_exceptions, **kwargs
            )

        configs = get_config_list(config, len(inputs))
        keys = [self._key_fn(inp) for inp in inputs]

        results: list[Any] = [None] * len(inputs)
        miss_indices: list[int] = []
        for i, key in enumerate(keys):
            if self._cache.contains(key):
                results[i] = self._cache.get(key)
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
                self._cache.set(keys[idx], output)

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
        if self._cache is None:
            return await self._runnable.abatch(
                inputs, config=config, return_exceptions=return_exceptions, **kwargs
            )

        configs = get_config_list(config, len(inputs))
        keys = [self._key_fn(inp) for inp in inputs]

        results: list[Any] = [None] * len(inputs)
        miss_indices: list[int] = []
        for i, key in enumerate(keys):
            if self._cache.contains(key):
                results[i] = self._cache.get(key)
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
                self._cache.set(keys[idx], output)

        return results

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "runnable": self._runnable,
            "cache": self._cache,
            "key_fn": self._key_fn,
        }
```

Note: `Cache.get(key)` alone can't distinguish "no entry" from "entry
present with value `None`" (both return `None`), so hit/miss is checked
via the public `Cache.contains(key)` first, then `Cache.get(key)` fetches
the value on a hit. This costs a second store lookup on a hit but only
uses `Cache`'s public API (`get`/`set`/`contains`), matching how
`Cache.contains`'s own docstring describes it being used for exactly
this kind of existence check. `Cache.get_or_compute` was considered but
doesn't fit: it calls its `fn` synchronously and un-awaited, so it can't
be used for `ainvoke` (which must `await self._runnable.ainvoke(...)`),
and using it only for the sync `invoke` path while `contains`/`get` are
used everywhere else would make the four methods inconsistent for no
benefit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/runnables/test_cache.py -v`
Expected: PASS — all tests green.

- [ ] **Step 5: Run the full test suite for the runnables package to check for regressions**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/runnables/ -v`
Expected: PASS — all tests green (this also covers `tests/unit/runnables/helpers.py`, which is unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/zenpyre/runnables/cache.py tests/unit/runnables/test_cache.py
git commit -m "refactor: back CachingRunnable with persista.cache.Cache instead of pickle files"
```

---

### Task 2: Update `CachingAgentFactory` to match `CachingRunnable`'s new signature

**Files:**
- Modify: `src/zenpyre/agents/factory/cache.py`
- Test: `tests/unit/agents/factory/test_cache.py`

**Interfaces:**
- Consumes: `CachingRunnable(runnable, cache: Cache | None = None, key_fn: Callable[[Input], str] | None = None)` from Task 1.
- Produces: `CachingAgentFactory(agent_factory, cache: Cache | None = None, key_fn: Callable[[dict[str, Any]], str] | None = None)`. `_get_repr_kwargs()` returns `{"agent_factory": ..., "cache": ..., "key_fn": ...}`.

- [ ] **Step 1: Write the failing tests (full rewrite of `tests/unit/agents/factory/test_cache.py`)**

Replace the entire file contents with:

```python
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from coola.equality import objects_are_equal
from persista.cache.cache import Cache

from zenpyre.agents.factory import AgentFactory, BaseAgentFactory, CachingAgentFactory
from zenpyre.utils.config import Config

MODULE = "zenpyre.agents.factory.cache"

MINIMAL_AGENT_FACTORY_TARGET = "tests.unit.agents.factory.test_cache.MinimalAgentFactory"


class MinimalAgentFactory(BaseAgentFactory):
    """Minimal concrete BaseAgentFactory for testing."""

    def make_agent(self) -> Any:
        return AgentFactory(MagicMock())


def _make_agent_factory() -> MagicMock:
    """Return a MagicMock standing in for a BaseAgentFactory."""
    return MagicMock(spec=BaseAgentFactory)


def _make_factory(**overrides: Any) -> CachingAgentFactory:
    """Return a CachingAgentFactory instance for testing."""
    kwargs = {
        "agent_factory": _make_agent_factory(),
        "cache": None,
        "key_fn": None,
    }
    kwargs.update(overrides)
    return CachingAgentFactory(**kwargs)


#########################################
#     Tests for CachingAgentFactory     #
#########################################


# --- Inheritance ---


def test_caching_agent_factory_is_base_agent_factory() -> None:
    assert isinstance(_make_factory(), BaseAgentFactory)


# --- __init__ stores args ---


def test_caching_agent_factory_stores_agent_factory() -> None:
    agent_factory = _make_agent_factory()
    factory = _make_factory(agent_factory=agent_factory)
    assert factory._agent_factory is agent_factory


def test_caching_agent_factory_stores_cache() -> None:
    cache = Cache()
    factory = _make_factory(cache=cache)
    assert factory._cache is cache


def test_caching_agent_factory_default_cache_is_none() -> None:
    factory = _make_factory()
    assert factory._cache is None


def test_caching_agent_factory_stores_key_fn() -> None:
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(key_fn=key_fn)
    assert factory._key_fn is key_fn


# --- __init__ resolves agent_factory ---


def test_caching_agent_factory_resolves_agent_factory_from_dict() -> None:
    factory = _make_factory(agent_factory={"_target_": MINIMAL_AGENT_FACTORY_TARGET})
    assert isinstance(factory._agent_factory, MinimalAgentFactory)


def test_caching_agent_factory_resolves_agent_factory_from_base_config() -> None:
    config = Config.from_kwargs(_target_=MINIMAL_AGENT_FACTORY_TARGET)
    factory = _make_factory(agent_factory=config)
    assert isinstance(factory._agent_factory, MinimalAgentFactory)


def test_caching_agent_factory_invalid_agent_factory_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseAgentFactory instance"):
        _make_factory(agent_factory="not-an-agent-factory")


# --- make_agent wiring ---


def test_caching_agent_factory_make_agent_builds_agent_from_factory() -> None:
    agent_factory = _make_agent_factory()
    factory = _make_factory(agent_factory=agent_factory)
    with patch(f"{MODULE}.CachingRunnable"):
        factory.make_agent()
        agent_factory.make_agent.assert_called_once_with()


def test_caching_agent_factory_make_agent_wraps_in_caching_runnable() -> None:
    agent_factory = _make_agent_factory()
    cache = Cache()
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(agent_factory=agent_factory, cache=cache, key_fn=key_fn)
    with patch(f"{MODULE}.CachingRunnable") as mock_caching_runnable_cls:
        factory.make_agent()
        mock_caching_runnable_cls.assert_called_once_with(
            runnable=agent_factory.make_agent.return_value,
            cache=cache,
            key_fn=key_fn,
        )


def test_caching_agent_factory_make_agent_returns_caching_runnable() -> None:
    factory = _make_factory()
    with patch(f"{MODULE}.CachingRunnable") as mock_caching_runnable_cls:
        result = factory.make_agent()
        assert result is mock_caching_runnable_cls.return_value


# --- _get_repr_kwargs ---


def test_caching_agent_factory_get_repr_kwargs() -> None:
    agent_factory = _make_agent_factory()
    cache = Cache()
    key_fn = lambda x: str(x)  # noqa: E731
    factory = _make_factory(agent_factory=agent_factory, cache=cache, key_fn=key_fn)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {
            "agent_factory": agent_factory,
            "cache": cache,
            "key_fn": key_fn,
        },
    )


# --- __repr__ and __str__ ---


def test_caching_agent_factory_repr_starts_with_class_name() -> None:
    factory = _make_factory()
    assert repr(factory).startswith("CachingAgentFactory(")


def test_caching_agent_factory_str_starts_with_class_name() -> None:
    factory = _make_factory()
    assert str(factory).startswith("CachingAgentFactory(")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/agents/factory/test_cache.py -v`
Expected: FAIL — `CachingAgentFactory.__init__() got an unexpected keyword argument 'cache'`.

- [ ] **Step 3: Rewrite `src/zenpyre/agents/factory/cache.py`**

Replace the entire file contents with:

```python
r"""Provide a concrete agent factory that wraps another agent factory
with caching."""

from __future__ import annotations

__all__ = ["CachingAgentFactory"]

from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin

from zenpyre.agents.factory.base import BaseAgentFactory
from zenpyre.runnables import CachingRunnable
from zenpyre.utils.resolve import resolve_object

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.runnables import Runnable
    from persista.cache.cache import Cache

    from zenpyre.utils.config import BaseConfig


class CachingAgentFactory(BaseAgentFactory, MultilineDisplayMixin):
    """A concrete agent factory that wraps another agent factory and
    caches the resulting agent's outputs via
    :class:`~zenpyre.runnables.CachingRunnable`.

    Each call to :meth:`make_agent` builds a fresh agent from
    ``agent_factory`` and wraps it in a
    :class:`~zenpyre.runnables.CachingRunnable`, so repeated calls to
    the wrapped agent with the same (or equivalent, per ``key_fn``)
    input are served from ``cache`` instead of re-invoking the
    underlying agent.

    Args:
        agent_factory: The factory used to build the underlying agent
            to cache.
        cache: The :class:`~persista.cache.cache.Cache` instance used
            to store cached results. If ``None``, caching is disabled.
        key_fn: An optional function used to compute a cache key from
            the agent's input. If ``None``,
            :class:`~zenpyre.runnables.CachingRunnable`'s default
            key-computation strategy is used.

    Example:
        ```pycon
        >>> from langchain_core.language_models import FakeListChatModel
        >>> from persista.cache.cache import Cache
        >>> from zenpyre.agents import AgentChatModel
        >>> from zenpyre.agents.factory import AgentFactory, CachingAgentFactory
        >>> inner_agent = AgentChatModel(model=FakeListChatModel(responses=["hello"]))
        >>> factory = CachingAgentFactory(
        ...     agent_factory=AgentFactory(inner_agent),
        ...     cache=Cache(),
        ... )
        >>> agent = factory.make_agent()  # doctest: +SKIP

        ```
    """

    def __init__(
        self,
        agent_factory: BaseAgentFactory | dict[str, Any] | BaseConfig,
        cache: Cache | None = None,
        key_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self._agent_factory = resolve_object(agent_factory, cls=BaseAgentFactory)
        self._cache = cache
        self._key_fn = key_fn

    def make_agent(self) -> Runnable[dict[str, Any], dict[str, Any]]:
        agent = self._agent_factory.make_agent()
        return CachingRunnable(
            runnable=agent,
            cache=self._cache,
            key_fn=self._key_fn,
        )

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "agent_factory": self._agent_factory,
            "cache": self._cache,
            "key_fn": self._key_fn,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/agents/factory/test_cache.py -v`
Expected: PASS — all tests green.

- [ ] **Step 5: Run the full test suite for the agents package to check for regressions**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/agents/ -v`
Expected: PASS — all tests green.

- [ ] **Step 6: Commit**

```bash
git add src/zenpyre/agents/factory/cache.py tests/unit/agents/factory/test_cache.py
git commit -m "refactor: update CachingAgentFactory for CachingRunnable's Cache-based API"
```

---

### Task 3: Full regression sweep and lint

**Files:**
- No new files. Verification only.

**Interfaces:**
- Consumes: everything produced in Tasks 1–2.
- Produces: nothing new; this task only verifies the whole change is internally consistent (no other file references the old `cache_dir`/`ignore_none` params on `CachingRunnable` or `CachingAgentFactory`).

- [ ] **Step 1: Search for any remaining references to the old API**

Run: `cd /Users/thibaut/workspace/code/zenpyre && grep -rn "cache_dir" src/zenpyre/runnables/ src/zenpyre/agents/factory/cache.py tests/unit/runnables/test_cache.py tests/unit/agents/factory/test_cache.py`
Expected: no matches (empty output). If there are matches, they are leftover references that must be fixed before continuing.

- [ ] **Step 2: Run the full unit test suite**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run --extra persista pytest tests/unit/ -v`
Expected: PASS — all tests green, no failures or errors anywhere else in the suite (e.g. no other module constructs `CachingRunnable`/`CachingAgentFactory` with the old signature).

- [ ] **Step 3: Run ruff lint on the changed files**

Run: `cd /Users/thibaut/workspace/code/zenpyre && uv run ruff check src/zenpyre/runnables/cache.py src/zenpyre/agents/factory/cache.py tests/unit/runnables/test_cache.py tests/unit/agents/factory/test_cache.py`
Expected: no lint errors.

- [ ] **Step 4: Commit any lint fixes, if needed**

```bash
git add -u
git commit -m "style: fix lint issues from CachingRunnable persista migration"
```

(Skip this step entirely if Step 3 produced no changes.)
