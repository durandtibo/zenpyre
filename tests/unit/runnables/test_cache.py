from __future__ import annotations

import asyncio

import pytest
from langchain_core.runnables import RunnableLambda

from tests.unit.runnables.helpers import TrackingRunnable
from zenpyre.runnables import CachingRunnable
from zenpyre.utils.imports import is_persista_available

if is_persista_available():
    from persista.cache import Cache

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


###################################
#     Tests for CachingRunnable   #
###################################


# --- constructor ---


@persista_available
def test_caching_runnable_default_key_fn_is_hash_object() -> None:
    from coola.hashing import hash_object

    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._key_fn is hash_object


@persista_available
def test_caching_runnable_cache_none_disables_caching() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._cache is None


@persista_available
def test_caching_runnable_stores_cache() -> None:
    cache = Cache()
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=cache)
    assert cached._cache is cache


# --- repr/str ---


@persista_available
def test_caching_runnable_repr_contains_class_name() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=Cache())
    assert "CachingRunnable" in repr(cached)


@persista_available
def test_caching_runnable_str_contains_class_name() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache=Cache())
    assert "CachingRunnable" in str(cached)


# --- invoke: caching disabled ---


@persista_available
def test_caching_runnable_invoke_no_cache_always_calls_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    assert cached.invoke("a") == "A"
    assert cached.invoke("a") == "A"
    assert inner.invoke_calls == ["a", "a"]


# --- invoke: caching enabled ---


@persista_available
def test_caching_runnable_invoke_returns_correct_result() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.invoke("hello") == "HELLO"


@persista_available
def test_caching_runnable_invoke_cache_miss_calls_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.invoke("a")
    assert inner.invoke_calls == ["a"]


@persista_available
def test_caching_runnable_invoke_cache_hit_does_not_call_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.invoke("a")
    inner.invoke_calls.clear()
    result = cached.invoke("a")
    assert result == "A"
    assert inner.invoke_calls == []


@persista_available
def test_caching_runnable_invoke_writes_cache_entry() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=cache, key_fn=_identity_key
    )
    cached.invoke("a")
    assert cache.get("a") == "A"


@persista_available
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


@persista_available
def test_caching_runnable_invoke_propagates_inner_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        cached.invoke("a")


# --- ainvoke ---


@persista_available
def test_caching_runnable_ainvoke_no_cache_uses_inner_ainvoke() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    result = asyncio.run(cached.ainvoke("a"))
    assert result == "A"
    assert inner.ainvoke_calls == ["a"]
    assert inner.invoke_calls == []


@persista_available
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


@persista_available
def test_caching_runnable_ainvoke_propagates_inner_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        asyncio.run(cached.ainvoke("a"))


# --- batch ---


@persista_available
def test_caching_runnable_batch_empty_list_returns_empty_list() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.batch([]) == []


@persista_available
def test_caching_runnable_batch_no_cache_delegates_to_inner_batch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    results = cached.batch(["a", "b"])
    assert results == ["A", "B"]
    assert inner.batch_calls == [["a", "b"]]


@persista_available
def test_caching_runnable_batch_returns_correct_results() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert cached.batch(["a", "b", "c"]) == ["A", "B", "C"]


@persista_available
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


@persista_available
def test_caching_runnable_batch_writes_cache_for_new_misses() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=cache, key_fn=_identity_key
    )
    cached.batch(["a", "b"])
    assert cache.get("a") == "A"
    assert cache.get("b") == "B"


@persista_available
def test_caching_runnable_batch_second_call_all_hits() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    cached.batch(["a", "b"])
    inner.batch_calls.clear()

    results = cached.batch(["a", "b"])

    assert results == ["A", "B"]
    assert inner.batch_calls == []


@persista_available
def test_caching_runnable_batch_raises_by_default_on_failure() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for b"):
        cached.batch(["a", "b", "c"])


@persista_available
def test_caching_runnable_batch_return_exceptions_keeps_raw_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    results = cached.batch(["a", "b", "c"], return_exceptions=True)
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


@persista_available
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


@persista_available
def test_caching_runnable_abatch_empty_list_returns_empty_list() -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache=Cache(), key_fn=_identity_key
    )
    assert asyncio.run(cached.abatch([])) == []


@persista_available
def test_caching_runnable_abatch_no_cache_delegates_to_inner_abatch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=None)
    results = asyncio.run(cached.abatch(["a", "b"]))
    assert results == ["A", "B"]
    assert inner.abatch_calls == [["a", "b"]]


@persista_available
def test_caching_runnable_abatch_calls_inner_abatch_only_for_misses() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache=Cache(), key_fn=_identity_key)
    asyncio.run(cached.ainvoke("a"))  # pre-populate the cache for "a"
    inner.ainvoke_calls.clear()
    inner.abatch_calls.clear()

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", "B", "C"]
    assert inner.abatch_calls == [["b", "c"]]


@persista_available
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


@persista_available
def test_caching_runnable_abatch_return_exceptions_keeps_raw_exception() -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=Cache(), key_fn=_identity_key
    )
    results = asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


@persista_available
def test_caching_runnable_abatch_return_exceptions_does_not_cache_failure() -> None:
    cache = Cache()
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache=cache, key_fn=_identity_key
    )
    asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert cache.get("b") is None


# --- ignore_none delegated to the Cache instance ---


@persista_available
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


@persista_available
def test_caching_runnable_cache_ignore_none_true_does_not_cache_none_result() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    result = cached.invoke("a")

    assert result is None
    assert not cache.contains("a")


@persista_available
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


@persista_available
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


@persista_available
def test_caching_runnable_cache_ignore_none_true_batch_mixed_results() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    results = cached.batch(["a", "b", "c"])

    assert results == ["A", None, "C"]
    assert cache.get("a") == "A"
    assert not cache.contains("b")
    assert cache.get("c") == "C"


@persista_available
def test_caching_runnable_cache_ignore_none_true_abatch_mixed_results() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", None, "C"]
    assert cache.get("a") == "A"
    assert not cache.contains("b")
    assert cache.get("c") == "C"


@persista_available
def test_caching_runnable_cache_ignore_none_true_ainvoke_does_not_cache_none() -> None:
    cache = Cache(ignore_none=True)
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache=cache, key_fn=_identity_key)

    result = asyncio.run(cached.ainvoke("a"))

    assert result is None
    assert not cache.contains("a")
