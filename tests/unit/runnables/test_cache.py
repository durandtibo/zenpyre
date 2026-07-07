from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.runnables import RunnableLambda

from tests.unit.runnables.helpers import TrackingRunnable
from zenpyre.runnables import CachingRunnable

if TYPE_CHECKING:
    from pathlib import Path

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


def test_caching_runnable_default_ignore_none(tmp_path: Path) -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache_dir=tmp_path)
    assert cached._ignore_none is False


def test_caching_runnable_stores_ignore_none(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x), cache_dir=tmp_path, ignore_none=True
    )
    assert cached._ignore_none is True


def test_caching_runnable_default_key_fn_is_hash_object() -> None:
    from coola.hashing import hash_object

    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._key_fn is hash_object


def test_caching_runnable_cache_dir_none_disables_caching() -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x))
    assert cached._cache_dir is None


# --- repr ---


def test_caching_runnable_repr_contains_class_name(tmp_path: Path) -> None:
    cached = CachingRunnable(runnable=RunnableLambda(lambda x: x), cache_dir=tmp_path)
    assert "CachingRunnable" in repr(cached)


# --- invoke: caching disabled ---


def test_caching_runnable_invoke_no_cache_dir_always_calls_inner() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=None)
    assert cached.invoke("a") == "A"
    assert cached.invoke("a") == "A"
    assert inner.invoke_calls == ["a", "a"]


# --- invoke: caching enabled ---


def test_caching_runnable_invoke_returns_correct_result(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    assert cached.invoke("hello") == "HELLO"


def test_caching_runnable_invoke_cache_miss_calls_inner(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    cached.invoke("a")
    assert inner.invoke_calls == ["a"]


def test_caching_runnable_invoke_cache_hit_does_not_call_inner(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    cached.invoke("a")
    inner.invoke_calls.clear()
    result = cached.invoke("a")
    assert result == "A"
    assert inner.invoke_calls == []


def test_caching_runnable_invoke_writes_cache_file(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    cached.invoke("a")
    assert (tmp_path / "a.pkl").is_file()


def test_caching_runnable_invoke_key_with_dot_not_truncated(tmp_path: Path) -> None:
    # Regression test: Path.with_suffix(".pkl") would previously turn a
    # key like "3.14" into "3.pkl", silently truncating the key.
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    cached.invoke("3.14")
    assert (tmp_path / "3.14.pkl").is_file()
    assert not (tmp_path / "3.pkl").is_file()


def test_caching_runnable_invoke_corrupt_cache_treated_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    filepath = tmp_path / "a.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(b"not a valid pickle")

    def raising_load(_path: Path) -> None:
        msg = "corrupt pickle"
        raise ValueError(msg)

    monkeypatch.setattr(f"{MODULE}.load_pickle", raising_load)

    with caplog.at_level("WARNING"):
        result = cached.invoke("a")

    assert result == "A"
    assert inner.invoke_calls == ["a"]
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_runnable_invoke_write_failure_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def raising_save(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{MODULE}.save_pickle", raising_save)
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = cached.invoke("a")

    assert result == "A"
    assert not (tmp_path / "a.pkl").exists()
    assert any("Failed to write cache" in message for message in caplog.messages)


def test_caching_runnable_invoke_propagates_inner_exception(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache_dir=tmp_path, key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        cached.invoke("a")


# --- ainvoke ---


def test_caching_runnable_ainvoke_no_cache_dir_uses_inner_ainvoke() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=None)
    result = asyncio.run(cached.ainvoke("a"))
    assert result == "A"
    assert inner.ainvoke_calls == ["a"]
    assert inner.invoke_calls == []


def test_caching_runnable_ainvoke_cache_miss_then_hit(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)

    result1 = asyncio.run(cached.ainvoke("a"))
    assert result1 == "A"
    assert inner.ainvoke_calls == ["a"]

    inner.ainvoke_calls.clear()
    result2 = asyncio.run(cached.ainvoke("a"))
    assert result2 == "A"
    assert inner.ainvoke_calls == []


def test_caching_runnable_ainvoke_corrupt_cache_treated_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    filepath = tmp_path / "a.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(b"not a valid pickle")

    def raising_load(_path: Path) -> None:
        msg = "corrupt pickle"
        raise ValueError(msg)

    monkeypatch.setattr(f"{MODULE}.load_pickle", raising_load)

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("a"))

    assert result == "A"
    assert inner.ainvoke_calls == ["a"]
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_runnable_ainvoke_write_failure_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def raising_save(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{MODULE}.save_pickle", raising_save)
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("a"))

    assert result == "A"
    assert not (tmp_path / "a.pkl").exists()
    assert any("Failed to write cache" in message for message in caplog.messages)


def test_caching_runnable_ainvoke_propagates_inner_exception(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="a"), cache_dir=tmp_path, key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for a"):
        asyncio.run(cached.ainvoke("a"))


# --- batch ---


def test_caching_runnable_batch_empty_list_returns_empty_list(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    assert cached.batch([]) == []


def test_caching_runnable_batch_no_cache_dir_delegates_to_inner_batch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=None)
    results = cached.batch(["a", "b"])
    assert results == ["A", "B"]
    assert inner.batch_calls == [["a", "b"]]


def test_caching_runnable_batch_returns_correct_results(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    assert cached.batch(["a", "b", "c"]) == ["A", "B", "C"]


def test_caching_runnable_batch_calls_inner_batch_only_for_misses(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    cached.invoke("a")  # pre-populate the cache for "a"
    inner.invoke_calls.clear()
    inner.batch_calls.clear()

    results = cached.batch(["a", "b", "c"])

    assert results == ["A", "B", "C"]
    # Only the misses ("b", "c") should be sent to inner.batch, as one
    # call -- not one call per miss, and "a" should not appear at all.
    assert inner.batch_calls == [["b", "c"]]


def test_caching_runnable_batch_writes_cache_for_new_misses(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    cached.batch(["a", "b"])
    assert (tmp_path / "a.pkl").is_file()
    assert (tmp_path / "b.pkl").is_file()


def test_caching_runnable_batch_second_call_all_hits(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    cached.batch(["a", "b"])
    inner.batch_calls.clear()

    results = cached.batch(["a", "b"])

    assert results == ["A", "B"]
    assert inner.batch_calls == []


def test_caching_runnable_batch_raises_by_default_on_failure(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache_dir=tmp_path, key_fn=_identity_key
    )
    with pytest.raises(ValueError, match="failed for b"):
        cached.batch(["a", "b", "c"])


def test_caching_runnable_batch_return_exceptions_keeps_raw_exception(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache_dir=tmp_path, key_fn=_identity_key
    )
    results = cached.batch(["a", "b", "c"], return_exceptions=True)
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


def test_caching_runnable_batch_return_exceptions_does_not_cache_failure(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache_dir=tmp_path, key_fn=_identity_key
    )
    cached.batch(["a", "b", "c"], return_exceptions=True)
    assert not (tmp_path / "b.pkl").exists()
    assert (tmp_path / "a.pkl").exists()
    assert (tmp_path / "c.pkl").exists()


# --- abatch ---


def test_caching_runnable_abatch_empty_list_returns_empty_list(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=RunnableLambda(lambda x: x.upper()), cache_dir=tmp_path, key_fn=_identity_key
    )
    assert asyncio.run(cached.abatch([])) == []


def test_caching_runnable_abatch_no_cache_dir_delegates_to_inner_abatch() -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=None)
    results = asyncio.run(cached.abatch(["a", "b"]))
    assert results == ["A", "B"]
    assert inner.abatch_calls == [["a", "b"]]


def test_caching_runnable_abatch_calls_inner_abatch_only_for_misses(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    asyncio.run(cached.ainvoke("a"))  # pre-populate the cache for "a"
    inner.ainvoke_calls.clear()
    inner.abatch_calls.clear()

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", "B", "C"]
    assert inner.abatch_calls == [["b", "c"]]


def test_caching_runnable_abatch_second_call_all_hits(tmp_path: Path) -> None:
    inner = TrackingRunnable()
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)
    asyncio.run(cached.abatch(["a", "b"]))
    inner.abatch_calls.clear()

    results = asyncio.run(cached.abatch(["a", "b"]))

    assert results == ["A", "B"]
    # No misses at all this time, so the "if miss_indices:" branch in
    # abatch must be skipped entirely -- inner.abatch must not be called.
    assert inner.abatch_calls == []


def test_caching_runnable_abatch_return_exceptions_keeps_raw_exception(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache_dir=tmp_path, key_fn=_identity_key
    )
    results = asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert results[0] == "A"
    assert isinstance(results[1], ValueError)
    assert results[2] == "C"


def test_caching_runnable_abatch_return_exceptions_does_not_cache_failure(tmp_path: Path) -> None:
    cached = CachingRunnable(
        runnable=_failing_lambda(fail_on="b"), cache_dir=tmp_path, key_fn=_identity_key
    )
    asyncio.run(cached.abatch(["a", "b", "c"], return_exceptions=True))
    assert not (tmp_path / "b.pkl").exists()


# --- ignore_none: default (False) ---


def test_caching_runnable_ignore_none_false_caches_none_result(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(runnable=inner, cache_dir=tmp_path, key_fn=_identity_key)

    result1 = cached.invoke("a")
    assert result1 is None
    assert (tmp_path / "a.pkl").is_file()

    inner.invoke_calls.clear()
    result2 = cached.invoke("a")
    assert result2 is None
    assert inner.invoke_calls == []  # cache hit, no call to inner


# --- ignore_none: True ---


def test_caching_runnable_ignore_none_true_does_not_cache_none_result(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = cached.invoke("a")

    assert result is None
    assert not (tmp_path / "a.pkl").exists()


def test_caching_runnable_ignore_none_true_none_result_is_a_miss_next_call(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    cached.invoke("a")
    inner.invoke_calls.clear()
    cached.invoke("a")

    # Nothing was ever cached, so the second call must also hit the inner
    # runnable.
    assert inner.invoke_calls == ["a"]


def test_caching_runnable_ignore_none_true_existing_none_cache_treated_as_miss(
    tmp_path: Path,
) -> None:
    import pickle

    (tmp_path / "planted.pkl").write_bytes(pickle.dumps(None))
    inner = TrackingRunnable()
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = cached.invoke("planted")

    assert result == "PLANTED"
    assert inner.invoke_calls == ["planted"]


def test_caching_runnable_ignore_none_true_existing_none_cache_treated_as_miss_ainvoke(
    tmp_path: Path,
) -> None:
    import pickle

    (tmp_path / "planted.pkl").write_bytes(pickle.dumps(None))
    inner = TrackingRunnable()
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = asyncio.run(cached.ainvoke("planted"))

    assert result == "PLANTED"
    assert inner.ainvoke_calls == ["planted"]


def test_caching_runnable_ignore_none_true_non_none_results_still_cached(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="skip-me")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    cached.invoke("a")
    assert (tmp_path / "a.pkl").is_file()

    inner.invoke_calls.clear()
    result = cached.invoke("a")
    assert result == "A"
    assert inner.invoke_calls == []  # cache hit


def test_caching_runnable_ignore_none_true_batch_mixed_results(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    results = cached.batch(["a", "b", "c"])

    assert results == ["A", None, "C"]
    assert (tmp_path / "a.pkl").is_file()
    assert not (tmp_path / "b.pkl").exists()
    assert (tmp_path / "c.pkl").is_file()


def test_caching_runnable_ignore_none_true_abatch_mixed_results(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="b")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    results = asyncio.run(cached.abatch(["a", "b", "c"]))

    assert results == ["A", None, "C"]
    assert (tmp_path / "a.pkl").is_file()
    assert not (tmp_path / "b.pkl").exists()
    assert (tmp_path / "c.pkl").is_file()


def test_caching_runnable_ignore_none_true_ainvoke_does_not_cache_none(tmp_path: Path) -> None:
    inner = TrackingRunnable(none_on="a")
    cached = CachingRunnable(
        runnable=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = asyncio.run(cached.ainvoke("a"))

    assert result is None
    assert not (tmp_path / "a.pkl").exists()
