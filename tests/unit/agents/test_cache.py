from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from coola.hashing import hash_object
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from zenpyre.agents import RunnableWithCache

MODULE = "zenpyre.agents.cache"


@pytest.fixture
def runnable() -> Runnable:
    return RunnableLambda(lambda x: x.upper())


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def make_counting_runnable() -> tuple[Runnable, Mock]:
    mock = Mock(side_effect=lambda x: x.upper())
    return RunnableLambda(mock), mock


#####################################
#   Tests for RunnableWithCache     #
#####################################

# --- Constructor ---


def test_runnable_with_cache_stores_runnable(runnable: Runnable) -> None:
    assert RunnableWithCache(runnable=runnable)._runnable is runnable


def test_runnable_with_cache_cache_dir_none_by_default(runnable: Runnable) -> None:
    assert RunnableWithCache(runnable=runnable)._cache_dir is None


def test_runnable_with_cache_cache_dir_stored_as_path(runnable: Runnable, cache_dir: Path) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=str(cache_dir))
    assert isinstance(cached._cache_dir, Path)


def test_runnable_with_cache_default_key_fn_is_hash_object(runnable: Runnable) -> None:
    assert RunnableWithCache(runnable=runnable)._key_fn is hash_object


def test_runnable_with_cache_stores_custom_key_fn(runnable: Runnable) -> None:
    key_fn = Mock()
    assert RunnableWithCache(runnable=runnable, key_fn=key_fn)._key_fn is key_fn


# --- repr and str ---


def test_runnable_with_cache_repr(runnable: Runnable) -> None:
    assert repr(RunnableWithCache(runnable=runnable)).startswith("RunnableWithCache(")


def test_runnable_with_cache_str(runnable: Runnable) -> None:
    assert str(RunnableWithCache(runnable=runnable)).startswith("RunnableWithCache(")


# --- invoke: no cache_dir ---


def test_runnable_with_cache_invoke_no_cache_dir_returns_result(runnable: Runnable) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=None)
    assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_no_cache_dir_calls_runnable_every_time() -> None:
    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=None)
    cached.invoke("hello")
    cached.invoke("hello")
    assert mock.call_count == 2


# --- invoke: cache miss ---


def test_runnable_with_cache_invoke_cache_miss_returns_result(
    cache_dir: Path, runnable: Runnable
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_cache_miss_calls_runnable(cache_dir: Path) -> None:
    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    assert mock.call_count == 1


def test_runnable_with_cache_invoke_cache_miss_writes_cache_file(
    cache_dir: Path, runnable: Runnable
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    assert any(cache_dir.glob("*.pkl"))


def test_runnable_with_cache_invoke_creates_cache_dir_if_missing(
    cache_dir: Path, runnable: Runnable
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    assert cache_dir.is_dir()


def test_runnable_with_cache_invoke_nested_cache_dir_created(
    tmp_path: Path, runnable: Runnable
) -> None:
    nested = tmp_path / "a" / "b" / "c"
    cached = RunnableWithCache(runnable=runnable, cache_dir=nested)
    cached.invoke("hello")
    assert any(nested.glob("*.pkl"))


def test_runnable_with_cache_invoke_different_inputs_different_cache_files(
    cache_dir: Path,
) -> None:
    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    cached.invoke("world")
    assert mock.call_count == 2
    assert len(list(cache_dir.glob("*.pkl"))) == 2


# --- invoke: cache hit ---


def test_runnable_with_cache_invoke_cache_hit_returns_result(
    cache_dir: Path, runnable: Runnable
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_cache_hit_skips_runnable(cache_dir: Path) -> None:
    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    cached.invoke("hello")
    assert mock.call_count == 1


def test_runnable_with_cache_invoke_cache_hit_does_not_call_runnable(cache_dir: Path) -> None:
    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")

    mock.side_effect = AssertionError("should not be called")
    assert cached.invoke("hello") == "HELLO"


# --- key_fn ---


def test_runnable_with_cache_invoke_uses_key_fn(cache_dir: Path, runnable: Runnable) -> None:
    key_fn = Mock(return_value="custom-key")
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir, key_fn=key_fn)
    cached.invoke("hello")
    key_fn.assert_called_once_with("hello")


def test_runnable_with_cache_invoke_uses_key_fn_for_cache_filename(
    cache_dir: Path, runnable: Runnable
) -> None:
    key_fn = Mock(return_value="custom-key")
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir, key_fn=key_fn)
    cached.invoke("hello")
    assert (cache_dir / "custom-key.pkl").is_file()


# --- corrupted cache ---


def test_runnable_with_cache_invoke_corrupted_cache_returns_result(
    cache_dir: Path, runnable: Runnable
) -> None:
    cache_dir.mkdir(parents=True)
    filepath = (cache_dir / hash_object("hello")).with_suffix(".pkl")
    filepath.write_bytes(b"not a valid pickle")
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_corrupted_cache_calls_runnable(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    filepath = (cache_dir / hash_object("hello")).with_suffix(".pkl")
    filepath.write_bytes(b"not a valid pickle")

    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")
    assert mock.call_count == 1


def test_runnable_with_cache_invoke_corrupted_cache_overwrites_with_fresh_result(
    cache_dir: Path,
) -> None:
    cache_dir.mkdir(parents=True)
    filepath = (cache_dir / hash_object("hello")).with_suffix(".pkl")
    filepath.write_bytes(b"not a valid pickle")

    runnable, mock = make_counting_runnable()
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    cached.invoke("hello")

    mock.side_effect = AssertionError("should not be called")
    assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_load_failure_logs_warning(
    cache_dir: Path, runnable: Runnable, caplog: pytest.LogCaptureFixture
) -> None:
    cache_dir.mkdir(parents=True)
    filepath = (cache_dir / hash_object("hello")).with_suffix(".pkl")
    filepath.write_bytes(b"not a valid pickle")

    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    with caplog.at_level("WARNING"):
        cached.invoke("hello")
    assert any("Failed to load cache" in record.message for record in caplog.records)


# --- save failure ---


def test_runnable_with_cache_invoke_save_failure_returns_result(
    cache_dir: Path, runnable: Runnable
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    with patch(f"{MODULE}.save_pickle", side_effect=OSError("disk full")):
        assert cached.invoke("hello") == "HELLO"


def test_runnable_with_cache_invoke_save_failure_logs_warning(
    cache_dir: Path, runnable: Runnable, caplog: pytest.LogCaptureFixture
) -> None:
    cached = RunnableWithCache(runnable=runnable, cache_dir=cache_dir)
    with (
        patch(f"{MODULE}.save_pickle", side_effect=OSError("disk full")),
        caplog.at_level("WARNING"),
    ):
        cached.invoke("hello")
    assert any("Failed to write cache" in record.message for record in caplog.records)


# --- passthrough of config and kwargs ---


def test_runnable_with_cache_invoke_passes_config_through(cache_dir: Path) -> None:
    received_config = {}

    def fn(x: str, config: RunnableConfig | None = None) -> str:
        received_config["config"] = config
        return x.upper()

    cached = RunnableWithCache(runnable=RunnableLambda(fn), cache_dir=cache_dir)
    config = RunnableConfig(tags=["test"])
    cached.invoke("hello", config=config)
    assert received_config["config"] is not None


def test_runnable_with_cache_invoke_passes_kwargs_through(cache_dir: Path) -> None:
    received_kwargs = {}

    def fn(x: str, **kwargs: Any) -> str:
        received_kwargs.update(kwargs)
        return x.upper()

    cached = RunnableWithCache(runnable=RunnableLambda(fn), cache_dir=cache_dir)
    cached.invoke("hello", extra_flag=True)
    assert received_kwargs == {"extra_flag": True}
