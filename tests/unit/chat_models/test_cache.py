from __future__ import annotations

import asyncio
import pickle
from typing import TYPE_CHECKING, Any

import pydantic
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from zenpyre.chat_models import CachingChatModel

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

MODULE = "zenpyre.chat_models.cache"


def _identity_key(_x: Any) -> str:
    return "k"


class TrackingChatModel(BaseChatModel):
    """A chat model that records how many times it was called, so tests
    can verify ``CachingChatModel`` skips the wrapped model on a cache
    hit."""

    responses: list[str]
    calls: list[Any] = pydantic.Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tracking"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        self.calls.append(messages)
        content = self.responses[len(self.calls) - 1]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


##################################
#   Tests for CachingChatModel   #
##################################


# --- constructor ---


def test_caching_chat_model_default_ignore_none(tmp_path: Path) -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]), cache_dir=tmp_path)
    assert cached.ignore_none is False


def test_caching_chat_model_cache_dir_none_disables_caching() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached.cache_dir is None


def test_caching_chat_model_is_base_chat_model() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert isinstance(cached, BaseChatModel)


# --- _identifying_params ---


def test_caching_chat_model_identifying_params_includes_inner_params(tmp_path: Path) -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]), cache_dir=tmp_path)
    assert cached._identifying_params == {
        "chat_model": {"responses": ["a"]},
        "cache_dir": tmp_path,
    }


def test_caching_chat_model_identifying_params_cache_dir_none() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached._identifying_params == {"chat_model": {"responses": ["a"]}, "cache_dir": None}


def test_caching_chat_model_identifying_params_inner_without_identifying_params(
    tmp_path: Path,
) -> None:
    cached = CachingChatModel(chat_model=TrackingChatModel(responses=["a"]), cache_dir=tmp_path)
    assert cached._identifying_params == {"chat_model": {}, "cache_dir": tmp_path}


# --- invoke: caching disabled ---


def test_caching_chat_model_invoke_no_cache_dir_always_calls_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, cache_dir=None)
    assert cached.invoke("hi").content == "A"
    assert cached.invoke("hi").content == "B"
    assert len(inner.calls) == 2


# --- invoke: caching enabled ---


def test_caching_chat_model_invoke_cache_miss_calls_inner(tmp_path: Path) -> None:
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key)
    assert cached.invoke("hi").content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_invoke_cache_hit_does_not_call_inner(tmp_path: Path) -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key)
    cached.invoke("hi")
    assert cached.invoke("hi").content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_invoke_writes_cache_file(tmp_path: Path) -> None:
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), cache_dir=tmp_path, key_fn=_identity_key
    )
    cached.invoke("hi")
    assert (tmp_path / "k.pkl").is_file()


def test_caching_chat_model_invoke_corrupt_cache_treated_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key)
    filepath = tmp_path / "k.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(b"not a valid pickle")

    def raising_load(_path: Path) -> None:
        msg = "corrupt pickle"
        raise ValueError(msg)

    monkeypatch.setattr(f"{MODULE}.load_pickle", raising_load)

    with caplog.at_level("WARNING"):
        result = cached.invoke("hi")

    assert result.content == "A"
    assert len(inner.calls) == 1
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_chat_model_invoke_write_failure_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def raising_save(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{MODULE}.save_pickle", raising_save)
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), cache_dir=tmp_path, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = cached.invoke("hi")

    assert result.content == "A"
    assert not (tmp_path / "k.pkl").exists()
    assert any("Failed to write cache" in message for message in caplog.messages)


# --- ainvoke ---


def test_caching_chat_model_ainvoke_no_cache_dir_uses_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, cache_dir=None)
    result = asyncio.run(cached.ainvoke("hi"))
    assert result.content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_ainvoke_cache_miss_then_hit(tmp_path: Path) -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key)

    result1 = asyncio.run(cached.ainvoke("hi"))
    assert result1.content == "A"

    result2 = asyncio.run(cached.ainvoke("hi"))
    assert result2.content == "A"
    assert len(inner.calls) == 1


# --- bind_tools ---


class ToolBindableChatModel(BaseChatModel):
    """A chat model whose ``bind_tools`` behaves like a real
    integration's -- returning a bound runnable rather than raising
    ``NotImplementedError`` like ``BaseChatModel``'s default."""

    @property
    def _llm_type(self) -> str:
        return "tool-bindable"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="A"))])

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> Any:
        return self.bind(tools=tools, **kwargs)


def test_caching_chat_model_bind_tools_returns_new_caching_chat_model(tmp_path: Path) -> None:
    def my_tool(x: int) -> int:
        """Double x."""
        return x * 2

    inner = ToolBindableChatModel()
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path)
    bound = cached.bind_tools([my_tool])
    assert isinstance(bound, CachingChatModel)
    assert bound.cache_dir == cached.cache_dir
    assert bound.chat_model is not inner


def test_caching_chat_model_accepts_bound_runnable_as_chat_model(tmp_path: Path) -> None:
    # Regression test: real integrations' bind_tools (e.g. ChatOpenAI's)
    # return `super().bind(...)`, a RunnableBinding, not a BaseChatModel
    # instance -- the `chat_model` field must accept that directly via
    # the constructor, not just via bind_tools' own model_copy.
    inner = ToolBindableChatModel()
    bound = inner.bind_tools([])
    cached = CachingChatModel(chat_model=bound, cache_dir=tmp_path, key_fn=_identity_key)
    assert cached.invoke("hi").content == "A"


# --- ignore_none ---


def test_caching_chat_model_invoke_ignore_none_true_does_not_cache_none_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]),
        cache_dir=tmp_path,
        key_fn=_identity_key,
        ignore_none=True,
    )
    monkeypatch.setattr(cached, "_call_chat_model", lambda *args, **kwargs: None)  # noqa: ARG005

    result = cached._generate(["hi"])

    assert result is None
    assert not (tmp_path / "k.pkl").exists()


def test_caching_chat_model_invoke_ignore_none_true_existing_none_cache_treated_as_miss(
    tmp_path: Path,
) -> None:
    (tmp_path / "k.pkl").write_bytes(pickle.dumps(None))
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(
        chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = cached.invoke("hi")

    assert result.content == "A"
    assert len(inner.calls) == 1


# --- ainvoke: corrupt cache / write failure ---


def test_caching_chat_model_ainvoke_corrupt_cache_treated_as_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key)
    filepath = tmp_path / "k.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(b"not a valid pickle")

    def raising_load(_path: Path) -> None:
        msg = "corrupt pickle"
        raise ValueError(msg)

    monkeypatch.setattr(f"{MODULE}.load_pickle", raising_load)

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert len(inner.calls) == 1
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_chat_model_ainvoke_write_failure_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def raising_save(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(f"{MODULE}.save_pickle", raising_save)
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), cache_dir=tmp_path, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert not (tmp_path / "k.pkl").exists()
    assert any("Failed to write cache" in message for message in caplog.messages)


def test_caching_chat_model_ainvoke_ignore_none_true_does_not_cache_none_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]),
        cache_dir=tmp_path,
        key_fn=_identity_key,
        ignore_none=True,
    )

    async def no_op_acall(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(cached, "_acall_chat_model", no_op_acall)

    result = asyncio.run(cached._agenerate(["hi"]))

    assert result is None
    assert not (tmp_path / "k.pkl").exists()


def test_caching_chat_model_ainvoke_ignore_none_true_existing_none_cache_treated_as_miss(
    tmp_path: Path,
) -> None:
    (tmp_path / "k.pkl").write_bytes(pickle.dumps(None))
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(
        chat_model=inner, cache_dir=tmp_path, key_fn=_identity_key, ignore_none=True
    )

    result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert len(inner.calls) == 1
