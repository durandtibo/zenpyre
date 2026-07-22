from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pydantic
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from persista.cache import Cache

from zenpyre.chat_models import CachingChatModel

if TYPE_CHECKING:
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


def test_caching_chat_model_default_ignore_none() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]), result_cache=Cache())
    assert cached.ignore_none is False


def test_caching_chat_model_result_cache_none_disables_caching() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached.result_cache is None


def test_caching_chat_model_is_base_chat_model() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert isinstance(cached, BaseChatModel)


# --- _identifying_params ---


def test_caching_chat_model_identifying_params_includes_inner_params() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]), result_cache=Cache())
    assert cached._identifying_params == {
        "chat_model": {"responses": ["a"]},
        "result_cache": True,
    }


def test_caching_chat_model_identifying_params_result_cache_none() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached._identifying_params == {"chat_model": {"responses": ["a"]}, "result_cache": False}


def test_caching_chat_model_identifying_params_inner_without_identifying_params() -> None:
    cached = CachingChatModel(chat_model=TrackingChatModel(responses=["a"]), result_cache=Cache())
    assert cached._identifying_params == {"chat_model": {}, "result_cache": True}


# --- invoke: caching disabled ---


def test_caching_chat_model_invoke_no_result_cache_always_calls_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, result_cache=None)
    assert cached.invoke("hi").content == "A"
    assert cached.invoke("hi").content == "B"
    assert len(inner.calls) == 2


# --- invoke: caching enabled ---


def test_caching_chat_model_invoke_cache_miss_calls_inner() -> None:
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, result_cache=Cache(), key_fn=_identity_key)
    assert cached.invoke("hi").content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_invoke_cache_hit_does_not_call_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, result_cache=Cache(), key_fn=_identity_key)
    cached.invoke("hi")
    assert cached.invoke("hi").content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_invoke_writes_cache_entry() -> None:
    cache = Cache()
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), result_cache=cache, key_fn=_identity_key
    )
    cached.invoke("hi")
    assert cache.get("k") is not None


def test_caching_chat_model_invoke_corrupt_cache_treated_as_miss(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingChatModel(responses=["A"])
    cache = Cache()
    cached = CachingChatModel(chat_model=inner, result_cache=cache, key_fn=_identity_key)

    def raising_get(_key: str) -> None:
        msg = "corrupt cache entry"
        raise ValueError(msg)

    monkeypatch.setattr(cache, "get", raising_get)

    with caplog.at_level("WARNING"):
        result = cached.invoke("hi")

    assert result.content == "A"
    assert len(inner.calls) == 1
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_chat_model_invoke_write_failure_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = Cache()

    def raising_set(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "backing store unavailable"
        raise OSError(msg)

    cache.set = raising_set
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), result_cache=cache, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = cached.invoke("hi")

    assert result.content == "A"
    assert any("Failed to write cache" in message for message in caplog.messages)


# --- ainvoke ---


def test_caching_chat_model_ainvoke_no_result_cache_uses_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, result_cache=None)
    result = asyncio.run(cached.ainvoke("hi"))
    assert result.content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_ainvoke_cache_miss_then_hit() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, result_cache=Cache(), key_fn=_identity_key)

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


def test_caching_chat_model_bind_tools_returns_new_caching_chat_model() -> None:
    def my_tool(x: int) -> int:
        """Double x."""
        return x * 2

    inner = ToolBindableChatModel()
    cache = Cache()
    cached = CachingChatModel(chat_model=inner, result_cache=cache)
    bound = cached.bind_tools([my_tool])
    assert isinstance(bound, CachingChatModel)
    assert bound.result_cache is cached.result_cache
    assert bound.chat_model is not inner


def test_caching_chat_model_accepts_bound_runnable_as_chat_model() -> None:
    # Regression test: real integrations' bind_tools (e.g. ChatOpenAI's)
    # return `super().bind(...)`, a RunnableBinding, not a BaseChatModel
    # instance -- the `chat_model` field must accept that directly via
    # the constructor, not just via bind_tools' own model_copy.
    inner = ToolBindableChatModel()
    bound = inner.bind_tools([])
    cached = CachingChatModel(chat_model=bound, result_cache=Cache(), key_fn=_identity_key)
    assert cached.invoke("hi").content == "A"


# --- ignore_none ---


def test_caching_chat_model_invoke_ignore_none_true_does_not_cache_none_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache()
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]),
        result_cache=cache,
        key_fn=_identity_key,
        ignore_none=True,
    )
    monkeypatch.setattr(cached, "_call_chat_model", lambda *args, **kwargs: None)  # noqa: ARG005

    result = cached._generate(["hi"])

    assert result is None
    assert cache.get("k") is None


def test_caching_chat_model_invoke_existing_none_cache_treated_as_miss() -> None:
    cache = Cache()
    cache.set("k", None)
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, result_cache=cache, key_fn=_identity_key)

    result = cached.invoke("hi")

    assert result.content == "A"
    assert len(inner.calls) == 1


# --- ainvoke: corrupt cache / write failure ---


def test_caching_chat_model_ainvoke_corrupt_cache_treated_as_miss(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    inner = TrackingChatModel(responses=["A"])
    cache = Cache()
    cached = CachingChatModel(chat_model=inner, result_cache=cache, key_fn=_identity_key)

    def raising_get(_key: str) -> None:
        msg = "corrupt cache entry"
        raise ValueError(msg)

    monkeypatch.setattr(cache, "get", raising_get)

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert len(inner.calls) == 1
    assert any("Failed to load cache" in message for message in caplog.messages)


def test_caching_chat_model_ainvoke_write_failure_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = Cache()

    def raising_set(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        msg = "backing store unavailable"
        raise OSError(msg)

    cache.set = raising_set
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]), result_cache=cache, key_fn=_identity_key
    )

    with caplog.at_level("WARNING"):
        result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert any("Failed to write cache" in message for message in caplog.messages)


def test_caching_chat_model_ainvoke_ignore_none_true_does_not_cache_none_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = Cache()
    cached = CachingChatModel(
        chat_model=FakeListChatModel(responses=["A"]),
        result_cache=cache,
        key_fn=_identity_key,
        ignore_none=True,
    )

    async def no_op_acall(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(cached, "_acall_chat_model", no_op_acall)

    result = asyncio.run(cached._agenerate(["hi"]))

    assert result is None
    assert cache.get("k") is None


def test_caching_chat_model_ainvoke_existing_none_cache_treated_as_miss() -> None:
    cache = Cache()
    cache.set("k", None)
    inner = TrackingChatModel(responses=["A"])
    cached = CachingChatModel(chat_model=inner, result_cache=cache, key_fn=_identity_key)

    result = asyncio.run(cached.ainvoke("hi"))

    assert result.content == "A"
    assert len(inner.calls) == 1
