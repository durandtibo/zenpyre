from __future__ import annotations

import asyncio
from typing import Any

import pydantic
import pytest
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from zenpyre.chat_models import CachingChatModel
from zenpyre.utils.imports import is_persista_available

if is_persista_available():
    from persista.cache import Cache

pytest.importorskip("persista")


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


def test_caching_chat_model_response_cache_none_disables_caching() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached.response_cache is None


def test_caching_chat_model_is_base_chat_model() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert isinstance(cached, BaseChatModel)


# --- _identifying_params ---


def test_caching_chat_model_identifying_params_includes_inner_params() -> None:
    with Cache() as cache:
        cached = CachingChatModel(
            chat_model=FakeListChatModel(responses=["a"]), response_cache=cache
        )
        assert cached._identifying_params == {
            "chat_model": {"responses": ["a"]},
            "response_cache": cache,
        }


def test_caching_chat_model_identifying_params_response_cache_none() -> None:
    cached = CachingChatModel(chat_model=FakeListChatModel(responses=["a"]))
    assert cached._identifying_params == {
        "chat_model": {"responses": ["a"]},
        "response_cache": None,
    }


def test_caching_chat_model_identifying_params_inner_without_identifying_params() -> None:
    with Cache() as cache:
        cached = CachingChatModel(
            chat_model=TrackingChatModel(responses=["a"]), response_cache=cache
        )
        assert cached._identifying_params == {"chat_model": {}, "response_cache": cache}


# --- invoke: caching disabled ---


def test_caching_chat_model_invoke_no_cache_always_calls_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, response_cache=None)
    assert cached.invoke("hi").content == "A"
    assert cached.invoke("hi").content == "B"
    assert len(inner.calls) == 2


# --- invoke: caching enabled ---


def test_caching_chat_model_invoke_cache_miss_calls_inner() -> None:
    inner = TrackingChatModel(responses=["A"])
    with Cache() as cache:
        cached = CachingChatModel(chat_model=inner, response_cache=cache, key_fn=_identity_key)
        assert cached.invoke("hi").content == "A"
        assert len(inner.calls) == 1


def test_caching_chat_model_invoke_cache_hit_does_not_call_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    with Cache() as cache:
        cached = CachingChatModel(chat_model=inner, response_cache=cache, key_fn=_identity_key)
        cached.invoke("hi")
        assert cached.invoke("hi").content == "A"
        assert len(inner.calls) == 1


def test_caching_chat_model_invoke_writes_cache_entry() -> None:
    with Cache() as cache:
        cached = CachingChatModel(
            chat_model=FakeListChatModel(responses=["A"]),
            response_cache=cache,
            key_fn=_identity_key,
        )
        cached.invoke("hi")
        assert cache.try_get("k")[0]


def test_caching_chat_model_invoke_caches_empty_string_result() -> None:
    # persista's Cache no longer has an `ignore_none` policy -- any
    # result, including a falsy one, is cached and served back as-is.
    inner = TrackingChatModel(responses=["", "B"])
    with Cache() as cache:
        cached = CachingChatModel(chat_model=inner, response_cache=cache, key_fn=_identity_key)
        cached.invoke("hi")
        assert cached.invoke("hi").content == ""
        assert len(inner.calls) == 1


# --- ainvoke ---


def test_caching_chat_model_ainvoke_no_cache_uses_inner() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    cached = CachingChatModel(chat_model=inner, response_cache=None)
    result = asyncio.run(cached.ainvoke("hi"))
    assert result.content == "A"
    assert len(inner.calls) == 1


def test_caching_chat_model_ainvoke_cache_miss_then_hit() -> None:
    inner = TrackingChatModel(responses=["A", "B"])
    with Cache() as cache:
        cached = CachingChatModel(chat_model=inner, response_cache=cache, key_fn=_identity_key)

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
    with Cache() as cache:
        cached = CachingChatModel(chat_model=inner, response_cache=cache)
        bound = cached.bind_tools([my_tool])
        assert isinstance(bound, CachingChatModel)
        assert bound.response_cache is cached.response_cache
        assert bound.chat_model is not inner


def test_caching_chat_model_accepts_bound_runnable_as_chat_model() -> None:
    # Regression test: real integrations' bind_tools (e.g. ChatOpenAI's)
    # return `super().bind(...)`, a RunnableBinding, not a BaseChatModel
    # instance -- the `chat_model` field must accept that directly via
    # the constructor, not just via bind_tools' own model_copy.
    inner = ToolBindableChatModel()
    bound = inner.bind_tools([])
    with Cache() as cache:
        cached = CachingChatModel(chat_model=bound, response_cache=cache, key_fn=_identity_key)
        assert cached.invoke("hi").content == "A"
