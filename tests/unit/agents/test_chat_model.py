r"""Contain unit tests for ``AgentChatModel``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from zenpyre.agents.chat_model import AgentChatModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


def _mock_model(response: AIMessage | None = None) -> MagicMock:
    """Return a ``MagicMock`` standing in for a ``BaseChatModel``.

    Args:
        response: The ``AIMessage`` returned by ``invoke``/``ainvoke``.
            Defaults to ``AIMessage(content="hello")``.

    Returns:
        A mock configured with ``invoke``, ``ainvoke``, ``batch``, and
        ``abatch`` methods that return/await ``response`` (or a list of
        ``response`` for the batch methods).
    """
    response = response if response is not None else AIMessage(content="hello")
    model = MagicMock()
    model.invoke.return_value = response
    model.ainvoke = AsyncMock(return_value=response)
    return model


####################################
#     Tests for AgentChatModel     #
####################################


# --- Constructor ---


def test_agent_chat_model_stores_model() -> None:
    model = _mock_model()
    agent = AgentChatModel(model=model)
    assert agent.model is model


def test_agent_chat_model_default_system_prompt() -> None:
    agent = AgentChatModel(model=_mock_model())
    assert agent.system_prompt is None


def test_agent_chat_model_stores_system_prompt() -> None:
    agent = AgentChatModel(model=_mock_model(), system_prompt="You are helpful.")
    assert agent.system_prompt == "You are helpful."


def test_agent_chat_model_default_response_format() -> None:
    agent = AgentChatModel(model=_mock_model())
    assert agent.response_format is None


def test_agent_chat_model_default_structured_model_is_none() -> None:
    agent = AgentChatModel(model=_mock_model())
    assert agent._structured_model is None


def test_agent_chat_model_response_format_binds_structured_model() -> None:
    model = _mock_model()
    structured = MagicMock()
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)

    model.with_structured_output.assert_called_once_with(dict, include_raw=True)
    assert agent._structured_model is structured


# --- _coerce_input ---


def test_coerce_input_str_wraps_in_human_message() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input("hi")
    assert result == [HumanMessage(content="hi")]


def test_coerce_input_list_of_strings() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input(["hi", "there"])
    assert result == [HumanMessage(content="hi"), HumanMessage(content="there")]


def test_coerce_input_list_of_base_messages_untouched() -> None:
    agent = AgentChatModel(model=_mock_model())
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = agent._coerce_input(messages)
    assert result == messages
    assert result is not messages


def test_coerce_input_list_mixed_strings_and_messages() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input(["hi", AIMessage(content="hello")])
    assert result == [HumanMessage(content="hi"), AIMessage(content="hello")]


def test_coerce_input_dict_with_messages_key() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input({"messages": ["hi"]})
    assert result == [HumanMessage(content="hi")]


def test_coerce_input_dict_missing_messages_key_defaults_to_empty() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input({})
    assert result == []


def test_coerce_input_unsupported_type_raises() -> None:
    agent = AgentChatModel(model=_mock_model())
    with pytest.raises(TypeError, match="Unsupported input type for AgentChatModel"):
        agent._coerce_input(42)  # type: ignore[arg-type]


def test_coerce_input_prepends_system_prompt() -> None:
    agent = AgentChatModel(model=_mock_model(), system_prompt="Be nice.")
    result = agent._coerce_input("hi")
    assert result == [SystemMessage(content="Be nice."), HumanMessage(content="hi")]


def test_coerce_input_does_not_duplicate_existing_system_message() -> None:
    agent = AgentChatModel(model=_mock_model(), system_prompt="Be nice.")
    result = agent._coerce_input([SystemMessage(content="Custom"), HumanMessage(content="hi")])
    assert result == [SystemMessage(content="Custom"), HumanMessage(content="hi")]


def test_coerce_input_no_system_prompt_does_not_prepend() -> None:
    agent = AgentChatModel(model=_mock_model())
    result = agent._coerce_input("hi")
    assert result == [HumanMessage(content="hi")]


def test_coerce_input_returns_new_list_each_call() -> None:
    agent = AgentChatModel(model=_mock_model())
    messages = [HumanMessage(content="hi")]
    result1 = agent._coerce_input(messages)
    result2 = agent._coerce_input(messages)
    assert result1 is not result2


# --- invoke ---


def test_invoke_returns_messages_and_output() -> None:
    model = _mock_model(response=AIMessage(content="hello there"))
    agent = AgentChatModel(model=model)

    result = agent.invoke("hi")

    assert result == {
        "messages": [HumanMessage(content="hi"), AIMessage(content="hello there")],
        "output": "hello there",
    }


def test_invoke_calls_model_with_coerced_messages() -> None:
    model = _mock_model()
    captured: list = []
    model.invoke.side_effect = lambda messages, **kwargs: (  # noqa: ARG005
        captured.append(list(messages)) or AIMessage(content="hello")
    )
    agent = AgentChatModel(model=model)

    agent.invoke("hi")

    assert captured[0] == [HumanMessage(content="hi")]


def test_invoke_forwards_config_and_kwargs() -> None:
    model = _mock_model()
    captured_kwargs: list = []
    model.invoke.side_effect = lambda messages, **kwargs: (  # noqa: ARG005
        captured_kwargs.append(kwargs) or AIMessage(content="hello")
    )
    agent = AgentChatModel(model=model)
    config = {"tags": ["test"]}

    agent.invoke("hi", config=config, temperature=0.5)

    assert captured_kwargs[0]["config"] == config
    assert captured_kwargs[0]["temperature"] == 0.5


def test_invoke_prepends_system_prompt_to_returned_messages() -> None:
    model = _mock_model(response=AIMessage(content="hello"))
    agent = AgentChatModel(model=model, system_prompt="Be nice.")

    result = agent.invoke("hi")

    assert result["messages"] == [
        SystemMessage(content="Be nice."),
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]


def test_invoke_with_structured_output_includes_structured_response() -> None:
    model = _mock_model()
    ai_message = AIMessage(content="42")
    structured = MagicMock()
    structured.invoke.return_value = {"raw": ai_message, "parsed": {"answer": 42}}
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = agent.invoke("what is the answer?")

    assert result == {
        "messages": [HumanMessage(content="what is the answer?"), ai_message],
        "output": "42",
        "structured_response": {"answer": 42},
    }


def test_invoke_with_structured_output_does_not_call_plain_model() -> None:
    model = _mock_model()
    structured = MagicMock()
    structured.invoke.return_value = {"raw": AIMessage(content="42"), "parsed": {}}
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    agent.invoke("hi")

    model.invoke.assert_not_called()
    structured.invoke.assert_called_once()


# --- ainvoke ---


def test_ainvoke_returns_messages_and_output() -> None:
    model = _mock_model(response=AIMessage(content="hello there"))
    agent = AgentChatModel(model=model)

    result = asyncio.run(agent.ainvoke("hi"))

    assert result == {
        "messages": [HumanMessage(content="hi"), AIMessage(content="hello there")],
        "output": "hello there",
    }


def test_ainvoke_calls_model_with_coerced_messages() -> None:
    model = _mock_model()
    captured: list = []

    async def _capture(
        messages: list[BaseMessage],
        **kwargs: Any,  # noqa: ARG001
    ) -> AIMessage:
        captured.append(list(messages))
        return AIMessage(content="hello")

    model.ainvoke = AsyncMock(side_effect=_capture)
    agent = AgentChatModel(model=model)

    asyncio.run(agent.ainvoke("hi"))

    assert captured[0] == [HumanMessage(content="hi")]


def test_ainvoke_forwards_config_and_kwargs() -> None:
    model = _mock_model()
    captured_kwargs: list = []

    async def _capture(
        messages: list[BaseMessage],  # noqa: ARG001
        **kwargs: Any,
    ) -> AIMessage:
        captured_kwargs.append(kwargs)
        return AIMessage(content="hello")

    model.ainvoke = AsyncMock(side_effect=_capture)
    agent = AgentChatModel(model=model)
    config = {"tags": ["test"]}

    asyncio.run(agent.ainvoke("hi", config=config, temperature=0.5))

    assert captured_kwargs[0]["config"] == config
    assert captured_kwargs[0]["temperature"] == 0.5


def test_ainvoke_with_structured_output_includes_structured_response() -> None:
    model = _mock_model()
    ai_message = AIMessage(content="42")
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value={"raw": ai_message, "parsed": {"answer": 42}})
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = asyncio.run(agent.ainvoke("what is the answer?"))

    assert result == {
        "messages": [HumanMessage(content="what is the answer?"), ai_message],
        "output": "42",
        "structured_response": {"answer": 42},
    }


def test_ainvoke_and_invoke_agree() -> None:
    model = _mock_model(response=AIMessage(content="hello"))
    agent = AgentChatModel(model=model)

    invoke_result = agent.invoke("hi")
    ainvoke_result = asyncio.run(agent.ainvoke("hi"))

    assert invoke_result == ainvoke_result


# --- batch ---


def test_batch_returns_one_result_per_input() -> None:
    model = _mock_model()
    model.batch.return_value = [AIMessage(content="a"), AIMessage(content="b")]
    agent = AgentChatModel(model=model)

    result = agent.batch(["hi", "there"])

    assert result == [
        {"messages": [HumanMessage(content="hi"), AIMessage(content="a")], "output": "a"},
        {"messages": [HumanMessage(content="there"), AIMessage(content="b")], "output": "b"},
    ]


def test_batch_calls_model_batch_with_coerced_messages() -> None:
    model = _mock_model()
    captured: list = []

    def _capture(
        all_messages: list[list[BaseMessage]],
        **kwargs: Any,  # noqa: ARG001
    ) -> list[AIMessage]:
        captured.append([list(m) for m in all_messages])
        return [AIMessage(content="a"), AIMessage(content="b")]

    model.batch.side_effect = _capture
    agent = AgentChatModel(model=model)

    agent.batch(["hi", "there"])

    assert captured[0] == [[HumanMessage(content="hi")], [HumanMessage(content="there")]]


def test_batch_empty_inputs() -> None:
    model = _mock_model()
    model.batch.return_value = []
    agent = AgentChatModel(model=model)

    result = agent.batch([])

    assert result == []


def test_batch_with_structured_output_includes_structured_response() -> None:
    model = _mock_model()
    structured = MagicMock()
    structured.batch.return_value = [
        {"raw": AIMessage(content="a"), "parsed": {"v": 1}},
        {"raw": AIMessage(content="b"), "parsed": {"v": 2}},
    ]
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = agent.batch(["hi", "there"])

    assert result == [
        {
            "messages": [HumanMessage(content="hi"), AIMessage(content="a")],
            "output": "a",
            "structured_response": {"v": 1},
        },
        {
            "messages": [HumanMessage(content="there"), AIMessage(content="b")],
            "output": "b",
            "structured_response": {"v": 2},
        },
    ]


def test_invoke_and_batch_agree() -> None:
    model = _mock_model(response=AIMessage(content="hello"))
    model.batch.return_value = [AIMessage(content="hello")]
    agent = AgentChatModel(model=model)

    invoke_result = agent.invoke("hi")
    batch_result = agent.batch(["hi"])

    assert [invoke_result] == batch_result


# --- abatch ---


def test_abatch_returns_one_result_per_input() -> None:
    model = _mock_model()
    model.abatch = AsyncMock(return_value=[AIMessage(content="a"), AIMessage(content="b")])
    agent = AgentChatModel(model=model)

    result = asyncio.run(agent.abatch(["hi", "there"]))

    assert result == [
        {"messages": [HumanMessage(content="hi"), AIMessage(content="a")], "output": "a"},
        {"messages": [HumanMessage(content="there"), AIMessage(content="b")], "output": "b"},
    ]


def test_abatch_calls_model_abatch_with_coerced_messages() -> None:
    model = _mock_model()
    captured: list = []

    async def _capture(
        all_messages: list[list[BaseMessage]],
        **kwargs: Any,  # noqa: ARG001
    ) -> list[AIMessage]:
        captured.append([list(m) for m in all_messages])
        return [AIMessage(content="a"), AIMessage(content="b")]

    model.abatch = AsyncMock(side_effect=_capture)
    agent = AgentChatModel(model=model)

    asyncio.run(agent.abatch(["hi", "there"]))

    assert captured[0] == [[HumanMessage(content="hi")], [HumanMessage(content="there")]]


def test_abatch_with_structured_output_includes_structured_response() -> None:
    model = _mock_model()
    structured = MagicMock()
    structured.abatch = AsyncMock(
        return_value=[{"raw": AIMessage(content="a"), "parsed": {"v": 1}}]
    )
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = asyncio.run(agent.abatch(["hi"]))

    assert result == [
        {
            "messages": [HumanMessage(content="hi"), AIMessage(content="a")],
            "output": "a",
            "structured_response": {"v": 1},
        }
    ]


def test_abatch_empty_inputs() -> None:
    model = _mock_model()
    model.abatch = AsyncMock(return_value=[])
    agent = AgentChatModel(model=model)

    result = asyncio.run(agent.abatch([]))

    assert result == []


def test_batch_and_abatch_agree() -> None:
    model = _mock_model()
    model.batch.return_value = [AIMessage(content="hello")]
    model.abatch = AsyncMock(return_value=[AIMessage(content="hello")])
    agent = AgentChatModel(model=model)

    batch_result = agent.batch(["hi"])
    abatch_result = asyncio.run(agent.abatch(["hi"]))

    assert batch_result == abatch_result


# --- stream ---


def test_stream_yields_model_chunks() -> None:
    model = _mock_model()
    chunks = [AIMessage(content="he"), AIMessage(content="llo")]
    model.stream.return_value = iter(chunks)
    agent = AgentChatModel(model=model)

    result = list(agent.stream("hi"))

    assert result == chunks


def test_stream_calls_model_with_coerced_messages() -> None:
    model = _mock_model()
    model.stream.return_value = iter([])
    agent = AgentChatModel(model=model)

    list(agent.stream("hi"))

    args, _kwargs = model.stream.call_args
    assert args[0] == [HumanMessage(content="hi")]


def test_stream_prepends_system_prompt() -> None:
    model = _mock_model()
    model.stream.return_value = iter([])
    agent = AgentChatModel(model=model, system_prompt="Be nice.")

    list(agent.stream("hi"))

    args, _kwargs = model.stream.call_args
    assert args[0] == [SystemMessage(content="Be nice."), HumanMessage(content="hi")]


def test_stream_ignores_response_format() -> None:
    model = _mock_model()
    model.stream.return_value = iter([AIMessage(content="hello")])
    structured = MagicMock()
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = list(agent.stream("hi"))

    assert result == [AIMessage(content="hello")]
    structured.stream.assert_not_called()
    model.stream.assert_called_once()


# --- astream ---


async def _collect_astream(agent: AgentChatModel, input_: str) -> list[AIMessage]:
    return [chunk async for chunk in agent.astream(input_)]


def test_astream_yields_model_chunks() -> None:
    model = _mock_model()
    chunks = [AIMessage(content="he"), AIMessage(content="llo")]

    async def _fake_astream(
        *args: object,  # noqa: ARG001
        **kwargs: object,  # noqa: ARG001
    ) -> AsyncGenerator[object]:
        for chunk in chunks:
            yield chunk

    model.astream = _fake_astream
    agent = AgentChatModel(model=model)

    result = asyncio.run(_collect_astream(agent, "hi"))

    assert result == chunks


def test_astream_ignores_response_format() -> None:
    model = _mock_model()

    async def _fake_astream(
        *args: object,  # noqa: ARG001
        **kwargs: object,  # noqa: ARG001
    ) -> AsyncGenerator[object]:
        yield AIMessage(content="hello")

    model.astream = _fake_astream
    structured = MagicMock()
    model.with_structured_output.return_value = structured

    agent = AgentChatModel(model=model, response_format=dict)
    result = asyncio.run(_collect_astream(agent, "hi"))

    assert result == [AIMessage(content="hello")]
    structured.astream.assert_not_called()


def test_stream_and_astream_agree() -> None:
    model = _mock_model()
    chunks = [AIMessage(content="he"), AIMessage(content="llo")]
    model.stream.return_value = iter(chunks)

    async def _fake_astream(
        *args: object,  # noqa: ARG001
        **kwargs: object,  # noqa: ARG001
    ) -> AsyncGenerator[object]:
        for chunk in chunks:
            yield chunk

    model.astream = _fake_astream
    agent = AgentChatModel(model=model)

    stream_result = list(agent.stream("hi"))
    astream_result = asyncio.run(_collect_astream(agent, "hi"))

    assert stream_result == astream_result
