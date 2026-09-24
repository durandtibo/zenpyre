from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from zenpyre.chat_models import ainvoke_structured_llm, invoke_structured_llm

MODULE = "zenpyre.chat_models.structured"

SYSTEM_PROMPT = "You are a helpful assistant."
USER_CONTENT = "What is the capital of France?"
TIMEBLOCK_MESSAGE = "LLM generated answer in {time}"


class _Answer(BaseModel):
    value: str


def _raw_response(parsed: _Answer | None = None) -> dict[str, object]:
    return {"raw": MagicMock(), "parsed": parsed, "parsing_error": None}


def _failed_response(error: Exception | None = None) -> dict[str, object]:
    return {"raw": MagicMock(), "parsed": None, "parsing_error": error or ValueError("bad output")}


###########################################
#     Tests for invoke_structured_llm     #
###########################################


def test_invoke_returns_parsed_and_raw_response() -> None:
    parsed = _Answer(value="Paris")
    raw_response = _raw_response(parsed)
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = raw_response

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        result_parsed, result_raw = invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    assert result_parsed is parsed
    assert result_raw is raw_response


def test_invoke_builds_runnable_with_include_raw() -> None:
    chat_model = MagicMock()
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _raw_response()

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm) as mock_builder:
        invoke_structured_llm(
            chat_model=chat_model,
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    mock_builder.assert_called_once_with(chat_model, output_type=_Answer, include_raw=True)


def test_invoke_sends_system_and_human_messages() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _raw_response()

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    (messages,), _ = structured_llm.invoke.call_args
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == USER_CONTENT


def test_invoke_uses_default_timeblock_message() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _raw_response()

    with (
        patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm),
        patch(f"{MODULE}.timeblock") as mock_timeblock,
    ):
        invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
        )

    mock_timeblock.assert_called_once_with("LLM generated answer in {time}")


def test_invoke_passes_custom_timeblock_message() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _raw_response()

    with (
        patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm),
        patch(f"{MODULE}.timeblock") as mock_timeblock,
    ):
        invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    mock_timeblock.assert_called_once_with(TIMEBLOCK_MESSAGE)


def test_invoke_returns_none_when_parsing_fails() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _raw_response(parsed=None)

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        parsed, _ = invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    assert parsed is None


def test_invoke_max_retries_defaults_to_zero_makes_single_call() -> None:
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _failed_response()

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
        )

    assert structured_llm.invoke.call_count == 1


def test_invoke_max_retries_stops_early_on_success() -> None:
    parsed = _Answer(value="Paris")
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = [_failed_response(), _raw_response(parsed)]

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        result_parsed, result_raw = invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=3,
        )

    assert result_parsed is parsed
    assert result_raw["parsing_error"] is None
    assert structured_llm.invoke.call_count == 2


def test_invoke_max_retries_exhausted_returns_last_failed_response() -> None:
    structured_llm = MagicMock()
    responses = [_failed_response(ValueError(f"error-{i}")) for i in range(3)]
    structured_llm.invoke.side_effect = responses

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        parsed, raw_response = invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=2,
        )

    assert parsed is None
    assert raw_response is responses[-1]
    assert structured_llm.invoke.call_count == 3


def test_invoke_max_retries_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"max_retries must be non-negative"):
        invoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=-1,
        )


############################################
#     Tests for ainvoke_structured_llm     #
############################################


async def test_ainvoke_returns_parsed_and_raw_response() -> None:
    parsed = _Answer(value="Paris")
    raw_response = _raw_response(parsed)
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return raw_response

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        result_parsed, result_raw = await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    assert result_parsed is parsed
    assert result_raw is raw_response


async def test_ainvoke_builds_runnable_with_include_raw() -> None:
    chat_model = MagicMock()
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return _raw_response()

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm) as mock_builder:
        await ainvoke_structured_llm(
            chat_model=chat_model,
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    mock_builder.assert_called_once_with(chat_model, output_type=_Answer, include_raw=True)


async def test_ainvoke_uses_default_timeblock_message() -> None:
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return _raw_response()

    structured_llm.ainvoke = _ainvoke

    with (
        patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm),
        patch(f"{MODULE}.timeblock") as mock_timeblock,
    ):
        await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
        )

    mock_timeblock.assert_called_once_with("LLM generated answer in {time}")


async def test_ainvoke_passes_custom_timeblock_message() -> None:
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return _raw_response()

    structured_llm.ainvoke = _ainvoke

    with (
        patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm),
        patch(f"{MODULE}.timeblock") as mock_timeblock,
    ):
        await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    mock_timeblock.assert_called_once_with(TIMEBLOCK_MESSAGE)


async def test_ainvoke_sends_system_and_human_messages() -> None:
    captured: dict[str, object] = {}
    structured_llm = MagicMock()

    async def _ainvoke(messages: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        captured["messages"] = messages
        return _raw_response()

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    messages = captured["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == SYSTEM_PROMPT
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == USER_CONTENT


async def test_ainvoke_returns_none_when_parsing_fails() -> None:
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return _raw_response(parsed=None)

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        parsed, _ = await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            timeblock_message=TIMEBLOCK_MESSAGE,
        )

    assert parsed is None


async def test_ainvoke_max_retries_defaults_to_zero_makes_single_call() -> None:
    calls = 0
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _failed_response()

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
        )

    assert calls == 1


async def test_ainvoke_max_retries_stops_early_on_success() -> None:
    parsed = _Answer(value="Paris")
    responses = [_failed_response(), _raw_response(parsed)]
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return responses.pop(0)

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        result_parsed, result_raw = await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=3,
        )

    assert result_parsed is parsed
    assert result_raw["parsing_error"] is None
    assert responses == []


async def test_ainvoke_max_retries_exhausted_returns_last_failed_response() -> None:
    responses = [_failed_response(ValueError(f"error-{i}")) for i in range(3)]
    last_response = responses[-1]
    structured_llm = MagicMock()

    async def _ainvoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        return responses.pop(0)

    structured_llm.ainvoke = _ainvoke

    with patch(f"{MODULE}.structured_output_runnable", return_value=structured_llm):
        parsed, raw_response = await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=2,
        )

    assert parsed is None
    assert raw_response is last_response
    assert responses == []


async def test_ainvoke_max_retries_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"max_retries must be non-negative"):
        await ainvoke_structured_llm(
            chat_model=MagicMock(),
            output_type=_Answer,
            system_prompt=SYSTEM_PROMPT,
            user_content=USER_CONTENT,
            max_retries=-1,
        )
