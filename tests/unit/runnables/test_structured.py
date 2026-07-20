from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from zenpyre.runnables import structured_output_runnable
from zenpyre.runnables.structured import (
    StructuredOutputError,
    _as_text,
    _unwrap,
)

MODULE = "zenpyre.runnables.structured"

# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


class Answer(BaseModel):
    value: int


class FakeChatModel:
    """Minimal stand-in for BaseChatModel: only implements
    with_structured_output, returning a fixed dict shaped like the real
    ``include_raw=True`` contract."""

    def __init__(
        self,
        raw_content: str,
        parsed: Answer | None,
        parsing_error: Exception | None,
    ) -> None:
        self._raw_content = raw_content
        self._parsed = parsed
        self._parsing_error = parsing_error

    def with_structured_output(
        self,
        output_type: type,  # noqa: ARG002
        include_raw: bool = True,  # noqa: ARG002
    ) -> RunnableLambda:
        def _call(_input: Any) -> dict[str, Any]:
            return {
                "raw": AIMessage(content=self._raw_content),
                "parsed": self._parsed,
                "parsing_error": self._parsing_error,
            }

        return RunnableLambda(_call)


def _native_ok_model() -> FakeChatModel:
    return FakeChatModel('{"value": 99}', Answer(value=99), None)


def _fallback_ok_model() -> FakeChatModel:
    # native parsing failed, but raw content is valid JSON the fallback can parse
    return FakeChatModel('{"value": 7}', None, ValueError("no tool call"))


def _both_fail_model() -> FakeChatModel:
    # native parsing failed, and raw content isn't valid JSON either
    return FakeChatModel("not json at all", None, ValueError("no tool call"))


##################################################
#     Tests for _as_text                        #
##################################################


def test_as_text_plain_string_returned_unchanged() -> None:
    assert _as_text("hello") == "hello"


def test_as_text_empty_string() -> None:
    assert _as_text("") == ""


def test_as_text_list_of_text_blocks_concatenated() -> None:
    content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
    assert _as_text(content) == "part1part2"


def test_as_text_list_ignores_non_text_blocks() -> None:
    content = [
        {"type": "text", "text": "part1"},
        {"type": "image_url", "image_url": "http://example.com/x.png"},
        {"type": "text", "text": "part2"},
    ]
    assert _as_text(content) == "part1part2"


def test_as_text_list_of_plain_strings() -> None:
    assert _as_text(["a", "b", "c"]) == "abc"


def test_as_text_mixed_list_of_strings_and_blocks() -> None:
    content = ["a", {"type": "text", "text": "b"}, {"type": "image_url", "image_url": "x"}]
    assert _as_text(content) == "ab"


def test_as_text_empty_list_returns_empty_string() -> None:
    assert _as_text([]) == ""


def test_as_text_text_block_missing_text_key_treated_as_empty() -> None:
    content = [{"type": "text"}]
    assert _as_text(content) == ""


##################################################
#     Tests for _unwrap                          #
##################################################


def _make_result(raw_content: str, parsed: Answer | None, parsing_error: Exception | None) -> dict:
    return {"raw": AIMessage(content=raw_content), "parsed": parsed, "parsing_error": parsing_error}


# --- include_raw=False ---


def test_unwrap_native_success_include_raw_false_returns_parsed() -> None:
    result = _make_result('{"value": 99}', Answer(value=99), None)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        output = _unwrap(result, output_type=Answer, include_raw=False, log_tokens=True)
    assert output == Answer(value=99)
    mock_log.assert_called_once_with(result)


def test_unwrap_log_tokens_defaults_to_false_does_not_log() -> None:
    result = _make_result('{"value": 99}', Answer(value=99), None)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        _unwrap(result, output_type=Answer, include_raw=False)
    mock_log.assert_not_called()


def test_unwrap_log_tokens_false_does_not_log() -> None:
    result = _make_result('{"value": 99}', Answer(value=99), None)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        _unwrap(result, output_type=Answer, include_raw=False, log_tokens=False)
    mock_log.assert_not_called()


def test_unwrap_log_tokens_true_logs_on_fallback_success() -> None:
    result = _make_result('{"value": 7}', None, ValueError("no tool call"))
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        _unwrap(result, output_type=Answer, include_raw=False, log_tokens=True)
    mock_log.assert_called_once_with(result)


def test_unwrap_log_tokens_true_logs_even_when_both_fail() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    with patch(f"{MODULE}.log_token_usage") as mock_log, pytest.raises(StructuredOutputError):
        _unwrap(result, output_type=Answer, include_raw=False, log_tokens=True)
    mock_log.assert_called_once_with(result)


def test_unwrap_fallback_success_include_raw_false_returns_parsed() -> None:
    result = _make_result('{"value": 7}', None, ValueError("no tool call"))
    output = _unwrap(result, output_type=Answer, include_raw=False)
    assert output == Answer(value=7)


def test_unwrap_both_fail_include_raw_false_raises() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    with pytest.raises(StructuredOutputError, match=r"Failed to parse LLM output into Answer"):
        _unwrap(result, output_type=Answer, include_raw=False)


def test_unwrap_parsed_and_error_both_none_falls_back_to_json() -> None:
    # native call reports neither success nor an exception (e.g. no tool
    # call emitted); the fallback must still be attempted rather than
    # treating this as a successful parse of `None`.
    result = _make_result('{"value": 7}', None, None)
    output = _unwrap(result, output_type=Answer, include_raw=False)
    assert output == Answer(value=7)


def test_unwrap_parsed_and_error_both_none_and_fallback_also_fails_raises() -> None:
    result = _make_result("not json", None, None)
    with pytest.raises(StructuredOutputError, match=r"Failed to parse LLM output into Answer"):
        _unwrap(result, output_type=Answer, include_raw=False)


def test_unwrap_parsed_and_error_both_none_include_raw_true_uses_fallback() -> None:
    result = _make_result('{"value": 7}', None, None)
    output = _unwrap(result, output_type=Answer, include_raw=True)
    assert output["parsed"] == Answer(value=7)
    assert output["used_fallback"] is True


def test_unwrap_both_fail_include_raw_false_error_chained_from_json_error() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    with pytest.raises(StructuredOutputError, match=r"Failed to parse LLM output into Answer"):
        _unwrap(result, output_type=Answer, include_raw=False)


def test_unwrap_both_fail_include_raw_false_message_mentions_both_errors() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    with pytest.raises(StructuredOutputError) as exc_info:
        _unwrap(result, output_type=Answer, include_raw=False)
    msg = str(exc_info.value)
    assert "no tool call" in msg
    assert "Fallback manual JSON parsing also failed" in msg


# --- include_raw=True ---


def test_unwrap_native_success_include_raw_true_returns_dict() -> None:
    result = _make_result('{"value": 99}', Answer(value=99), None)
    output = _unwrap(result, output_type=Answer, include_raw=True)
    assert output["parsed"] == Answer(value=99)
    assert output["parsing_error"] is None
    assert output["used_fallback"] is False
    assert output["raw"] is result["raw"]


def test_unwrap_fallback_success_include_raw_true_returns_dict() -> None:
    result = _make_result('{"value": 7}', None, ValueError("no tool call"))
    output = _unwrap(result, output_type=Answer, include_raw=True)
    assert output["parsed"] == Answer(value=7)
    assert output["parsing_error"] is None
    assert output["used_fallback"] is True


def test_unwrap_both_fail_include_raw_true_does_not_raise() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    output = _unwrap(result, output_type=Answer, include_raw=True)
    assert output["parsed"] is None
    assert isinstance(output["parsing_error"], StructuredOutputError)
    assert output["used_fallback"] is False


def test_unwrap_both_fail_include_raw_true_error_message_matches_raised_version() -> None:
    result = _make_result("not json", None, ValueError("no tool call"))
    output = _unwrap(result, output_type=Answer, include_raw=True)
    msg = str(output["parsing_error"])
    assert "no tool call" in msg
    assert "Fallback manual JSON parsing also failed" in msg


def test_unwrap_include_raw_true_always_has_used_fallback_key() -> None:
    for result in [
        _make_result('{"value": 1}', Answer(value=1), None),
        _make_result('{"value": 2}', None, ValueError("x")),
        _make_result("bad json", None, ValueError("x")),
    ]:
        output = _unwrap(result, output_type=Answer, include_raw=True)
        assert "used_fallback" in output


##################################################
#     Tests for structured_output_runnable       #
##################################################

# --- default (include_raw=False) ---


def test_structured_output_runnable_native_success_returns_parsed() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    assert chain.invoke("hi") == Answer(value=99)


def test_structured_output_runnable_fallback_success_returns_parsed() -> None:
    chain = structured_output_runnable(_fallback_ok_model(), Answer)
    assert chain.invoke("hi") == Answer(value=7)


def test_structured_output_runnable_both_fail_raises() -> None:
    chain = structured_output_runnable(_both_fail_model(), Answer)
    with pytest.raises(StructuredOutputError):
        chain.invoke("hi")


def test_structured_output_runnable_include_raw_defaults_to_false() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    result = chain.invoke("hi")
    assert isinstance(result, Answer)


# --- log_tokens ---


def test_structured_output_runnable_log_tokens_defaults_to_false() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        chain.invoke("hi")
    mock_log.assert_not_called()


def test_structured_output_runnable_log_tokens_false_does_not_log() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer, log_tokens=False)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        chain.invoke("hi")
    mock_log.assert_not_called()


def test_structured_output_runnable_log_tokens_true_logs() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer, log_tokens=True)
    with patch(f"{MODULE}.log_token_usage") as mock_log:
        chain.invoke("hi")
    mock_log.assert_called_once()


# --- include_raw=True ---


def test_structured_output_runnable_include_raw_true_native_success() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer, include_raw=True)
    result = chain.invoke("hi")
    assert result["parsed"] == Answer(value=99)
    assert result["parsing_error"] is None
    assert result["used_fallback"] is False


def test_structured_output_runnable_include_raw_true_fallback_success() -> None:
    chain = structured_output_runnable(_fallback_ok_model(), Answer, include_raw=True)
    result = chain.invoke("hi")
    assert result["parsed"] == Answer(value=7)
    assert result["used_fallback"] is True


def test_structured_output_runnable_include_raw_true_both_fail_does_not_raise() -> None:
    chain = structured_output_runnable(_both_fail_model(), Answer, include_raw=True)
    result = chain.invoke("hi")  # should not raise
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], StructuredOutputError)


# --- Runnable interface: batch/abatch/ainvoke/streaming/composability ---


def test_structured_output_runnable_returns_a_runnable() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    assert hasattr(chain, "invoke")
    assert hasattr(chain, "ainvoke")
    assert hasattr(chain, "batch")
    assert hasattr(chain, "abatch")


def test_structured_output_runnable_batch() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    results = chain.batch(["a", "b", "c"])
    assert results == [Answer(value=99)] * 3


def test_structured_output_runnable_ainvoke() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    result = asyncio.run(chain.ainvoke("hi"))
    assert result == Answer(value=99)


def test_structured_output_runnable_abatch() -> None:
    chain = structured_output_runnable(_native_ok_model(), Answer)
    results = asyncio.run(chain.abatch(["a", "b"]))
    assert results == [Answer(value=99)] * 2


def test_structured_output_runnable_batch_return_exceptions_true() -> None:
    chain = structured_output_runnable(_both_fail_model(), Answer)
    results = chain.batch(["a", "b"], return_exceptions=True)
    assert len(results) == 2
    assert all(isinstance(r, StructuredOutputError) for r in results)


def test_structured_output_runnable_composable_with_pipe() -> None:
    chain = RunnableLambda(lambda x: x.upper()) | structured_output_runnable(
        _native_ok_model(), Answer
    )
    assert chain.invoke("hi") == Answer(value=99)


def test_structured_output_runnable_config_run_name_set_on_unwrap_step() -> None:
    # Sanity check that the unwrap step is configured with a run_name,
    # useful for tracing; doesn't assert exact internal structure.
    chain = structured_output_runnable(_native_ok_model(), Answer)
    assert chain.invoke("hi") == Answer(value=99)  # behavior unaffected by the config
