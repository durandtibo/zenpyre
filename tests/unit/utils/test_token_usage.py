"""Unit tests for get_invoke_token_usage."""

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    UsageMetadata,
)

from zenpyre.utils.token_usage import (
    format_token_usage,
    get_batch_token_usage,
    get_invoke_token_usage,
    get_token_usage,
    log_token_usage,
)

MODULE = "zenpyre.utils.token_usage"


######################################################
#     Tests for format_token_usage                   #
######################################################


def test_format_token_usage_returns_str() -> None:
    usage = UsageMetadata(input_tokens=1234, output_tokens=567, total_tokens=1801)
    result = format_token_usage(usage)
    assert isinstance(result, str)


def test_format_token_usage_docstring_example() -> None:
    usage = UsageMetadata(input_tokens=1234, output_tokens=567, total_tokens=1801)
    assert format_token_usage(usage) == (
        "Token usage\n  Input tokens:  1,234\n  Output tokens:   567\n  Total tokens:  1,801"
    )


def test_format_token_usage_zero_values() -> None:
    usage = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    assert format_token_usage(usage) == (
        "Token usage\n  Input tokens:  0\n  Output tokens: 0\n  Total tokens:  0"
    )


def test_format_token_usage_missing_keys_default_to_zero() -> None:
    assert format_token_usage({}) == (
        "Token usage\n  Input tokens:  0\n  Output tokens: 0\n  Total tokens:  0"
    )


def test_format_token_usage_adds_thousands_separator() -> None:
    usage = UsageMetadata(input_tokens=1000000, output_tokens=1, total_tokens=1000001)
    assert format_token_usage(usage) == (
        "Token usage\n"
        "  Input tokens:  1,000,000\n"
        "  Output tokens:         1\n"
        "  Total tokens:  1,000,001"
    )


def test_format_token_usage_single_digit_values() -> None:
    usage = UsageMetadata(input_tokens=1, output_tokens=22, total_tokens=333)
    assert format_token_usage(usage) == (
        "Token usage\n  Input tokens:    1\n  Output tokens:  22\n  Total tokens:  333"
    )


def test_format_token_usage_output_wider_than_input_and_total() -> None:
    usage = UsageMetadata(input_tokens=5, output_tokens=123456, total_tokens=99)
    assert format_token_usage(usage) == (
        "Token usage\n  Input tokens:        5\n  Output tokens: 123,456\n  Total tokens:       99"
    )


######################################################
#     Tests for get_invoke_token_usage               #
######################################################


def test_get_invoke_token_usage_returns_usage_metadata() -> None:
    result = get_invoke_token_usage({"messages": []})
    assert isinstance(result, dict)


def test_get_invoke_token_usage_empty_messages_returns_zeros() -> None:
    result = get_invoke_token_usage({"messages": []})
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_missing_messages_key_returns_zeros() -> None:
    result = get_invoke_token_usage({})
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_single_ai_message() -> None:
    messages = [
        AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )
    ]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_invoke_token_usage_sums_multiple_ai_messages() -> None:
    messages = [
        AIMessage(
            content="a",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        AIMessage(
            content="b",
            usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
        ),
    ]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=30, output_tokens=13, total_tokens=43)


def test_get_invoke_token_usage_skips_non_ai_messages() -> None:
    messages = [
        HumanMessage(content="hello"),
        SystemMessage(content="you are a bot"),
        AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    ]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_invoke_token_usage_skips_ai_message_without_usage_metadata() -> None:
    messages = [
        AIMessage(content="no usage here"),
        AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
    ]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_invoke_token_usage_all_ai_messages_without_usage_metadata() -> None:
    messages = [AIMessage(content="a"), AIMessage(content="b")]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_missing_individual_usage_fields() -> None:
    messages = [Mock(spec=AIMessage, content="a", usage_metadata={"input_tokens": 7})]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=7, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_total_tokens_independent_of_input_output() -> None:
    # total_tokens can exceed input_tokens + output_tokens when a provider
    # reports extra token categories (e.g. reasoning tokens).
    messages = [
        AIMessage(
            content="a",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=20),
        )
    ]
    result = get_invoke_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=20)


def test_get_invoke_token_usage_single_base_message_with_usage() -> None:
    message = AIMessage(
        content="hi",
        usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    result = get_invoke_token_usage(message)
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_invoke_token_usage_single_ai_message_without_usage_metadata() -> None:
    message = AIMessage(content="no usage here")
    result = get_invoke_token_usage(message)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_single_non_ai_message_returns_zeros() -> None:
    message = HumanMessage(content="hello")
    result = get_invoke_token_usage(message)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_invoke_token_usage_single_system_message_returns_zeros() -> None:
    message = SystemMessage(content="you are a bot")
    result = get_invoke_token_usage(message)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


######################################################
#     Tests for get_batch_token_usage                #
######################################################


def test_get_batch_token_usage_returns_usage_metadata() -> None:
    result = get_batch_token_usage([])
    assert isinstance(result, dict)


def test_get_batch_token_usage_empty_results_returns_zeros() -> None:
    result = get_batch_token_usage([])
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_batch_token_usage_single_result() -> None:
    results = [
        {
            "messages": [
                AIMessage(
                    content="hi",
                    usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ]
        }
    ]
    result = get_batch_token_usage(results)
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_batch_token_usage_sums_multiple_results() -> None:
    results = [
        {
            "messages": [
                AIMessage(
                    content="a",
                    usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ]
        },
        {
            "messages": [
                AIMessage(
                    content="b",
                    usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
                )
            ]
        },
    ]
    result = get_batch_token_usage(results)
    assert result == UsageMetadata(input_tokens=30, output_tokens=13, total_tokens=43)


def test_get_batch_token_usage_result_with_no_usage_contributes_zero() -> None:
    results = [
        {"messages": [AIMessage(content="no usage")]},
        {
            "messages": [
                AIMessage(
                    content="b",
                    usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
                )
            ]
        },
    ]
    result = get_batch_token_usage(results)
    assert result == UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28)


def test_get_batch_token_usage_calls_get_invoke_token_usage_once_per_result() -> None:
    results = [{"messages": []}, {"messages": []}, {"messages": []}]
    with patch(
        f"{MODULE}.get_invoke_token_usage",
        return_value=UsageMetadata(input_tokens=1, output_tokens=1, total_tokens=2),
    ) as mock_get_invoke:
        result = get_batch_token_usage(results)
    assert mock_get_invoke.call_count == 3
    assert result == UsageMetadata(input_tokens=3, output_tokens=3, total_tokens=6)


def test_get_batch_token_usage_passes_each_result_to_get_invoke_token_usage() -> None:
    results = [{"messages": ["r1"]}, {"messages": ["r2"]}]
    with patch(
        f"{MODULE}.get_invoke_token_usage",
        return_value=UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0),
    ) as mock_get_invoke:
        get_batch_token_usage(results)
    mock_get_invoke.assert_any_call(results[0])
    mock_get_invoke.assert_any_call(results[1])


######################################################
#     Tests for get_token_usage                     #
######################################################

# --- dict input (single invoke result) ---


def test_get_token_usage_dict_returns_usage_metadata() -> None:
    result = get_token_usage({"messages": []})
    assert isinstance(result, dict)


def test_get_token_usage_dict_empty_messages_returns_zeros() -> None:
    result = get_token_usage({"messages": []})
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_token_usage_dict_sums_ai_messages() -> None:
    messages = [
        AIMessage(
            content="hi",
            usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
        )
    ]
    result = get_token_usage({"messages": messages})
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_token_usage_dict_delegates_to_get_invoke_token_usage() -> None:
    with patch(f"{MODULE}.get_invoke_token_usage") as mock_invoke:
        mock_invoke.return_value = UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)
        result = get_token_usage({"messages": []})
    mock_invoke.assert_called_once_with({"messages": []})
    assert result == UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)


# --- BaseMessage input ---


def test_get_token_usage_base_message_returns_usage_metadata() -> None:
    message = AIMessage(content="hi")
    result = get_token_usage(message)
    assert isinstance(result, dict)


def test_get_token_usage_base_message_with_usage() -> None:
    message = AIMessage(
        content="hi",
        usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    result = get_token_usage(message)
    assert result == UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)


def test_get_token_usage_base_message_without_usage_returns_zeros() -> None:
    message = AIMessage(content="no usage")
    result = get_token_usage(message)
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_token_usage_base_message_delegates_to_get_invoke_token_usage() -> None:
    message = AIMessage(content="hi")
    with patch(f"{MODULE}.get_invoke_token_usage") as mock_invoke:
        mock_invoke.return_value = UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)
        result = get_token_usage(message)
    mock_invoke.assert_called_once_with(message)
    assert result == UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)


# --- list input (batch result) ---


def test_get_token_usage_list_returns_usage_metadata() -> None:
    result = get_token_usage([])
    assert isinstance(result, dict)


def test_get_token_usage_empty_list_returns_zeros() -> None:
    result = get_token_usage([])
    assert result == UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)


def test_get_token_usage_list_sums_across_batch() -> None:
    batch = [
        {
            "messages": [
                AIMessage(
                    content="a",
                    usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ]
        },
        {
            "messages": [
                AIMessage(
                    content="b",
                    usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
                )
            ]
        },
    ]
    result = get_token_usage(batch)
    assert result == UsageMetadata(input_tokens=30, output_tokens=13, total_tokens=43)


def test_get_token_usage_list_delegates_to_get_batch_token_usage() -> None:
    batch = [{"messages": []}]
    with patch(f"{MODULE}.get_batch_token_usage") as mock_batch:
        mock_batch.return_value = UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)
        result = get_token_usage(batch)
    mock_batch.assert_called_once_with(batch)
    assert result == UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)


# --- invalid input ---


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(42, id="int"),
        pytest.param("not a valid result", id="str"),
        pytest.param(None, id="none"),
        pytest.param(3.14, id="float"),
        pytest.param({1, 2, 3}, id="set"),
    ],
)
def test_get_token_usage_invalid_type_raises_type_error(result: object) -> None:
    with pytest.raises(TypeError, match=r"Expected a BaseMessage, a dict"):
        get_token_usage(result)


def test_get_token_usage_invalid_type_error_message_includes_type_name() -> None:
    with pytest.raises(TypeError, match=r"int"):
        get_token_usage(42)


######################################################
#     Tests for log_token_usage                      #
######################################################


def test_log_token_usage_single_invoke_result(caplog: pytest.LogCaptureFixture) -> None:
    result = {
        "messages": [
            AIMessage(
                content="hi",
                usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        ]
    }
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage(result)
    expected_usage = UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15)
    assert caplog.messages == [format_token_usage(expected_usage)]


def test_log_token_usage_batch_result(caplog: pytest.LogCaptureFixture) -> None:
    results = [
        {
            "messages": [
                AIMessage(
                    content="a",
                    usage_metadata=UsageMetadata(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            ]
        },
        {
            "messages": [
                AIMessage(
                    content="b",
                    usage_metadata=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
                )
            ]
        },
    ]
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage(results)
    expected_usage = UsageMetadata(input_tokens=30, output_tokens=13, total_tokens=43)
    assert caplog.messages == [format_token_usage(expected_usage)]


def test_log_token_usage_empty_dict_logs_zeros(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage({})
    expected_usage = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    assert caplog.messages == [format_token_usage(expected_usage)]


def test_log_token_usage_empty_list_logs_zeros(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage([])
    expected_usage = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
    assert caplog.messages == [format_token_usage(expected_usage)]


def test_log_token_usage_logs_at_info_level(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage({})
    assert caplog.records[0].levelno == logging.INFO


def test_log_token_usage_logs_exactly_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=MODULE):
        log_token_usage({})
    assert len(caplog.records) == 1


def test_log_token_usage_invalid_type_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    log_token_usage(123)
    assert caplog.records == []
