from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from zenpyre.utils.json_to_structured import (
    JsonStructuredOutputParseError,
    _extract_json_object,
    _fix_common_json_issues,
    _strip_code_fences,
    parse_json_to_structured,
    parse_json_to_structured_with_retry,
)

MODULE = "zenpyre.utils.json_to_structured"


# ---------------------------------------------------------------------------
# Minimal schema fixtures
# ---------------------------------------------------------------------------


class ProductReview(BaseModel):
    summary: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)


class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str


class MovieReview(BaseModel):
    """Structured extraction target for a movie review."""

    title: str = Field(description="Title of the movie being reviewed.")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed.")
    score: int = Field(description="Reviewer's rating out of 10.")


###########################################
#     Tests for _strip_code_fences        #
###########################################


def test_strip_code_fences_removes_json_fence() -> None:
    content = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(content) == '{"a": 1}'


def test_strip_code_fences_removes_bare_fence() -> None:
    content = '```\n{"a": 1}\n```'
    assert _strip_code_fences(content) == '{"a": 1}'


def test_strip_code_fences_no_fence_returns_stripped_text() -> None:
    content = '  {"a": 1}  \n'
    assert _strip_code_fences(content) == '{"a": 1}'


def test_strip_code_fences_only_opening_fence() -> None:
    content = '```json\n{"a": 1}'
    assert _strip_code_fences(content) == '{"a": 1}'


def test_strip_code_fences_only_closing_fence() -> None:
    content = '{"a": 1}\n```'
    assert _strip_code_fences(content) == '{"a": 1}'


def test_strip_code_fences_json_marker_case_sensitivity() -> None:
    # Only lowercase "json" marker is stripped; anything else is left as-is
    # apart from the backticks themselves.
    content = '```JSON\n{"a": 1}\n```'
    result = _strip_code_fences(content)
    assert result == 'JSON\n{"a": 1}'


def test_strip_code_fences_empty_string() -> None:
    assert _strip_code_fences("") == ""


def test_strip_code_fences_preserves_internal_content_untouched() -> None:
    content = '```json\n{"a": "```nested```"}\n```'
    assert _strip_code_fences(content) == '{"a": "```nested```"}'


###########################################
#     Tests for _extract_json_object      #
###########################################


def test_extract_json_object_simple_object() -> None:
    assert _extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_with_surrounding_prose() -> None:
    text = 'prefix {"a": 1} suffix'
    assert _extract_json_object(text) == '{"a": 1}'


def test_extract_json_object_returns_first_balanced_object_directly() -> None:
    text = 'prefix {"a": {"b": 1}} suffix'
    assert _extract_json_object(text) == '{"a": {"b": 1}}'


def test_extract_json_object_stops_at_first_complete_object() -> None:
    # Two separate top-level objects; only the first should be returned.
    text = '{"a": 1} {"b": 2}'
    assert _extract_json_object(text) == '{"a": 1}'


def test_extract_json_object_ignores_braces_inside_strings() -> None:
    text = '{"a": "value with {brace}"}'
    assert _extract_json_object(text) == '{"a": "value with {brace}"}'


def test_extract_json_object_handles_escaped_quote_inside_string() -> None:
    text = '{"a": "quote \\" here"}'
    assert _extract_json_object(text) == '{"a": "quote \\" here"}'


def test_extract_json_object_handles_escaped_backslash_inside_string() -> None:
    text = '{"a": "backslash \\\\"}'
    assert _extract_json_object(text) == '{"a": "backslash \\\\"}'


def test_extract_json_object_no_opening_brace_raises() -> None:
    with pytest.raises(ValueError, match=r"No opening brace"):
        _extract_json_object("no braces here")


def test_extract_json_object_unterminated_object_raises() -> None:
    with pytest.raises(ValueError, match=r"No matching closing brace"):
        _extract_json_object('{"a": 1')


def test_extract_json_object_empty_string_raises() -> None:
    with pytest.raises(ValueError, match=r"No opening brace"):
        _extract_json_object("")


def test_extract_json_object_nested_objects_return_outermost() -> None:
    text = '{"a": {"b": {"c": 1}}}'
    assert _extract_json_object(text) == '{"a": {"b": {"c": 1}}}'


###########################################
#     Tests for _fix_common_json_issues   #
###########################################


def test_fix_common_json_issues_removes_trailing_comma_before_closing_brace() -> None:
    assert _fix_common_json_issues('{"a": 1,}') == '{"a": 1}'


def test_fix_common_json_issues_removes_trailing_comma_before_closing_bracket() -> None:
    assert _fix_common_json_issues("[1, 2,]") == "[1, 2]"


def test_fix_common_json_issues_removes_trailing_comma_with_whitespace() -> None:
    assert _fix_common_json_issues('{"a": 1,  \n}') == '{"a": 1}'


def test_fix_common_json_issues_no_trailing_comma_unchanged() -> None:
    content = '{"a": 1, "b": 2}'
    assert _fix_common_json_issues(content) == content


def test_fix_common_json_issues_nested_trailing_commas() -> None:
    content = '{"a": [1, 2,], "b": 3,}'
    assert _fix_common_json_issues(content) == '{"a": [1, 2], "b": 3}'


def test_fix_common_json_issues_empty_string() -> None:
    assert _fix_common_json_issues("") == ""


def test_fix_common_json_issues_does_not_affect_commas_inside_strings() -> None:
    # Note: this is a known limitation of the regex-based approach — a comma
    # immediately followed by a brace/bracket *inside* a string value would
    # also be stripped. This test documents current (imperfect) behavior.
    content = '{"a": "text, more}"}'
    result = _fix_common_json_issues(content)
    assert result == content  # no trailing comma directly before } or ] here


###########################################
#     Tests for parse_json_to_structured  #
###########################################


# --- Return type and basic parsing ---


def test_parse_json_to_structured_returns_schema_instance() -> None:
    content = '{"summary": "Great product", "rating": 5}'
    assert isinstance(parse_json_to_structured(content, ProductReview), ProductReview)


def test_parse_json_to_structured_populates_fields_correctly() -> None:
    content = '{"summary": "Great product", "rating": 5}'
    result = parse_json_to_structured(content, ProductReview)
    assert result.summary == "Great product"
    assert result.rating == 5


def test_parse_json_to_structured_plain_json_no_fences() -> None:
    content = '{"city": "Vancouver", "temperature_celsius": 14.0, "condition": "rainy"}'
    result = parse_json_to_structured(content, WeatherReport)
    assert result.city == "Vancouver"


# --- Code fence handling ---


def test_parse_json_to_structured_strips_json_fence() -> None:
    content = '```json\n{"summary": "Nice", "rating": 4}\n```'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 4


def test_parse_json_to_structured_strips_bare_fence() -> None:
    content = '```\n{"summary": "Nice", "rating": 4}\n```'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 4


# --- Extraction from surrounding text ---


def test_parse_json_to_structured_extracts_json_from_surrounding_prose() -> None:
    content = 'Sure! Here is the review:\n{"summary": "Nice", "rating": 4}\nHope that helps!'
    result = parse_json_to_structured(content, ProductReview)
    assert result.summary == "Nice"


def test_parse_json_to_structured_handles_nested_braces() -> None:
    content = 'Result: {"summary": "Contains {curly} braces", "rating": 3} end of message'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 3


def test_parse_json_to_structured_ignores_braces_inside_strings() -> None:
    content = '{"summary": "A \\"quoted {brace}\\" example", "rating": 2}'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 2


def test_parse_json_to_structured_handles_escaped_backslash_in_string() -> None:
    # Surrounding prose forces fallback to _extract_json_object, which must
    # correctly skip over the escaped backslash while scanning for the
    # closing brace.
    content = 'Result: {"summary": "Path is C:\\\\Users", "rating": 4} end of message'
    result = parse_json_to_structured(content, ProductReview)
    assert result.summary == "Path is C:\\Users"


def test_parse_json_to_structured_handles_escaped_quote_before_closing_brace() -> None:
    # Ensures an escaped quote right before the string's closing quote
    # doesn't get misread as the end of the string during brace matching.
    content = 'Result: {"summary": "Ends with a quote\\"", "rating": 3} end of message'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 3


def test_parse_json_to_structured_handles_multiple_consecutive_escapes() -> None:
    # Two consecutive escaped backslashes must each toggle the `escape`
    # flag correctly rather than one canceling the other out.
    content = 'Result: {"summary": "Double backslash: \\\\\\\\", "rating": 2} end of message'
    result = parse_json_to_structured(content, ProductReview)
    assert result.summary == "Double backslash: \\\\"
    assert result.rating == 2


def test_parse_json_to_structured_returns_on_final_matching_brace() -> None:
    # Forces the scanner through multiple "}" characters where depth stays
    # above 0 (decrement only, no return) before the final "}" that brings
    # depth back to 0 and triggers the return statement.
    content = 'Result: {"summary": "ok", "nested": {"a": {"b": 1}}, "rating": 5} end of message'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 5


# --- Common JSON syntax issue fixes ---


def test_parse_json_to_structured_fixes_trailing_comma() -> None:
    content = '{"summary": "Nice", "rating": 4,}'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 4


def test_parse_json_to_structured_fixes_trailing_comma_in_fenced_block() -> None:
    content = '```json\n{"summary": "Nice", "rating": 4,}\n```'
    result = parse_json_to_structured(content, ProductReview)
    assert result.rating == 4


# --- Failure: no JSON found ---


def test_parse_json_to_structured_no_braces_raises() -> None:
    with pytest.raises(JsonStructuredOutputParseError, match=r"Could not extract"):
        parse_json_to_structured("no json here at all", ProductReview)


def test_parse_json_to_structured_unterminated_object_raises() -> None:
    with pytest.raises(JsonStructuredOutputParseError, match=r"Could not extract"):
        parse_json_to_structured('{"summary": "unterminated"', ProductReview)


def test_parse_json_to_structured_error_carries_raw_content() -> None:
    content = "not json"
    with pytest.raises(JsonStructuredOutputParseError) as exc_info:
        parse_json_to_structured(content, ProductReview)
    assert exc_info.value.raw_content == content


# --- Failure: schema validation ---


def test_parse_json_to_structured_missing_required_field_raises() -> None:
    with pytest.raises(JsonStructuredOutputParseError, match=r"failed schema validation"):
        parse_json_to_structured('{"summary": "Nice"}', ProductReview)


def test_parse_json_to_structured_out_of_range_value_raises() -> None:
    with pytest.raises(JsonStructuredOutputParseError, match=r"failed schema validation"):
        parse_json_to_structured('{"summary": "Nice", "rating": 9}', ProductReview)


def test_parse_json_to_structured_wrong_type_raises() -> None:
    with pytest.raises(JsonStructuredOutputParseError, match=r"failed schema validation"):
        parse_json_to_structured('{"summary": "Nice", "rating": "five"}', ProductReview)


def test_parse_json_to_structured_validation_error_carries_raw_content() -> None:
    content = '{"summary": "Nice", "rating": 9}'
    with pytest.raises(JsonStructuredOutputParseError) as exc_info:
        parse_json_to_structured(content, ProductReview)
    assert exc_info.value.raw_content == content


def test_parse_json_to_structured_plain_key_value_text_raises() -> None:
    content = "Title: Dune: Part Two\nSentiment: Positive\nScore: 9/10"
    with pytest.raises(JsonStructuredOutputParseError, match=r"Could not extract"):
        parse_json_to_structured(content, MovieReview)


#######################################################
#     Tests for parse_json_to_structured_with_retry    #
#######################################################


# --- Success paths ---


def test_parse_json_to_structured_with_retry_succeeds_on_first_attempt() -> None:
    calls = []

    def invoke_fn() -> str:
        calls.append(1)
        return '{"summary": "Nice", "rating": 4}'

    result = parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=3)
    assert result.rating == 4
    assert len(calls) == 1


def test_parse_json_to_structured_with_retry_succeeds_after_failures() -> None:
    responses = iter(["not json", '{"summary": "Nice"}', '{"summary": "Nice", "rating": 4}'])

    def invoke_fn() -> str:
        return next(responses)

    result = parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=3)
    assert result.rating == 4


def test_parse_json_to_structured_with_retry_calls_invoke_fn_once_per_attempt() -> None:
    call_count = 0
    responses = ["bad"] * 2 + ['{"summary": "Nice", "rating": 4}']

    def invoke_fn() -> str:
        nonlocal call_count
        result = responses[call_count]
        call_count += 1
        return result

    parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=3)
    assert call_count == 3


# --- Exhaustion ---


def test_parse_json_to_structured_with_retry_raises_after_max_attempts() -> None:
    def invoke_fn() -> str:
        return "not json"

    with pytest.raises(JsonStructuredOutputParseError):
        parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=3)


def test_parse_json_to_structured_with_retry_respects_max_attempts_count() -> None:
    call_count = 0

    def invoke_fn() -> str:
        nonlocal call_count
        call_count += 1
        return "not json"

    with pytest.raises(JsonStructuredOutputParseError):
        parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=5)
    assert call_count == 5


def test_parse_json_to_structured_with_retry_default_max_attempts_is_three() -> None:
    call_count = 0

    def invoke_fn() -> str:
        nonlocal call_count
        call_count += 1
        return "not json"

    with pytest.raises(JsonStructuredOutputParseError):
        parse_json_to_structured_with_retry(invoke_fn, ProductReview)
    assert call_count == 3


def test_parse_json_to_structured_with_retry_error_reflects_last_attempt() -> None:
    responses = iter(['{"summary": "first"}', '{"summary": "last"}'])

    def invoke_fn() -> str:
        return next(responses)

    with pytest.raises(JsonStructuredOutputParseError) as exc_info:
        parse_json_to_structured_with_retry(invoke_fn, ProductReview, max_attempts=2)
    assert exc_info.value.raw_content == '{"summary": "last"}'
