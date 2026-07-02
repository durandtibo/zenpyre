"""Unit tests for LambdaProcessor."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

from zenpyre.data_processors import LambdaProcessor

#######################################
#   Tests for LambdaProcessor         #
#######################################

# --- Constructor ---


def test_lambda_processor_stores_fn() -> None:
    fn = len
    assert LambdaProcessor(fn=fn)._fn is fn


# --- repr and str ---


def test_lambda_processor_repr_contains_class_name() -> None:
    assert "LambdaProcessor" in repr(LambdaProcessor(fn=len))


def test_lambda_processor_str_contains_class_name() -> None:
    assert "LambdaProcessor" in str(LambdaProcessor(fn=len))


def test_lambda_processor_repr_contains_fn() -> None:
    assert "len" in repr(LambdaProcessor(fn=len))


# --- process: applies fn once to the whole input ---


def test_lambda_processor_process_applies_fn_to_whole_input() -> None:
    assert LambdaProcessor(fn=len).process(["a", "b", "c"]) == 3


def test_lambda_processor_process_returns_fn_output_directly() -> None:
    assert LambdaProcessor(fn=sorted).process([3, 1, 2]) == [1, 2, 3]


def test_lambda_processor_process_calls_fn_exactly_once() -> None:
    fn = MagicMock(spec=Callable, return_value="result")
    LambdaProcessor(fn=fn).process([1, 2, 3])
    fn.assert_called_once()


def test_lambda_processor_process_passes_input_unchanged_to_fn() -> None:
    fn = MagicMock(spec=Callable, return_value=None)
    data = [1, 2, 3]
    LambdaProcessor(fn=fn).process(data)
    fn.assert_called_once_with(data)


# --- process: works with non-sequence input ---


def test_lambda_processor_process_works_with_scalar_input() -> None:
    assert LambdaProcessor(fn=str.upper).process("hello") == "HELLO"


def test_lambda_processor_process_works_with_dict_input() -> None:
    assert LambdaProcessor(fn=lambda d: d["key"]).process({"key": "value"}) == "value"


# --- process: type conversion ---


def test_lambda_processor_process_changes_type() -> None:
    result = LambdaProcessor(fn=str).process(42)
    assert result == "42"
    assert isinstance(result, str)


# --- process: lambda function ---


def test_lambda_processor_process_applies_lambda() -> None:
    assert LambdaProcessor(fn=lambda x: x * 2).process(5) == 10
