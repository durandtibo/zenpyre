from __future__ import annotations

import pytest

from zenpyre.data_processors import (
    BaseProcessor,
    FirstNProcessor,
    resolve_data_processor,
)

FIRST_N_PROCESSOR_TARGET = "zenpyre.data_processors.FirstNProcessor"


def _make_processor() -> FirstNProcessor:
    """Return a FirstNProcessor instance for testing."""
    return FirstNProcessor(n=5)


##############################################
#     Tests for resolve_data_processor       #
##############################################


# --- Pass-through ---


def test_resolve_data_processor_returns_base_processor_instance() -> None:
    assert isinstance(resolve_data_processor(_make_processor()), BaseProcessor)


def test_resolve_data_processor_passthrough_returns_same_instance() -> None:
    processor = _make_processor()
    assert resolve_data_processor(processor) is processor


# --- From dict ---


def test_resolve_data_processor_from_dict_returns_base_processor() -> None:
    result = resolve_data_processor({"_target_": FIRST_N_PROCESSOR_TARGET, "n": 5})
    assert isinstance(result, BaseProcessor)


def test_resolve_data_processor_from_dict_returns_correct_type() -> None:
    result = resolve_data_processor({"_target_": FIRST_N_PROCESSOR_TARGET, "n": 5})
    assert isinstance(result, FirstNProcessor)


# --- Invalid input ---


def test_resolve_data_processor_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseProcessor instance"):
        resolve_data_processor("not-a-processor")
