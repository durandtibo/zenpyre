from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from zenpyre.runnables import InputOutputPair

#####################################
#     Tests for InputOutputPair     #
#####################################


def test_input_output_pair_stores_fields() -> None:
    pair = InputOutputPair(input="hello", output="HELLO")
    assert pair.input == "hello"
    assert pair.output == "HELLO"


def test_input_output_pair_is_frozen() -> None:
    pair = InputOutputPair(input="hello", output="HELLO")
    with pytest.raises(FrozenInstanceError):
        pair.output = "CHANGED"


def test_input_output_pair_equality() -> None:
    pair_a = InputOutputPair(input="hello", output="HELLO")
    pair_b = InputOutputPair(input="hello", output="HELLO")
    assert pair_a == pair_b


def test_input_output_pair_inequality_different_output() -> None:
    pair_a = InputOutputPair(input="hello", output="HELLO")
    pair_b = InputOutputPair(input="hello", output="WORLD")
    assert pair_a != pair_b
