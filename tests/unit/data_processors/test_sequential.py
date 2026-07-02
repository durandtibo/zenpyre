from __future__ import annotations

from unittest.mock import Mock

from zenpyre.data_processors import BaseProcessor, LambdaProcessor, SequentialProcessor

##########################################
#   Tests for SequentialProcessor        #
##########################################


# --- Constructor ---


def test_sequential_processor_stores_processors() -> None:
    p1 = LambdaProcessor(fn=str)
    p2 = LambdaProcessor(fn=len)
    seq = SequentialProcessor(p1, p2)
    assert seq._processors == [p1, p2]


def test_sequential_processor_no_processors_returns_input() -> None:
    assert SequentialProcessor().process(42) == 42


def test_sequential_processor_no_processors_returns_input_unchanged() -> None:
    data = [3, 1, 2]
    assert SequentialProcessor().process(data) is data


def test_sequential_processor_single_processor_is_valid() -> None:
    seq = SequentialProcessor(LambdaProcessor(fn=str))
    assert len(seq._processors) == 1


# --- repr and str ---


def test_sequential_processor_repr_contains_class_name() -> None:
    seq = SequentialProcessor(LambdaProcessor(fn=str))
    assert "SequentialProcessor" in repr(seq)


def test_sequential_processor_str_contains_class_name() -> None:
    seq = SequentialProcessor(LambdaProcessor(fn=str))
    assert "SequentialProcessor" in str(seq)


# --- process ---


def test_sequential_processor_process_single_processor() -> None:
    seq = SequentialProcessor(LambdaProcessor(fn=lambda x: x * 2))
    assert seq.process(21) == 42


def test_sequential_processor_process_chains_processors_in_order() -> None:
    seq = SequentialProcessor(
        LambdaProcessor(fn=lambda x: x * 2),
        LambdaProcessor(fn=str),
    )
    assert seq.process(21) == "42"


def test_sequential_processor_process_passes_output_of_each_as_input_to_next() -> None:
    calls = []

    def track(x: int) -> int:
        calls.append(x)
        return x + 1

    seq = SequentialProcessor(LambdaProcessor(fn=track), LambdaProcessor(fn=track))
    seq.process(0)
    assert calls == [0, 1]


def test_sequential_processor_process_three_processors() -> None:
    seq = SequentialProcessor(
        LambdaProcessor(fn=lambda x: x + 1),
        LambdaProcessor(fn=lambda x: x * 3),
        LambdaProcessor(fn=str),
    )
    assert seq.process(1) == "6"


def test_sequential_processor_process_calls_each_processor_once() -> None:
    p1 = Mock(spec=BaseProcessor, process=Mock(return_value="intermediate"))
    p2 = Mock(spec=BaseProcessor, process=Mock(return_value="final"))
    seq = SequentialProcessor(p1, p2)
    result = seq.process("input")
    p1.process.assert_called_once_with("input")
    p2.process.assert_called_once_with("intermediate")
    assert result == "final"


def test_sequential_processor_process_returns_last_processor_output() -> None:
    seq = SequentialProcessor(
        LambdaProcessor(fn=lambda x: x * 2),
        LambdaProcessor(fn=lambda x: x + 10),
    )
    assert seq.process(5) == 20


def test_sequential_processor_process_does_not_mutate_input() -> None:
    original = [3, 1, 2]
    seq = SequentialProcessor(LambdaProcessor(fn=sorted))
    seq.process(original)
    assert original == [3, 1, 2]
