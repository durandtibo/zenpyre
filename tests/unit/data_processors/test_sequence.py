from __future__ import annotations

from unittest.mock import MagicMock

from zenpyre.data_processors import BaseProcessor, LambdaProcessor, SequenceProcessor

MODULE = "zenpyre.data_processors.sequence"


#########################################
#   Tests for SequenceProcessor         #
#########################################


# --- Constructor ---


def test_sequence_processor_stores_processor() -> None:
    processor = LambdaProcessor(fn=len)
    assert SequenceProcessor(processor=processor)._processor is processor


def test_sequence_processor_default_progress_description() -> None:
    processor = SequenceProcessor(processor=LambdaProcessor(fn=len))
    assert processor._progress_description == "Processing items..."


def test_sequence_processor_stores_custom_progress_description() -> None:
    processor = SequenceProcessor(
        processor=LambdaProcessor(fn=len),
        progress_description="Custom description",
    )
    assert processor._progress_description == "Custom description"


def test_sequence_processor_repr_contains_class_name() -> None:
    assert "SequenceProcessor" in repr(SequenceProcessor(processor=LambdaProcessor(fn=len)))


def test_sequence_processor_str_contains_class_name() -> None:
    assert "SequenceProcessor" in str(SequenceProcessor(processor=LambdaProcessor(fn=len)))


# --- process ---


def test_sequence_processor_process_returns_list() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len)).process(["a", "bb"])
    assert isinstance(result, list)


def test_sequence_processor_process_applies_processor_to_each_item() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=str.upper)).process(["hello", "world"])
    assert result == ["HELLO", "WORLD"]


def test_sequence_processor_process_preserves_order() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len)).process(["a", "bb", "ccc"])
    assert result == [1, 2, 3]


def test_sequence_processor_process_calls_processor_once_per_item() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.return_value = 0
    SequenceProcessor(processor=mock_processor).process([1, 2, 3])
    assert mock_processor.process.call_count == 3


def test_sequence_processor_process_passes_each_item_to_processor() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = lambda x: x
    SequenceProcessor(processor=mock_processor).process([10, 20, 30])
    calls = [c.args[0] for c in mock_processor.process.call_args_list]
    assert calls == [10, 20, 30]


def test_sequence_processor_process_empty_sequence_returns_empty_list() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len)).process([])
    assert result == []


def test_sequence_processor_process_single_item() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=str.upper)).process(["hello"])
    assert result == ["HELLO"]
