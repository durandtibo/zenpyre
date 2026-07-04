from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

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


def test_sequence_processor_default_raise_on_error_is_true() -> None:
    processor = SequenceProcessor(processor=LambdaProcessor(fn=len))
    assert processor._raise_on_error is True


def test_sequence_processor_stores_custom_raise_on_error() -> None:
    processor = SequenceProcessor(processor=LambdaProcessor(fn=len), raise_on_error=False)
    assert processor._raise_on_error is False


def test_sequence_processor_default_max_workers_is_zero() -> None:
    processor = SequenceProcessor(processor=LambdaProcessor(fn=len))
    assert processor._max_workers == 0


def test_sequence_processor_stores_custom_max_workers() -> None:
    processor = SequenceProcessor(processor=LambdaProcessor(fn=len), max_workers=4)
    assert processor._max_workers == 4


@pytest.mark.parametrize("max_workers", [-1, -10])
def test_sequence_processor_negative_max_workers_raises(max_workers: int) -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 0"):
        SequenceProcessor(processor=LambdaProcessor(fn=len), max_workers=max_workers)


def test_sequence_processor_repr_contains_class_name() -> None:
    assert "SequenceProcessor" in repr(SequenceProcessor(processor=LambdaProcessor(fn=len)))


def test_sequence_processor_repr_contains_raise_on_error() -> None:
    assert "raise_on_error" in repr(SequenceProcessor(processor=LambdaProcessor(fn=len)))


def test_sequence_processor_repr_contains_max_workers() -> None:
    assert "max_workers" in repr(SequenceProcessor(processor=LambdaProcessor(fn=len)))


def test_sequence_processor_str_contains_class_name() -> None:
    assert "SequenceProcessor" in str(SequenceProcessor(processor=LambdaProcessor(fn=len)))


# --- process: sequential (max_workers=0, default) ---


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


# --- process: raise_on_error=True (default), sequential ---


def test_sequence_processor_process_raises_by_default_on_item_error() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        SequenceProcessor(processor=mock_processor).process([1])


def test_sequence_processor_process_raise_on_error_true_explicit() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    with pytest.raises(ValueError, match="boom"):
        SequenceProcessor(processor=mock_processor, raise_on_error=True).process([1])


def test_sequence_processor_process_stops_at_first_error_when_raising() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = [1, ValueError("boom"), 3]
    with pytest.raises(ValueError, match="boom"):
        SequenceProcessor(processor=mock_processor).process([1, 2, 3])
    assert mock_processor.process.call_count == 2


# --- process: raise_on_error=False, sequential ---


def test_sequence_processor_process_skips_failed_items_when_not_raising() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = [1, ValueError("boom"), 3]
    result = SequenceProcessor(processor=mock_processor, raise_on_error=False).process([1, 2, 3])
    assert result == [1, 3]


def test_sequence_processor_process_continues_processing_all_items_when_not_raising() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = [1, ValueError("boom"), 3]
    SequenceProcessor(processor=mock_processor, raise_on_error=False).process([1, 2, 3])
    assert mock_processor.process.call_count == 3


def test_sequence_processor_process_all_items_fail_returns_empty_list() -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    result = SequenceProcessor(processor=mock_processor, raise_on_error=False).process([1, 2, 3])
    assert result == []


def test_sequence_processor_process_no_errors_when_not_raising() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len), raise_on_error=False).process(
        ["a", "bb", "ccc"]
    )
    assert result == [1, 2, 3]


def test_sequence_processor_process_logs_exception_for_failed_item(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    with caplog.at_level("ERROR", logger=MODULE):
        SequenceProcessor(processor=mock_processor, raise_on_error=False).process([1])
    assert "Failed to process item" in caplog.text


def test_sequence_processor_process_logs_warning_with_skip_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = [1, ValueError("boom"), ValueError("boom")]
    with caplog.at_level("WARNING", logger=MODULE):
        SequenceProcessor(processor=mock_processor, raise_on_error=False).process([1, 2, 3])
    assert "Skipped 2 item(s)" in caplog.text


def test_sequence_processor_process_no_warning_logged_when_no_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger=MODULE):
        SequenceProcessor(processor=LambdaProcessor(fn=len), raise_on_error=False).process(
            ["a", "bb"]
        )
    assert "Skipped" not in caplog.text


def test_sequence_processor_process_empty_sequence_not_raising_returns_empty_list() -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len), raise_on_error=False).process([])
    assert result == []


# --- process: parallel path (max_workers >= 1) ---
#
# NOTE: with max_workers > 1, tasks may run out of order across threads, so
# these tests avoid relying on `side_effect` call sequences and instead use
# item-dependent behavior (e.g. raising based on the item's value) to stay
# deterministic regardless of execution order.


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_returns_list(max_workers: int) -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len), max_workers=max_workers).process(
        ["a", "bb"]
    )
    assert isinstance(result, list)


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_applies_processor_to_each_item(
    max_workers: int,
) -> None:
    result = SequenceProcessor(
        processor=LambdaProcessor(fn=str.upper), max_workers=max_workers
    ).process(["hello", "world"])
    assert result == ["HELLO", "WORLD"]


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_preserves_order(max_workers: int) -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len), max_workers=max_workers).process(
        ["a", "bb", "ccc", "dddd", "eeeee"]
    )
    assert result == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_calls_processor_once_per_item(
    max_workers: int,
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.return_value = 0
    SequenceProcessor(processor=mock_processor, max_workers=max_workers).process([1, 2, 3])
    assert mock_processor.process.call_count == 3


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_passes_each_item_to_processor(
    max_workers: int,
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = lambda x: x
    SequenceProcessor(processor=mock_processor, max_workers=max_workers).process([10, 20, 30])
    calls = {c.args[0] for c in mock_processor.process.call_args_list}
    assert calls == {10, 20, 30}


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_empty_sequence_returns_empty_list(
    max_workers: int,
) -> None:
    result = SequenceProcessor(processor=LambdaProcessor(fn=len), max_workers=max_workers).process(
        []
    )
    assert result == []


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_single_item(max_workers: int) -> None:
    result = SequenceProcessor(
        processor=LambdaProcessor(fn=str.upper), max_workers=max_workers
    ).process(["hello"])
    assert result == ["HELLO"]


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_raises_by_default_on_item_error(
    max_workers: int,
) -> None:
    def fn(x: int) -> int:
        if x == 2:
            msg = "boom"
            raise ValueError(msg)
        return x

    with pytest.raises(ValueError, match="boom"):
        SequenceProcessor(processor=LambdaProcessor(fn=fn), max_workers=max_workers).process(
            [1, 2, 3]
        )


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_skips_failed_items_when_not_raising(
    max_workers: int,
) -> None:
    def fn(x: int) -> int:
        if x == 2:
            msg = "boom"
            raise ValueError(msg)
        return x

    result = SequenceProcessor(
        processor=LambdaProcessor(fn=fn), raise_on_error=False, max_workers=max_workers
    ).process([1, 2, 3])
    assert result == [1, 3]


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_continues_processing_all_items_when_not_raising(
    max_workers: int,
) -> None:
    def fn(x: int) -> int:
        if x == 2:
            msg = "boom"
            raise ValueError(msg)
        return x

    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = fn
    SequenceProcessor(
        processor=mock_processor, raise_on_error=False, max_workers=max_workers
    ).process([1, 2, 3])
    assert mock_processor.process.call_count == 3


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_all_items_fail_returns_empty_list(
    max_workers: int,
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    result = SequenceProcessor(
        processor=mock_processor, raise_on_error=False, max_workers=max_workers
    ).process([1, 2, 3])
    assert result == []


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_no_errors_when_not_raising(
    max_workers: int,
) -> None:
    result = SequenceProcessor(
        processor=LambdaProcessor(fn=len), raise_on_error=False, max_workers=max_workers
    ).process(["a", "bb", "ccc"])
    assert result == [1, 2, 3]


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_logs_exception_for_failed_item(
    max_workers: int, caplog: pytest.LogCaptureFixture
) -> None:
    mock_processor = MagicMock(spec=BaseProcessor)
    mock_processor.process.side_effect = ValueError("boom")
    with caplog.at_level("ERROR", logger=MODULE):
        SequenceProcessor(
            processor=mock_processor, raise_on_error=False, max_workers=max_workers
        ).process([1])
    assert "Failed to process item" in caplog.text


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_logs_warning_with_skip_count(
    max_workers: int, caplog: pytest.LogCaptureFixture
) -> None:
    def fn(x: int) -> int:
        if x in (2, 3):
            msg = "boom"
            raise ValueError(msg)
        return x

    with caplog.at_level("WARNING", logger=MODULE):
        SequenceProcessor(
            processor=LambdaProcessor(fn=fn), raise_on_error=False, max_workers=max_workers
        ).process([1, 2, 3])
    assert "Skipped 2 item(s)" in caplog.text


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_no_warning_logged_when_no_errors(
    max_workers: int, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger=MODULE):
        SequenceProcessor(
            processor=LambdaProcessor(fn=len), raise_on_error=False, max_workers=max_workers
        ).process(["a", "bb"])
    assert "Skipped" not in caplog.text


@pytest.mark.parametrize("max_workers", [1, 2, 4])
def test_sequence_processor_process_parallel_empty_sequence_not_raising_returns_empty_list(
    max_workers: int,
) -> None:
    result = SequenceProcessor(
        processor=LambdaProcessor(fn=len), raise_on_error=False, max_workers=max_workers
    ).process([])
    assert result == []


def test_sequence_processor_process_parallel_uses_multiple_threads() -> None:
    """Verify that with max_workers > 1, items are actually processed
    concurrently rather than one thread handling everything."""
    seen_threads: set[int] = set()
    lock = threading.Lock()

    def fn(x: int) -> int:
        with lock:
            seen_threads.add(threading.get_ident())
        time.sleep(0.05)
        return x

    SequenceProcessor(processor=LambdaProcessor(fn=fn), max_workers=4).process([1, 2, 3, 4])
    assert len(seen_threads) > 1


def test_sequence_processor_process_sequential_uses_single_thread() -> None:
    """Verify that max_workers=0 (default) runs entirely on the calling
    thread, with no thread pool involved."""
    main_thread = threading.get_ident()
    seen_threads: set[int] = set()

    def fn(x: int) -> int:
        seen_threads.add(threading.get_ident())
        return x

    SequenceProcessor(processor=LambdaProcessor(fn=fn), max_workers=0).process([1, 2, 3])
    assert seen_threads == {main_thread}
