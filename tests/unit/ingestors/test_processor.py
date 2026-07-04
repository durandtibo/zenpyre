from __future__ import annotations

import pytest

from zenpyre.data_processors import FirstNProcessor
from zenpyre.ingestors import InMemoryIngestor, ProcessorIngestor


def make_ingestor(data: list, n: int) -> ProcessorIngestor:
    return ProcessorIngestor(source=InMemoryIngestor(data=data), processor=FirstNProcessor(n=n))


####################################
#   Tests for ProcessorIngestor    #
####################################

# --- Constructor ---


def test_processor_ingestor_stores_source() -> None:
    source = InMemoryIngestor(data=[1, 2, 3])
    processor = FirstNProcessor(n=2)
    assert ProcessorIngestor(source=source, processor=processor)._source is source


def test_processor_ingestor_stores_processor() -> None:
    source = InMemoryIngestor(data=[1, 2, 3])
    processor = FirstNProcessor(n=2)
    assert ProcessorIngestor(source=source, processor=processor)._processor is processor


# --- repr and str ---


def test_processor_ingestor_repr_contains_class_name() -> None:
    assert repr(make_ingestor(data=[], n=1)).startswith("ProcessorIngestor(")


def test_processor_ingestor_str_contains_class_name() -> None:
    assert str(make_ingestor(data=[], n=1)).startswith("ProcessorIngestor(")


def test_processor_ingestor_repr_contains_source_repr() -> None:
    assert "InMemoryIngestor" in repr(make_ingestor(data=[], n=1))


def test_processor_ingestor_repr_contains_processor_repr() -> None:
    assert "FirstNProcessor" in repr(make_ingestor(data=[], n=1))


# --- ingest: return type ---


def test_processor_ingestor_ingest_returns_list() -> None:
    assert isinstance(make_ingestor(data=[1, 2, 3], n=2).ingest(), list)


# --- ingest: delegates to source then processor ---


def test_processor_ingestor_ingest_returns_processed_output() -> None:
    assert make_ingestor(data=[1, 2, 3, 4, 5], n=3).ingest() == [1, 2, 3]


def test_processor_ingestor_ingest_returns_all_when_n_exceeds_length() -> None:
    assert make_ingestor(data=[1, 2], n=10).ingest() == [1, 2]


def test_processor_ingestor_ingest_returns_empty_list_for_empty_sequence() -> None:
    assert make_ingestor(data=[], n=3).ingest() == []


def test_processor_ingestor_ingest_returns_single_item() -> None:
    assert make_ingestor(data=[1, 2, 3], n=1).ingest() == [1]


def test_processor_ingestor_ingest_preserves_order() -> None:
    result = make_ingestor(data=["a", "b", "c", "d"], n=2).ingest()
    assert result == ["a", "b"]


# --- ingest: accepts tuple input from source ---


def test_processor_ingestor_ingest_accepts_tuple_from_source() -> None:
    ingestor = ProcessorIngestor(
        source=InMemoryIngestor(data=(1, 2, 3, 4)),
        processor=FirstNProcessor(n=2),
    )
    assert ingestor.ingest() == [1, 2]


# --- ingest: calls source and processor exactly once ---


def test_processor_ingestor_ingest_calls_source_ingest() -> None:
    class RecordingIngestor(InMemoryIngestor):
        def __init__(self, data: list) -> None:
            super().__init__(data=data)
            self.call_count = 0

        def ingest(self) -> list:
            self.call_count += 1
            return super().ingest()

    source = RecordingIngestor(data=[1, 2, 3])
    ProcessorIngestor(source=source, processor=FirstNProcessor(n=2)).ingest()
    assert source.call_count == 1


def test_processor_ingestor_ingest_calls_processor_process() -> None:
    class RecordingProcessor(FirstNProcessor):
        def __init__(self, n: int) -> None:
            super().__init__(n=n)
            self.call_count = 0
            self.received_data = None

        def process(self, data: list) -> list:
            self.call_count += 1
            self.received_data = data
            return super().process(data)

    processor = RecordingProcessor(n=2)
    ProcessorIngestor(source=InMemoryIngestor(data=[1, 2, 3]), processor=processor).ingest()
    assert processor.call_count == 1
    assert processor.received_data == [1, 2, 3]


# --- ingest: propagates errors ---


def test_processor_ingestor_ingest_propagates_source_error() -> None:
    class FailingIngestor(InMemoryIngestor):
        def ingest(self) -> list:
            msg = "source failed"
            raise RuntimeError(msg)

    ingestor = ProcessorIngestor(
        source=FailingIngestor(data=[]),
        processor=FirstNProcessor(n=1),
    )
    with pytest.raises(RuntimeError, match=r"source failed"):
        ingestor.ingest()


def test_processor_ingestor_ingest_propagates_processor_error() -> None:
    class FailingProcessor(FirstNProcessor):
        def process(self, data: list) -> list:  # noqa: ARG002
            msg = "processor failed"
            raise RuntimeError(msg)

    ingestor = ProcessorIngestor(
        source=InMemoryIngestor(data=[1, 2, 3]),
        processor=FailingProcessor(n=1),
    )
    with pytest.raises(RuntimeError, match=r"processor failed"):
        ingestor.ingest()
