from __future__ import annotations

from unittest.mock import patch

import pytest

from zenpyre.data_processors import SortRecordsByMetadataProcessor
from zenpyre.records import Record

MODULE = "zenpyre.data_processors.sort_records"


@pytest.fixture
def records() -> list[Record]:
    return [
        Record(id="b", metadata={"source": "b.txt", "page": 2}),
        Record(id="a", metadata={"source": "a.txt", "page": 1}),
        Record(id="c", metadata={"source": "c.txt", "page": 3}),
    ]


###################################################
#   Tests for SortRecordsByMetadataProcessor      #
###################################################


# --- Constructor ---


def test_sort_records_by_metadata_processor_stores_metadata_key() -> None:
    assert SortRecordsByMetadataProcessor(metadata_key="source")._metadata_key == "source"


def test_sort_records_by_metadata_processor_keep_missing_default_true() -> None:
    assert SortRecordsByMetadataProcessor(metadata_key="source")._keep_missing is True


def test_sort_records_by_metadata_processor_stores_keep_missing() -> None:
    p = SortRecordsByMetadataProcessor(metadata_key="source", keep_missing=False)
    assert p._keep_missing is False


def test_sort_records_by_metadata_processor_reverse_default_false() -> None:
    assert SortRecordsByMetadataProcessor(metadata_key="source")._reverse is False


def test_sort_records_by_metadata_processor_stores_reverse() -> None:
    p = SortRecordsByMetadataProcessor(metadata_key="source", reverse=True)
    assert p._reverse is True


def test_sort_records_by_metadata_processor_repr_contains_class_name() -> None:
    assert "SortRecordsByMetadataProcessor" in repr(
        SortRecordsByMetadataProcessor(metadata_key="source")
    )


def test_sort_records_by_metadata_processor_str_contains_class_name() -> None:
    assert "SortRecordsByMetadataProcessor" in str(
        SortRecordsByMetadataProcessor(metadata_key="source")
    )


# --- process ---


def test_sort_records_by_metadata_processor_process_returns_list(
    records: list[Record],
) -> None:
    result = SortRecordsByMetadataProcessor(metadata_key="source").process(records)
    assert isinstance(result, list)


def test_sort_records_by_metadata_processor_process_sorts_ascending(
    records: list[Record],
) -> None:
    result = SortRecordsByMetadataProcessor(metadata_key="source").process(records)
    assert [r.metadata["source"] for r in result] == ["a.txt", "b.txt", "c.txt"]


def test_sort_records_by_metadata_processor_process_sorts_descending(
    records: list[Record],
) -> None:
    result = SortRecordsByMetadataProcessor(metadata_key="source", reverse=True).process(records)
    assert [r.metadata["source"] for r in result] == ["c.txt", "b.txt", "a.txt"]


def test_sort_records_by_metadata_processor_process_keep_missing_true(
    records: list[Record],
) -> None:
    records_with_missing = [*records, Record(id="x")]
    result = SortRecordsByMetadataProcessor(metadata_key="source").process(records_with_missing)
    assert len(result) == 4
    assert result[-1].id == "x"


def test_sort_records_by_metadata_processor_process_keep_missing_false(
    records: list[Record],
) -> None:
    records_with_missing = [*records, Record(id="x")]
    result = SortRecordsByMetadataProcessor(metadata_key="source", keep_missing=False).process(
        records_with_missing
    )
    assert len(result) == 3
    assert all("source" in r.metadata for r in result)


def test_sort_records_by_metadata_processor_process_does_not_mutate_input(
    records: list[Record],
) -> None:
    original_order = [r.id for r in records]
    SortRecordsByMetadataProcessor(metadata_key="source").process(records)
    assert [r.id for r in records] == original_order


def test_sort_records_by_metadata_processor_process_calls_sort_by_metadata(
    records: list[Record],
) -> None:
    with patch(f"{MODULE}.sort_by_metadata", return_value=records) as mock_sort:
        SortRecordsByMetadataProcessor(
            metadata_key="source", keep_missing=False, reverse=True
        ).process(records)
    mock_sort.assert_called_once_with(records, "source", keep_missing=False, reverse=True)


def test_sort_records_by_metadata_processor_process_empty_list() -> None:
    result = SortRecordsByMetadataProcessor(metadata_key="source").process([])
    assert result == []
