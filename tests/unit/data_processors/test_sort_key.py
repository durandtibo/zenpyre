"""Unit tests for SortByKeyProcessor."""

from __future__ import annotations

import pytest

from zenpyre.data_processors import SortByKeyProcessor

########################################
#   Tests for SortByKeyProcessor       #
########################################


# --- Constructor ---


def test_sort_by_key_processor_stores_key() -> None:
    assert SortByKeyProcessor(key="score")._key == "score"


def test_sort_by_key_processor_reverse_default_false() -> None:
    assert SortByKeyProcessor(key="score")._reverse is False


def test_sort_by_key_processor_stores_reverse() -> None:
    assert SortByKeyProcessor(key="score", reverse=True)._reverse is True


def test_sort_by_key_processor_repr_contains_class_name() -> None:
    assert "SortByKeyProcessor" in repr(SortByKeyProcessor(key="score"))


def test_sort_by_key_processor_str_contains_class_name() -> None:
    assert "SortByKeyProcessor" in str(SortByKeyProcessor(key="score"))


def test_sort_by_key_processor_repr_contains_key() -> None:
    assert "score" in repr(SortByKeyProcessor(key="score"))


# --- process ---


def test_sort_by_key_processor_process_returns_list() -> None:
    data = [{"score": 1}]
    assert isinstance(SortByKeyProcessor(key="score").process(data), list)


def test_sort_by_key_processor_process_sorts_ascending() -> None:
    data = [{"score": 3}, {"score": 1}, {"score": 2}]
    assert SortByKeyProcessor(key="score").process(data) == [
        {"score": 1},
        {"score": 2},
        {"score": 3},
    ]


def test_sort_by_key_processor_process_sorts_descending() -> None:
    data = [{"score": 3}, {"score": 1}, {"score": 2}]
    assert SortByKeyProcessor(key="score", reverse=True).process(data) == [
        {"score": 3},
        {"score": 2},
        {"score": 1},
    ]


def test_sort_by_key_processor_process_already_sorted() -> None:
    data = [{"score": 1}, {"score": 2}, {"score": 3}]
    assert SortByKeyProcessor(key="score").process(data) == data


def test_sort_by_key_processor_process_single_item() -> None:
    data = [{"score": 42}]
    assert SortByKeyProcessor(key="score").process(data) == [{"score": 42}]


def test_sort_by_key_processor_process_string_values() -> None:
    data = [{"name": "Charlie"}, {"name": "Alice"}, {"name": "Bob"}]
    assert SortByKeyProcessor(key="name").process(data) == [
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"},
    ]


def test_sort_by_key_processor_process_dicts_with_extra_keys() -> None:
    data = [{"score": 3, "label": "C"}, {"score": 1, "label": "A"}, {"score": 2, "label": "B"}]
    assert SortByKeyProcessor(key="score").process(data) == [
        {"score": 1, "label": "A"},
        {"score": 2, "label": "B"},
        {"score": 3, "label": "C"},
    ]


def test_sort_by_key_processor_process_missing_key_raises() -> None:
    data = [{"score": 1}, {"other": 2}]
    with pytest.raises(KeyError):
        SortByKeyProcessor(key="score").process(data)


def test_sort_by_key_processor_process_does_not_mutate_input() -> None:
    data = [{"score": 3}, {"score": 1}, {"score": 2}]
    original = [d.copy() for d in data]
    SortByKeyProcessor(key="score").process(data)
    assert data == original


def test_sort_by_key_processor_process_empty_sequence_returns_empty_list() -> None:
    assert SortByKeyProcessor(key="score").process([]) == []


def test_sort_by_key_processor_process_tuple_input() -> None:
    data = ({"score": 3}, {"score": 1})
    result = SortByKeyProcessor(key="score").process(data)
    assert result == [{"score": 1}, {"score": 3}]
