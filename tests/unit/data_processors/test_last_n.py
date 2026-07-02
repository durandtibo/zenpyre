from __future__ import annotations

import pytest

from zenpyre.data_processors import LastNProcessor

########################################
#   Tests for LastNProcessor           #
########################################


# --- Constructor ---


def test_last_n_processor_stores_n() -> None:
    assert LastNProcessor(n=3)._n == 3


def test_last_n_processor_n_zero_raises() -> None:
    with pytest.raises(ValueError, match=r"n must be a positive integer, got 0"):
        LastNProcessor(n=0)


def test_last_n_processor_n_negative_raises() -> None:
    with pytest.raises(ValueError, match=r"n must be a positive integer, got -1"):
        LastNProcessor(n=-1)


def test_last_n_processor_n_one_is_valid() -> None:
    assert LastNProcessor(n=1)._n == 1


def test_last_n_processor_repr_contains_class_name() -> None:
    assert "LastNProcessor" in repr(LastNProcessor(n=2))


def test_last_n_processor_str_contains_class_name() -> None:
    assert "LastNProcessor" in str(LastNProcessor(n=2))


def test_last_n_processor_repr_contains_n() -> None:
    assert "3" in repr(LastNProcessor(n=3))


# --- process ---


def test_last_n_processor_process_returns_list() -> None:
    assert isinstance(LastNProcessor(n=2).process([1, 2, 3]), list)


def test_last_n_processor_process_returns_last_n_items() -> None:
    assert LastNProcessor(n=3).process([1, 2, 3, 4, 5]) == [3, 4, 5]


def test_last_n_processor_process_returns_all_when_sequence_shorter_than_n() -> None:
    assert LastNProcessor(n=10).process([1, 2, 3]) == [1, 2, 3]


def test_last_n_processor_process_returns_all_when_sequence_equals_n() -> None:
    assert LastNProcessor(n=3).process([1, 2, 3]) == [1, 2, 3]


def test_last_n_processor_process_returns_single_item() -> None:
    assert LastNProcessor(n=1).process([10, 20, 30]) == [30]


def test_last_n_processor_process_empty_sequence_returns_empty_list() -> None:
    assert LastNProcessor(n=5).process([]) == []


def test_last_n_processor_process_does_not_mutate_input() -> None:
    data = [3, 1, 2, 4]
    LastNProcessor(n=2).process(data)
    assert data == [3, 1, 2, 4]


def test_last_n_processor_process_preserves_order() -> None:
    assert LastNProcessor(n=3).process([5, 3, 1, 4, 2]) == [1, 4, 2]


def test_last_n_processor_process_works_with_strings() -> None:
    assert LastNProcessor(n=2).process(["a", "b", "c"]) == ["b", "c"]


def test_last_n_processor_process_works_with_tuple_input() -> None:
    assert LastNProcessor(n=2).process((10, 20, 30)) == [20, 30]
