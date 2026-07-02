from __future__ import annotations

import pytest

from zenpyre.data_processors import FirstNProcessor

########################################
#   Tests for FirstNProcessor          #
########################################


# --- Constructor ---


def test_first_n_processor_stores_n() -> None:
    assert FirstNProcessor(n=3)._n == 3


def test_first_n_processor_n_zero_raises() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer, got 0"):
        FirstNProcessor(n=0)


def test_first_n_processor_n_negative_raises() -> None:
    with pytest.raises(ValueError, match="n must be a positive integer, got -1"):
        FirstNProcessor(n=-1)


def test_first_n_processor_n_one_is_valid() -> None:
    assert FirstNProcessor(n=1)._n == 1


def test_first_n_processor_repr_contains_class_name() -> None:
    assert "FirstNProcessor" in repr(FirstNProcessor(n=2))


def test_first_n_processor_str_contains_class_name() -> None:
    assert "FirstNProcessor" in str(FirstNProcessor(n=2))


def test_first_n_processor_repr_contains_n() -> None:
    assert "3" in repr(FirstNProcessor(n=3))


# --- process ---


def test_first_n_processor_process_returns_list() -> None:
    assert isinstance(FirstNProcessor(n=2).process([1, 2, 3]), list)


def test_first_n_processor_process_returns_first_n_items() -> None:
    assert FirstNProcessor(n=3).process([1, 2, 3, 4, 5]) == [1, 2, 3]


def test_first_n_processor_process_returns_all_when_sequence_shorter_than_n() -> None:
    assert FirstNProcessor(n=10).process([1, 2, 3]) == [1, 2, 3]


def test_first_n_processor_process_returns_all_when_sequence_equals_n() -> None:
    assert FirstNProcessor(n=3).process([1, 2, 3]) == [1, 2, 3]


def test_first_n_processor_process_returns_single_item() -> None:
    assert FirstNProcessor(n=1).process([10, 20, 30]) == [10]


def test_first_n_processor_process_empty_sequence_returns_empty_list() -> None:
    assert FirstNProcessor(n=5).process([]) == []


def test_first_n_processor_process_does_not_mutate_input() -> None:
    data = [3, 1, 2, 4]
    FirstNProcessor(n=2).process(data)
    assert data == [3, 1, 2, 4]


def test_first_n_processor_process_preserves_order() -> None:
    assert FirstNProcessor(n=3).process([5, 3, 1, 4, 2]) == [5, 3, 1]


def test_first_n_processor_process_works_with_strings() -> None:
    assert FirstNProcessor(n=2).process(["a", "b", "c"]) == ["a", "b"]


def test_first_n_processor_process_works_with_tuple_input() -> None:
    assert FirstNProcessor(n=2).process((10, 20, 30)) == [10, 20]
