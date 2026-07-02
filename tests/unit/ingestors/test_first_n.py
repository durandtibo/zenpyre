from __future__ import annotations

import pytest

from zenpyre.ingestors import FirstNIngestor, InMemoryIngestor


def make_ingestor(data: list, n: int) -> FirstNIngestor:
    return FirstNIngestor(ingestor=InMemoryIngestor(data=data), n=n)


##################################
#   Tests for FirstNIngestor     #
##################################

# --- Constructor ---


def test_first_n_ingestor_stores_ingestor() -> None:
    inner = InMemoryIngestor(data=[1, 2, 3])
    assert FirstNIngestor(ingestor=inner, n=2)._ingestor is inner


def test_first_n_ingestor_stores_n() -> None:
    assert FirstNIngestor(ingestor=InMemoryIngestor(data=[]), n=3)._n == 3


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(-10, id="large-negative"),
    ],
)
def test_first_n_ingestor_raises_for_non_positive_n(n: int) -> None:
    with pytest.raises(ValueError, match=r"n must be a positive integer"):
        FirstNIngestor(ingestor=InMemoryIngestor(data=[]), n=n)


# --- repr and str ---


def test_first_n_ingestor_repr_contains_class_name() -> None:
    assert "FirstNIngestor" in repr(make_ingestor(data=[], n=1))


def test_first_n_ingestor_str_contains_class_name() -> None:
    assert "FirstNIngestor" in str(make_ingestor(data=[], n=1))


def test_first_n_ingestor_repr_contains_n() -> None:
    assert "3" in repr(make_ingestor(data=[], n=3))


# --- ingest: return type ---


def test_first_n_ingestor_ingest_returns_list() -> None:
    assert isinstance(make_ingestor(data=[1, 2, 3], n=2).ingest(), list)


# --- ingest: slicing ---


def test_first_n_ingestor_ingest_returns_first_n_items() -> None:
    assert make_ingestor(data=[1, 2, 3, 4, 5], n=3).ingest() == [1, 2, 3]


def test_first_n_ingestor_ingest_returns_first_one_item() -> None:
    assert make_ingestor(data=[1, 2, 3], n=1).ingest() == [1]


def test_first_n_ingestor_ingest_returns_all_when_n_equals_length() -> None:
    assert make_ingestor(data=[1, 2, 3], n=3).ingest() == [1, 2, 3]


def test_first_n_ingestor_ingest_returns_all_when_n_exceeds_length() -> None:
    assert make_ingestor(data=[1, 2], n=10).ingest() == [1, 2]


def test_first_n_ingestor_ingest_returns_empty_list_for_empty_sequence() -> None:
    assert make_ingestor(data=[], n=3).ingest() == []


def test_first_n_ingestor_ingest_preserves_order() -> None:
    result = make_ingestor(data=["a", "b", "c", "d"], n=2).ingest()
    assert result == ["a", "b"]


# --- ingest: accepts tuple input ---


def test_first_n_ingestor_ingest_accepts_tuple_sequence() -> None:
    ingestor = FirstNIngestor(ingestor=InMemoryIngestor(data=(1, 2, 3, 4)), n=2)
    assert ingestor.ingest() == [1, 2]
