from __future__ import annotations

import pytest

from zenpyre.ingestors import InMemoryIngestor, LastNIngestor


def make_ingestor(data: list, n: int) -> LastNIngestor:
    return LastNIngestor(source=InMemoryIngestor(data=data), n=n)


#################################
#   Tests for LastNIngestor     #
#################################

# --- Constructor ---


def test_last_n_ingestor_stores_ingestor() -> None:
    inner = InMemoryIngestor(data=[1, 2, 3])
    assert LastNIngestor(source=inner, n=2)._source is inner


def test_last_n_ingestor_stores_n() -> None:
    assert LastNIngestor(source=InMemoryIngestor(data=[]), n=3)._n == 3


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(-10, id="large-negative"),
    ],
)
def test_last_n_ingestor_raises_for_non_positive_n(n: int) -> None:
    with pytest.raises(ValueError, match=r"n must be a positive integer"):
        LastNIngestor(source=InMemoryIngestor(data=[]), n=n)


# --- repr and str ---


def test_last_n_ingestor_repr() -> None:
    assert repr(make_ingestor(data=[], n=1)).startswith("LastNIngestor(")


def test_last_n_ingestor_str() -> None:
    assert str(make_ingestor(data=[], n=1)).startswith("LastNIngestor(")


# --- ingest: return type ---


def test_last_n_ingestor_ingest_returns_list() -> None:
    assert isinstance(make_ingestor(data=[1, 2, 3], n=2).ingest(), list)


# --- ingest: slicing ---


def test_last_n_ingestor_ingest_returns_last_n_items() -> None:
    assert make_ingestor(data=[1, 2, 3, 4, 5], n=3).ingest() == [3, 4, 5]


def test_last_n_ingestor_ingest_returns_last_one_item() -> None:
    assert make_ingestor(data=[1, 2, 3], n=1).ingest() == [3]


def test_last_n_ingestor_ingest_returns_all_when_n_equals_length() -> None:
    assert make_ingestor(data=[1, 2, 3], n=3).ingest() == [1, 2, 3]


def test_last_n_ingestor_ingest_returns_all_when_n_exceeds_length() -> None:
    assert make_ingestor(data=[1, 2], n=10).ingest() == [1, 2]


def test_last_n_ingestor_ingest_returns_empty_list_for_empty_sequence() -> None:
    assert make_ingestor(data=[], n=3).ingest() == []


def test_last_n_ingestor_ingest_preserves_order() -> None:
    result = make_ingestor(data=["a", "b", "c", "d"], n=2).ingest()
    assert result == ["c", "d"]


# --- ingest: accepts tuple input ---


def test_last_n_ingestor_ingest_accepts_tuple_sequence() -> None:
    ingestor = LastNIngestor(source=InMemoryIngestor(data=(1, 2, 3, 4)), n=2)
    assert ingestor.ingest() == [3, 4]
