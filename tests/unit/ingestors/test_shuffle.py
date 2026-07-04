from __future__ import annotations

import random

from zenpyre.ingestors import InMemoryIngestor, ShuffleIngestor


def make_ingestor(data: list, seed: int | None = None) -> ShuffleIngestor:
    return ShuffleIngestor(source=InMemoryIngestor(data=data), seed=seed)


#################################
#   Tests for ShuffleIngestor   #
#################################

# --- Constructor ---


def test_shuffle_ingestor_stores_source() -> None:
    inner = InMemoryIngestor(data=[1, 2, 3])
    assert ShuffleIngestor(source=inner)._source is inner


def test_shuffle_ingestor_stores_seed() -> None:
    assert ShuffleIngestor(source=InMemoryIngestor(data=[]), seed=42)._seed == 42


def test_shuffle_ingestor_default_seed_is_none() -> None:
    assert ShuffleIngestor(source=InMemoryIngestor(data=[]))._seed is None


def test_shuffle_ingestor_creates_random_instance() -> None:
    assert isinstance(ShuffleIngestor(source=InMemoryIngestor(data=[]))._rng, random.Random)


# --- repr and str ---


def test_shuffle_ingestor_repr_contains_class_name() -> None:
    assert repr(make_ingestor(data=[])).startswith("ShuffleIngestor(")


def test_shuffle_ingestor_str_contains_class_name() -> None:
    assert str(make_ingestor(data=[])).startswith("ShuffleIngestor(")


def test_shuffle_ingestor_repr_contains_seed() -> None:
    assert "42" in repr(make_ingestor(data=[], seed=42))


# --- ingest: return type ---


def test_shuffle_ingestor_ingest_returns_list() -> None:
    assert isinstance(make_ingestor(data=[1, 2, 3], seed=1).ingest(), list)


# --- ingest: content preservation ---


def test_shuffle_ingestor_ingest_preserves_all_items() -> None:
    result = make_ingestor(data=[1, 2, 3, 4, 5], seed=1).ingest()
    assert sorted(result) == [1, 2, 3, 4, 5]


def test_shuffle_ingestor_ingest_preserves_length() -> None:
    result = make_ingestor(data=[1, 2, 3, 4, 5], seed=1).ingest()
    assert len(result) == 5


def test_shuffle_ingestor_ingest_returns_empty_list_for_empty_sequence() -> None:
    assert make_ingestor(data=[], seed=1).ingest() == []


def test_shuffle_ingestor_ingest_returns_single_item_unchanged() -> None:
    assert make_ingestor(data=[42], seed=1).ingest() == [42]


def test_shuffle_ingestor_ingest_preserves_duplicate_items() -> None:
    result = make_ingestor(data=[1, 1, 2, 2, 3], seed=1).ingest()
    assert sorted(result) == [1, 1, 2, 2, 3]


# --- ingest: accepts tuple input ---


def test_shuffle_ingestor_ingest_accepts_tuple_sequence() -> None:
    ingestor = ShuffleIngestor(source=InMemoryIngestor(data=(1, 2, 3, 4)), seed=1)
    assert sorted(ingestor.ingest()) == [1, 2, 3, 4]


# --- ingest: does not mutate source data ---


def test_shuffle_ingestor_ingest_does_not_mutate_source_list() -> None:
    data = [1, 2, 3, 4, 5]
    make_ingestor(data=data, seed=1).ingest()
    assert data == [1, 2, 3, 4, 5]


# --- ingest: seed reproducibility across instances ---


def test_shuffle_ingestor_ingest_same_seed_same_instance_data_reproducible() -> None:
    result1 = make_ingestor(data=[1, 2, 3, 4, 5], seed=123).ingest()
    result2 = make_ingestor(data=[1, 2, 3, 4, 5], seed=123).ingest()
    assert result1 == result2


def test_shuffle_ingestor_ingest_different_seeds_can_produce_different_orders() -> None:
    data = list(range(20))
    result1 = make_ingestor(data=data, seed=1).ingest()
    result2 = make_ingestor(data=data, seed=2).ingest()
    assert result1 != result2


# --- ingest: repeated calls on the same instance advance the RNG state ---


def test_shuffle_ingestor_ingest_repeated_calls_can_differ() -> None:
    ingestor = make_ingestor(data=list(range(20)), seed=1)
    result1 = ingestor.ingest()
    result2 = ingestor.ingest()
    assert result1 != result2


def test_shuffle_ingestor_ingest_repeated_calls_preserve_items() -> None:
    ingestor = make_ingestor(data=[1, 2, 3, 4, 5], seed=1)
    result1 = ingestor.ingest()
    result2 = ingestor.ingest()
    assert sorted(result1) == sorted(result2) == [1, 2, 3, 4, 5]


# --- ingest: no seed still produces a valid shuffle ---


def test_shuffle_ingestor_ingest_without_seed_preserves_all_items() -> None:
    result = make_ingestor(data=[1, 2, 3, 4, 5]).ingest()
    assert sorted(result) == [1, 2, 3, 4, 5]
