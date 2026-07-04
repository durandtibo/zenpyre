from __future__ import annotations

import random

import pytest

from zenpyre.data_processors import ShuffleProcessor


def make_processor(seed: int | None = None) -> ShuffleProcessor:
    return ShuffleProcessor(seed=seed)


##################################
#   Tests for ShuffleProcessor   #
##################################

# --- Constructor ---


def test_shuffle_processor_stores_seed() -> None:
    assert ShuffleProcessor(seed=42)._seed == 42


def test_shuffle_processor_default_seed_is_none() -> None:
    assert ShuffleProcessor()._seed is None


def test_shuffle_processor_creates_random_instance() -> None:
    assert isinstance(ShuffleProcessor()._rng, random.Random)


# --- repr and str ---


def test_shuffle_processor_repr_contains_class_name() -> None:
    assert repr(make_processor()).startswith("ShuffleProcessor(")


def test_shuffle_processor_str_contains_class_name() -> None:
    assert str(make_processor()).startswith("ShuffleProcessor(")


def test_shuffle_processor_repr_contains_seed() -> None:
    assert "42" in repr(make_processor(seed=42))


# --- process: return type ---


def test_shuffle_processor_process_returns_list() -> None:
    assert isinstance(make_processor(seed=1).process([1, 2, 3]), list)


# --- process: content preservation ---


def test_shuffle_processor_process_preserves_all_items() -> None:
    result = make_processor(seed=1).process([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]


def test_shuffle_processor_process_preserves_length() -> None:
    result = make_processor(seed=1).process([1, 2, 3, 4, 5])
    assert len(result) == 5


def test_shuffle_processor_process_returns_empty_list_for_empty_sequence() -> None:
    assert make_processor(seed=1).process([]) == []


def test_shuffle_processor_process_returns_single_item_unchanged() -> None:
    assert make_processor(seed=1).process([42]) == [42]


def test_shuffle_processor_process_preserves_duplicate_items() -> None:
    result = make_processor(seed=1).process([1, 1, 2, 2, 3])
    assert sorted(result) == [1, 1, 2, 2, 3]


# --- process: accepts tuple input ---


def test_shuffle_processor_process_accepts_tuple_sequence() -> None:
    result = make_processor(seed=1).process((1, 2, 3, 4))
    assert sorted(result) == [1, 2, 3, 4]


# --- process: does not mutate input ---


def test_shuffle_processor_process_does_not_mutate_input_list() -> None:
    data = [1, 2, 3, 4, 5]
    make_processor(seed=1).process(data)
    assert data == [1, 2, 3, 4, 5]


# --- process: raises for non-iterable input ---


def test_shuffle_processor_process_raises_for_non_iterable_input() -> None:
    with pytest.raises(TypeError):
        make_processor(seed=1).process(42)


# --- process: seed reproducibility across instances ---


def test_shuffle_processor_process_same_seed_same_instance_data_reproducible() -> None:
    result1 = make_processor(seed=123).process([1, 2, 3, 4, 5])
    result2 = make_processor(seed=123).process([1, 2, 3, 4, 5])
    assert result1 == result2


def test_shuffle_processor_process_different_seeds_can_produce_different_orders() -> None:
    data = list(range(20))
    result1 = make_processor(seed=1).process(data)
    result2 = make_processor(seed=2).process(data)
    assert result1 != result2


# --- process: repeated calls on the same instance advance the RNG state ---


def test_shuffle_processor_process_repeated_calls_can_differ() -> None:
    processor = make_processor(seed=1)
    result1 = processor.process(list(range(20)))
    result2 = processor.process(list(range(20)))
    assert result1 != result2


def test_shuffle_processor_process_repeated_calls_preserve_items() -> None:
    processor = make_processor(seed=1)
    result1 = processor.process([1, 2, 3, 4, 5])
    result2 = processor.process([1, 2, 3, 4, 5])
    assert sorted(result1) == sorted(result2) == [1, 2, 3, 4, 5]


# --- process: without seed still produces a valid shuffle ---


def test_shuffle_processor_process_without_seed_preserves_all_items() -> None:
    result = make_processor().process([1, 2, 3, 4, 5])
    assert sorted(result) == [1, 2, 3, 4, 5]


# --- process: reusing the same processor on different inputs ---


def test_shuffle_processor_process_can_be_reused_on_different_inputs() -> None:
    processor = make_processor(seed=1)
    result1 = processor.process([1, 2, 3])
    result2 = processor.process(["a", "b", "c"])
    assert sorted(result1) == [1, 2, 3]
    assert sorted(result2) == ["a", "b", "c"]
