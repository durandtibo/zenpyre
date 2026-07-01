from __future__ import annotations

from pathlib import Path

import pytest

from zenpyre.ingestors import InMemoryIngestor

###################################
#   Tests for InMemoryIngestor    #
###################################

# --- Constructor ---


def test_in_memory_ingestor_stores_data() -> None:
    assert InMemoryIngestor(data="hello")._data == "hello"


# --- repr & str ---


def test_in_memory_ingestor_repr_contains_class_name() -> None:
    assert repr(InMemoryIngestor(data="hello")).startswith("InMemoryIngestor(")


def test_in_memory_ingestor_str_contains_class_name() -> None:
    assert str(InMemoryIngestor(data="hello")).startswith("InMemoryIngestor(")


# --- ingest ---


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("hello", id="str"),
        pytest.param(42, id="int"),
        pytest.param(3.14, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="none"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param({"key": "value"}, id="dict"),
        pytest.param(Path("filing.md"), id="path"),
    ],
)
def test_in_memory_ingestor_ingest_returns_data(data: object) -> None:
    assert InMemoryIngestor(data=data).ingest() == data


def test_in_memory_ingestor_ingest_returns_copy() -> None:
    data = object()
    assert InMemoryIngestor(data=data).ingest() is not data


def test_in_memory_ingestor_ingest_returns_same_object() -> None:
    data = object()
    assert InMemoryIngestor(data=data, copy=False).ingest() is data


def test_in_memory_ingestor_ingest_is_idempotent() -> None:
    ingestor = InMemoryIngestor(data="hello")
    assert ingestor.ingest() == ingestor.ingest()


# --- copy parameter ---


def test_in_memory_ingestor_ingest_default_returns_deep_copy() -> None:
    data = [1, 2, 3]
    ingestor = InMemoryIngestor(data=data)
    result = ingestor.ingest()
    assert result == data
    assert result is not data


def test_in_memory_ingestor_ingest_copy_true_returns_deep_copy() -> None:
    data = [1, 2, 3]
    ingestor = InMemoryIngestor(data=data, copy=True)
    result = ingestor.ingest()
    assert result is not data


def test_in_memory_ingestor_ingest_copy_false_returns_same_object() -> None:
    data = [1, 2, 3]
    ingestor = InMemoryIngestor(data=data, copy=False)
    assert ingestor.ingest() is data


def test_in_memory_ingestor_ingest_copy_false_mutations_affect_stored_data() -> None:
    data = [1, 2, 3]
    ingestor = InMemoryIngestor(data=data, copy=False)
    ingestor.ingest().append(4)
    assert ingestor._data == [1, 2, 3, 4]


def test_in_memory_ingestor_ingest_copy_true_mutations_do_not_affect_stored_data() -> None:
    data = [1, 2, 3]
    ingestor = InMemoryIngestor(data=data, copy=True)
    ingestor.ingest().append(4)
    assert ingestor._data == [1, 2, 3]


def test_in_memory_ingestor_default_copy_is_true() -> None:
    assert InMemoryIngestor(data=[])._copy is True


def test_in_memory_ingestor_copy_false_stored() -> None:
    assert InMemoryIngestor(data=[], copy=False)._copy is False
