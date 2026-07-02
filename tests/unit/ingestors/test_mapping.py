from __future__ import annotations

from typing import Any

from coola.equality import objects_are_equal

from zenpyre.ingestors import InMemoryIngestor, MappingIngestor

#################################
#   Tests for MappingIngestor   #
#################################

# --- Constructor ---


def test_mapping_ingestor_stores_ingestors() -> None:
    inner = {"a": InMemoryIngestor(data=1), "b": InMemoryIngestor(data=2)}
    assert objects_are_equal(MappingIngestor(ingestors=inner)._ingestors, inner)


# --- repr and str ---


def test_mapping_ingestor_repr_contains_class_name() -> None:
    assert repr(MappingIngestor(ingestors={})).startswith("MappingIngestor(")


def test_mapping_ingestor_str_contains_class_name() -> None:
    assert str(MappingIngestor(ingestors={})).startswith("MappingIngestor(")


# --- ingest: return type ---


def test_mapping_ingestor_ingest_returns_dict() -> None:
    assert isinstance(MappingIngestor(ingestors={}).ingest(), dict)


def test_mapping_ingestor_ingest_empty_ingestors_returns_empty_dict() -> None:
    assert MappingIngestor(ingestors={}).ingest() == {}


# --- ingest: delegation ---


def test_mapping_ingestor_ingest_calls_each_ingestor_once() -> None:
    results = []

    class TrackingIngestor(InMemoryIngestor[str]):
        def ingest(self) -> str:
            results.append(self._data)
            return self._data

    MappingIngestor(
        ingestors={"a": TrackingIngestor(data="a"), "b": TrackingIngestor(data="b")}
    ).ingest()
    assert results == ["a", "b"]


def test_mapping_ingestor_ingest_returns_correct_keys() -> None:
    ingestors = {
        "10-K": InMemoryIngestor(data="annual"),
        "10-Q": InMemoryIngestor(data="quarterly"),
    }
    result = MappingIngestor(ingestors=ingestors).ingest()
    assert set(result.keys()) == {"10-K", "10-Q"}


def test_mapping_ingestor_ingest_maps_keys_to_correct_values() -> None:
    ingestors = {
        "10-K": InMemoryIngestor(data="annual"),
        "10-Q": InMemoryIngestor(data="quarterly"),
    }
    result = MappingIngestor(ingestors=ingestors).ingest()
    assert result["10-K"] == "annual"
    assert result["10-Q"] == "quarterly"


def test_mapping_ingestor_ingest_returns_correct_number_of_entries() -> None:
    ingestors = {i: InMemoryIngestor(data=i) for i in range(5)}
    assert len(MappingIngestor(ingestors=ingestors).ingest()) == 5


# --- ingest: heterogeneous types ---


def test_mapping_ingestor_ingest_handles_mixed_value_types() -> None:
    ingestors = {
        "text": InMemoryIngestor(data="hello"),
        "number": InMemoryIngestor(data=42),
        "items": InMemoryIngestor(data=[1, 2, 3]),
    }
    result = MappingIngestor[Any](ingestors=ingestors).ingest()
    assert result == {"text": "hello", "number": 42, "items": [1, 2, 3]}


def test_mapping_ingestor_ingest_handles_non_string_keys() -> None:
    ingestors = {1: InMemoryIngestor(data="one"), 2: InMemoryIngestor(data="two")}
    result = MappingIngestor(ingestors=ingestors).ingest()
    assert result == {1: "one", 2: "two"}


# --- ingest: single ingestor ---


def test_mapping_ingestor_ingest_single_ingestor_returns_single_entry_dict() -> None:
    result = MappingIngestor(ingestors={"only": InMemoryIngestor(data="value")}).ingest()
    assert result == {"only": "value"}
