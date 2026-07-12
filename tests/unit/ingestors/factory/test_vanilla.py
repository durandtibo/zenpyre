from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.ingestors import BaseIngestor, InMemoryIngestor
from zenpyre.ingestors.factory import BaseIngestorFactory, IngestorFactory


def _make_ingestor() -> InMemoryIngestor:
    """Return an InMemoryIngestor instance for testing."""
    return InMemoryIngestor([1, 2, 3])


##############################################
#     Tests for IngestorFactory              #
##############################################


# --- Inheritance ---


def test_ingestor_factory_is_base_ingestor_factory() -> None:
    assert isinstance(IngestorFactory(_make_ingestor()), BaseIngestorFactory)


# --- make_ingestor ---


def test_ingestor_factory_make_ingestor_returns_base_ingestor() -> None:
    factory = IngestorFactory(_make_ingestor())
    assert isinstance(factory.make_ingestor(), BaseIngestor)


def test_ingestor_factory_make_ingestor_returns_same_instance() -> None:
    ingestor = _make_ingestor()
    factory = IngestorFactory(ingestor)
    assert factory.make_ingestor() is ingestor


def test_ingestor_factory_make_ingestor_returns_same_instance_across_calls() -> None:
    ingestor = _make_ingestor()
    factory = IngestorFactory(ingestor)
    assert factory.make_ingestor() is factory.make_ingestor()


# --- _get_repr_kwargs ---


def test_ingestor_factory_get_repr_kwargs() -> None:
    ingestor = _make_ingestor()
    factory = IngestorFactory(ingestor)
    assert objects_are_equal(factory._get_repr_kwargs(), {"ingestor": ingestor})


# --- __repr__ and __str__ ---


def test_ingestor_factory_repr_starts_with_class_name() -> None:
    factory = IngestorFactory(_make_ingestor())
    assert repr(factory).startswith("IngestorFactory(")


def test_ingestor_factory_str_starts_with_class_name() -> None:
    factory = IngestorFactory(_make_ingestor())
    assert str(factory).startswith("IngestorFactory(")


def test_ingestor_factory_repr_contains_ingestor() -> None:
    factory = IngestorFactory(_make_ingestor())
    assert "ingestor" in repr(factory)


def test_ingestor_factory_str_contains_ingestor() -> None:
    factory = IngestorFactory(_make_ingestor())
    assert "ingestor" in str(factory)
