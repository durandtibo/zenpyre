from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.ingestors import InMemoryIngestor
from zenpyre.ingestors.base import BaseIngestor
from zenpyre.ingestors.factory import (
    BaseIngestorFactory,
    ConfigurableIngestorFactory,
)

IN_MEMORY_INGESTOR_TARGET = "zenpyre.ingestors.InMemoryIngestor"


def _make_ingestor() -> InMemoryIngestor:
    """Return an InMemoryIngestor instance for testing."""
    return InMemoryIngestor([1, 2, 3])


##################################################
#     Tests for ConfigurableIngestorFactory      #
##################################################


# --- Inheritance ---


def test_configurable_ingestor_factory_is_base_ingestor_factory() -> None:
    assert isinstance(ConfigurableIngestorFactory(_make_ingestor()), BaseIngestorFactory)


# --- make_ingestor from instance ---


def test_configurable_ingestor_factory_make_ingestor_returns_base_ingestor() -> None:
    factory = ConfigurableIngestorFactory(_make_ingestor())
    assert isinstance(factory.make_ingestor(), BaseIngestor)


def test_configurable_ingestor_factory_make_ingestor_returns_same_instance() -> None:
    ingestor = _make_ingestor()
    factory = ConfigurableIngestorFactory(ingestor)
    assert factory.make_ingestor() is ingestor


# --- make_ingestor from dict ---


def test_configurable_ingestor_factory_make_ingestor_from_dict_returns_base_ingestor() -> None:
    factory = ConfigurableIngestorFactory(
        {"_target_": IN_MEMORY_INGESTOR_TARGET, "data": [1, 2, 3]}
    )
    assert isinstance(factory.make_ingestor(), BaseIngestor)


def test_configurable_ingestor_factory_make_ingestor_from_dict_returns_correct_type() -> None:
    factory = ConfigurableIngestorFactory(
        {"_target_": IN_MEMORY_INGESTOR_TARGET, "data": [1, 2, 3]}
    )
    assert isinstance(factory.make_ingestor(), InMemoryIngestor)


# --- _get_repr_kwargs ---


def test_configurable_ingestor_factory_get_repr_kwargs_instance() -> None:
    ingestor = _make_ingestor()
    factory = ConfigurableIngestorFactory(ingestor)
    assert objects_are_equal(factory._get_repr_kwargs(), {"ingestor": ingestor})


def test_configurable_ingestor_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": IN_MEMORY_INGESTOR_TARGET, "data": [1, 2, 3]}
    factory = ConfigurableIngestorFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"ingestor": config})


# --- __repr__ and __str__ ---


def test_configurable_ingestor_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableIngestorFactory(_make_ingestor())
    assert repr(factory).startswith("ConfigurableIngestorFactory(")


def test_configurable_ingestor_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableIngestorFactory(_make_ingestor())
    assert str(factory).startswith("ConfigurableIngestorFactory(")


def test_configurable_ingestor_factory_repr_contains_ingestor() -> None:
    factory = ConfigurableIngestorFactory(_make_ingestor())
    assert "ingestor" in repr(factory)


def test_configurable_ingestor_factory_str_contains_ingestor() -> None:
    factory = ConfigurableIngestorFactory(_make_ingestor())
    assert "ingestor" in str(factory)
