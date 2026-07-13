from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.data_processors import FirstNProcessor
from zenpyre.data_processors.base import BaseProcessor
from zenpyre.data_processors.factory import (
    BaseProcessorFactory,
    ConfigurableProcessorFactory,
)

FIRST_N_PROCESSOR_TARGET = "zenpyre.data_processors.FirstNProcessor"


def _make_processor() -> FirstNProcessor:
    """Return a FirstNProcessor instance for testing."""
    return FirstNProcessor(n=5)


##################################################
#     Tests for ConfigurableProcessorFactory      #
##################################################


# --- Inheritance ---


def test_configurable_processor_factory_is_base_processor_factory() -> None:
    assert isinstance(ConfigurableProcessorFactory(_make_processor()), BaseProcessorFactory)


# --- make_processor from instance ---


def test_configurable_processor_factory_make_processor_returns_base_processor() -> None:
    factory = ConfigurableProcessorFactory(_make_processor())
    assert isinstance(factory.make_processor(), BaseProcessor)


def test_configurable_processor_factory_make_processor_returns_same_instance() -> None:
    processor = _make_processor()
    factory = ConfigurableProcessorFactory(processor)
    assert factory.make_processor() is processor


# --- make_processor from dict ---


def test_configurable_processor_factory_make_processor_from_dict_returns_base_processor() -> None:
    factory = ConfigurableProcessorFactory({"_target_": FIRST_N_PROCESSOR_TARGET, "n": 5})
    assert isinstance(factory.make_processor(), BaseProcessor)


def test_configurable_processor_factory_make_processor_from_dict_returns_correct_type() -> None:
    factory = ConfigurableProcessorFactory({"_target_": FIRST_N_PROCESSOR_TARGET, "n": 5})
    assert isinstance(factory.make_processor(), FirstNProcessor)


# --- _get_repr_kwargs ---


def test_configurable_processor_factory_get_repr_kwargs_instance() -> None:
    processor = _make_processor()
    factory = ConfigurableProcessorFactory(processor)
    assert objects_are_equal(factory._get_repr_kwargs(), {"processor": processor})


def test_configurable_processor_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": FIRST_N_PROCESSOR_TARGET, "n": 5}
    factory = ConfigurableProcessorFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"processor": config})


# --- __repr__ and __str__ ---


def test_configurable_processor_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableProcessorFactory(_make_processor())
    assert repr(factory).startswith("ConfigurableProcessorFactory(")


def test_configurable_processor_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableProcessorFactory(_make_processor())
    assert str(factory).startswith("ConfigurableProcessorFactory(")


def test_configurable_processor_factory_repr_contains_processor() -> None:
    factory = ConfigurableProcessorFactory(_make_processor())
    assert "processor" in repr(factory)


def test_configurable_processor_factory_str_contains_processor() -> None:
    factory = ConfigurableProcessorFactory(_make_processor())
    assert "processor" in str(factory)
