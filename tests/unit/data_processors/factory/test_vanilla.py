from __future__ import annotations

from coola.equality import objects_are_equal

from zenpyre.data_processors import BaseProcessor, FirstNProcessor
from zenpyre.data_processors.factory import BaseProcessorFactory, ProcessorFactory


def _make_processor() -> FirstNProcessor:
    """Return a FirstNProcessor instance for testing."""
    return FirstNProcessor(n=5)


##############################################
#     Tests for ProcessorFactory             #
##############################################


# --- Inheritance ---


def test_processor_factory_is_base_processor_factory() -> None:
    assert isinstance(ProcessorFactory(_make_processor()), BaseProcessorFactory)


# --- make_processor ---


def test_processor_factory_make_processor_returns_base_processor() -> None:
    factory = ProcessorFactory(_make_processor())
    assert isinstance(factory.make_processor(), BaseProcessor)


def test_processor_factory_make_processor_returns_same_instance() -> None:
    processor = _make_processor()
    factory = ProcessorFactory(processor)
    assert factory.make_processor() is processor


def test_processor_factory_make_processor_returns_same_instance_across_calls() -> None:
    processor = _make_processor()
    factory = ProcessorFactory(processor)
    assert factory.make_processor() is factory.make_processor()


# --- _get_repr_kwargs ---


def test_processor_factory_get_repr_kwargs() -> None:
    processor = _make_processor()
    factory = ProcessorFactory(processor)
    assert objects_are_equal(factory._get_repr_kwargs(), {"processor": processor})


# --- __repr__ and __str__ ---


def test_processor_factory_repr_starts_with_class_name() -> None:
    factory = ProcessorFactory(_make_processor())
    assert repr(factory).startswith("ProcessorFactory(")


def test_processor_factory_str_starts_with_class_name() -> None:
    factory = ProcessorFactory(_make_processor())
    assert str(factory).startswith("ProcessorFactory(")


def test_processor_factory_repr_contains_processor() -> None:
    factory = ProcessorFactory(_make_processor())
    assert "processor" in repr(factory)


def test_processor_factory_str_contains_processor() -> None:
    factory = ProcessorFactory(_make_processor())
    assert "processor" in str(factory)
