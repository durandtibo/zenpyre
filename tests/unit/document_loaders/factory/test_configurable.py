from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal
from langchain_core.document_loaders import BaseLoader

from zenpyre.document_loaders.factory import (
    BaseLoaderFactory,
    ConfigurableLoaderFactory,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.documents import Document

MINIMAL_LOADER_TARGET = "tests.unit.document_loaders.factory.test_configurable.MinimalLoader"


class MinimalLoader(BaseLoader):
    """Minimal concrete BaseLoader for testing."""

    def lazy_load(self) -> Iterator[Document]:
        return iter([])


def _make_loader() -> MinimalLoader:
    """Return a MinimalLoader instance for testing."""
    return MinimalLoader()


##############################################
#     Tests for ConfigurableLoaderFactory    #
##############################################


# --- Inheritance ---


def test_configurable_loader_factory_is_base_loader_factory() -> None:
    assert isinstance(ConfigurableLoaderFactory(_make_loader()), BaseLoaderFactory)


# --- make_loader from instance ---


def test_configurable_loader_factory_make_loader_returns_base_loader() -> None:
    factory = ConfigurableLoaderFactory(_make_loader())
    assert isinstance(factory.make_loader(), BaseLoader)


def test_configurable_loader_factory_make_loader_returns_same_instance() -> None:
    loader = _make_loader()
    factory = ConfigurableLoaderFactory(loader)
    assert factory.make_loader() is loader


# --- make_loader from dict ---


def test_configurable_loader_factory_make_loader_from_dict_returns_base_loader() -> None:
    factory = ConfigurableLoaderFactory({"_target_": MINIMAL_LOADER_TARGET})
    assert isinstance(factory.make_loader(), BaseLoader)


def test_configurable_loader_factory_make_loader_from_dict_returns_correct_type() -> None:
    factory = ConfigurableLoaderFactory({"_target_": MINIMAL_LOADER_TARGET})
    assert isinstance(factory.make_loader(), MinimalLoader)


# --- _get_repr_kwargs ---


def test_configurable_loader_factory_get_repr_kwargs_instance() -> None:
    loader = _make_loader()
    factory = ConfigurableLoaderFactory(loader)
    assert objects_are_equal(factory._get_repr_kwargs(), {"loader": loader})


def test_configurable_loader_factory_get_repr_kwargs_dict_input() -> None:
    config = {"_target_": MINIMAL_LOADER_TARGET}
    factory = ConfigurableLoaderFactory(config)
    assert objects_are_equal(factory._get_repr_kwargs(), {"loader": config})


# --- __repr__ and __str__ ---


def test_configurable_loader_factory_repr_starts_with_class_name() -> None:
    factory = ConfigurableLoaderFactory(_make_loader())
    assert repr(factory).startswith("ConfigurableLoaderFactory(")


def test_configurable_loader_factory_str_starts_with_class_name() -> None:
    factory = ConfigurableLoaderFactory(_make_loader())
    assert str(factory).startswith("ConfigurableLoaderFactory(")


def test_configurable_loader_factory_repr_contains_loader() -> None:
    factory = ConfigurableLoaderFactory(_make_loader())
    assert "loader" in repr(factory)


def test_configurable_loader_factory_str_contains_loader() -> None:
    factory = ConfigurableLoaderFactory(_make_loader())
    assert "loader" in str(factory)
