from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal
from langchain_core.document_loaders import BaseLoader

from zenpyre.document_loaders.factory import BaseLoaderFactory, LoaderFactory

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.documents import Document


class MinimalLoader(BaseLoader):
    """Minimal concrete BaseLoader for testing."""

    def lazy_load(self) -> Iterator[Document]:
        return iter([])


###################################
#     Tests for LoaderFactory     #
###################################


# --- Inheritance ---


def test_loader_factory_is_base_loader_factory() -> None:
    assert isinstance(LoaderFactory(MinimalLoader()), BaseLoaderFactory)


# --- make_loader ---


def test_loader_factory_make_loader_returns_base_loader() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert isinstance(factory.make_loader(), BaseLoader)


def test_loader_factory_make_loader_returns_same_instance() -> None:
    loader = MinimalLoader()
    factory = LoaderFactory(loader)
    assert factory.make_loader() is loader


def test_loader_factory_make_loader_returns_same_instance_on_repeated_calls() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert factory.make_loader() is factory.make_loader()


def test_loader_factory_different_instances_independent() -> None:
    factory_a = LoaderFactory(MinimalLoader())
    factory_b = LoaderFactory(MinimalLoader())
    assert factory_a.make_loader() is not factory_b.make_loader()


# --- _get_repr_kwargs ---


def test_loader_factory_get_repr_kwargs() -> None:
    loader = MinimalLoader()
    factory = LoaderFactory(loader)
    assert objects_are_equal(factory._get_repr_kwargs(), {"loader": loader})


# --- __repr__ and __str__ ---


def test_loader_factory_repr_starts_with_class_name() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert repr(factory).startswith("LoaderFactory(")


def test_loader_factory_str_starts_with_class_name() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert str(factory).startswith("LoaderFactory(")


def test_loader_factory_repr_contains_loader() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert "loader" in repr(factory)


def test_loader_factory_str_contains_loader() -> None:
    factory = LoaderFactory(MinimalLoader())
    assert "loader" in str(factory)
