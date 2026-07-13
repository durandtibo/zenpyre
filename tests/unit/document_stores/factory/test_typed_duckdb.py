from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal

from zenpyre.document_stores import BaseDocumentStore, TypedDuckDBDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    TypedDuckDBDocumentStoreFactory,
)
from zenpyre.testing.fixtures import duckdb_available

if TYPE_CHECKING:
    from pathlib import Path


def _make_path(tmp_path: Path) -> Path:
    """Return a DuckDB file path for testing."""
    return tmp_path / "documents.duckdb"


######################################################
#     Tests for TypedDuckDBDocumentStoreFactory      #
######################################################


# --- Inheritance ---


@duckdb_available
def test_typed_duckdb_document_store_factory_is_base_document_store_factory(
    tmp_path: Path,
) -> None:
    assert isinstance(
        TypedDuckDBDocumentStoreFactory(_make_path(tmp_path)), BaseDocumentStoreFactory
    )


# --- make_document_store ---


@duckdb_available
def test_typed_duckdb_document_store_factory_make_document_store_returns_base_document_store(
    tmp_path: Path,
) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


@duckdb_available
def test_typed_duckdb_document_store_factory_make_document_store_returns_typed_duckdb_document_store(
    tmp_path: Path,
) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert isinstance(factory.make_document_store(), TypedDuckDBDocumentStore)


@duckdb_available
def test_typed_duckdb_document_store_factory_make_document_store_returns_new_instance_each_call(
    tmp_path: Path,
) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert factory.make_document_store() is not factory.make_document_store()


@duckdb_available
def test_typed_duckdb_document_store_factory_make_document_store_applies_metadata_schema(
    tmp_path: Path,
) -> None:
    factory = TypedDuckDBDocumentStoreFactory(
        _make_path(tmp_path), metadata_schema={"year": "INTEGER"}
    )
    store = factory.make_document_store()
    assert "year" in store.get_columns_info()


@duckdb_available
def test_typed_duckdb_document_store_factory_in_memory_path_left_unchanged() -> None:
    # Regression test: ":memory:" must not be resolved into a real file
    # path (e.g. relative to the cwd), which would silently create a
    # file named ":memory:" on disk instead of an in-memory database.
    factory = TypedDuckDBDocumentStoreFactory(":memory:")
    assert factory._path == ":memory:"
    factory.make_document_store().close()


# --- _get_repr_kwargs ---


@duckdb_available
def test_typed_duckdb_document_store_factory_get_repr_kwargs_no_extra_kwargs(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    factory = TypedDuckDBDocumentStoreFactory(path)
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": path, "metadata_schema": None})


@duckdb_available
def test_typed_duckdb_document_store_factory_get_repr_kwargs_with_metadata_schema(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    metadata_schema = {"year": "INTEGER"}
    factory = TypedDuckDBDocumentStoreFactory(path, metadata_schema=metadata_schema)
    assert objects_are_equal(
        factory._get_repr_kwargs(), {"path": path, "metadata_schema": metadata_schema}
    )


@duckdb_available
def test_typed_duckdb_document_store_factory_get_repr_kwargs_with_extra_kwargs(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    factory = TypedDuckDBDocumentStoreFactory(path, read_only=True)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "metadata_schema": None, "read_only": True},
    )


# --- __repr__ and __str__ ---


@duckdb_available
def test_typed_duckdb_document_store_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert repr(factory).startswith("TypedDuckDBDocumentStoreFactory(")


@duckdb_available
def test_typed_duckdb_document_store_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert str(factory).startswith("TypedDuckDBDocumentStoreFactory(")


@duckdb_available
def test_typed_duckdb_document_store_factory_repr_contains_path(tmp_path: Path) -> None:
    factory = TypedDuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert "path" in repr(factory)
