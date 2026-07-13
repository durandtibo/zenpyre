from __future__ import annotations

from typing import TYPE_CHECKING

from coola.equality import objects_are_equal

from zenpyre.document_stores import BaseDocumentStore, DuckDBDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    DuckDBDocumentStoreFactory,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_path(tmp_path: Path) -> Path:
    """Return a DuckDB file path for testing."""
    return tmp_path / "documents.duckdb"


##################################################
#     Tests for DuckDBDocumentStoreFactory       #
##################################################


# --- Inheritance ---


def test_duckdb_document_store_factory_is_base_document_store_factory(tmp_path: Path) -> None:
    assert isinstance(DuckDBDocumentStoreFactory(_make_path(tmp_path)), BaseDocumentStoreFactory)


# --- make_document_store ---


def test_duckdb_document_store_factory_make_document_store_returns_base_document_store(
    tmp_path: Path,
) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert isinstance(factory.make_document_store(), BaseDocumentStore)


def test_duckdb_document_store_factory_make_document_store_returns_duckdb_document_store(
    tmp_path: Path,
) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert isinstance(factory.make_document_store(), DuckDBDocumentStore)


def test_duckdb_document_store_factory_make_document_store_returns_new_instance_each_call(
    tmp_path: Path,
) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert factory.make_document_store() is not factory.make_document_store()


# --- _get_repr_kwargs ---


def test_duckdb_document_store_factory_get_repr_kwargs_no_extra_kwargs(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = DuckDBDocumentStoreFactory(path)
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": path})


def test_duckdb_document_store_factory_get_repr_kwargs_with_extra_kwargs(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = DuckDBDocumentStoreFactory(path, read_only=True)
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": path, "read_only": True})


# --- __repr__ and __str__ ---


def test_duckdb_document_store_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert repr(factory).startswith("DuckDBDocumentStoreFactory(")


def test_duckdb_document_store_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert str(factory).startswith("DuckDBDocumentStoreFactory(")


def test_duckdb_document_store_factory_repr_contains_path(tmp_path: Path) -> None:
    factory = DuckDBDocumentStoreFactory(_make_path(tmp_path))
    assert "path" in repr(factory)
