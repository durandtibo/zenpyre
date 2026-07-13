from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from zenpyre.document_stores import BaseDocumentStore, SQLiteDocumentStore
from zenpyre.document_stores.factory import (
    BaseDocumentStoreFactory,
    SQLiteDocumentStoreFactory,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_path(tmp_path: Path) -> Path:
    """Return a SQLite file path for testing."""
    return tmp_path / "documents.sqlite"


###################################################
#     Tests for SQLiteDocumentStoreFactory        #
###################################################


# --- Inheritance ---


def test_sqlite_document_store_factory_is_base_document_store_factory(tmp_path: Path) -> None:
    assert isinstance(SQLiteDocumentStoreFactory(_make_path(tmp_path)), BaseDocumentStoreFactory)


# --- make_document_store ---


def test_sqlite_document_store_factory_make_document_store_returns_base_document_store(
    tmp_path: Path,
) -> None:
    factory = SQLiteDocumentStoreFactory(_make_path(tmp_path))
    store = factory.make_document_store()
    try:
        assert isinstance(store, BaseDocumentStore)
    finally:
        store.close()


def test_sqlite_document_store_factory_make_document_store_returns_sqlite_document_store(
    tmp_path: Path,
) -> None:
    factory = SQLiteDocumentStoreFactory(_make_path(tmp_path))
    store = factory.make_document_store()
    try:
        assert isinstance(store, SQLiteDocumentStore)
    finally:
        store.close()


def test_sqlite_document_store_factory_creates_file(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = SQLiteDocumentStoreFactory(path)
    factory.make_document_store().close()
    assert path.exists()


def test_sqlite_document_store_factory_read_only_requires_existing_file(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    with pytest.raises(Exception):  # noqa: B017, PT011
        SQLiteDocumentStoreFactory(path, read_only=True).make_document_store()


def test_sqlite_document_store_factory_in_memory_path_left_unchanged() -> None:
    factory = SQLiteDocumentStoreFactory(":memory:")
    assert factory._path == ":memory:"
    factory.make_document_store().close()


# --- _get_repr_kwargs ---


def test_sqlite_document_store_factory_get_repr_kwargs_no_extra_kwargs(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = SQLiteDocumentStoreFactory(path)
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": path, "read_only": False})


def test_sqlite_document_store_factory_get_repr_kwargs_read_only(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = SQLiteDocumentStoreFactory(path, read_only=True)
    assert objects_are_equal(factory._get_repr_kwargs(), {"path": path, "read_only": True})


def test_sqlite_document_store_factory_get_repr_kwargs_with_extra_kwargs(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = SQLiteDocumentStoreFactory(path, timeout=5.0)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "read_only": False, "timeout": 5.0},
    )


# --- __repr__ and __str__ ---


def test_sqlite_document_store_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = SQLiteDocumentStoreFactory(_make_path(tmp_path))
    assert repr(factory).startswith("SQLiteDocumentStoreFactory(")


def test_sqlite_document_store_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = SQLiteDocumentStoreFactory(_make_path(tmp_path))
    assert str(factory).startswith("SQLiteDocumentStoreFactory(")


def test_sqlite_document_store_factory_repr_contains_path(tmp_path: Path) -> None:
    factory = SQLiteDocumentStoreFactory(_make_path(tmp_path))
    assert "path" in repr(factory)
