from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from coola.equality import objects_are_equal

from zenpyre.record_stores import BaseRecordStore, TypedSQLiteRecordStore
from zenpyre.record_stores.factory import (
    BaseRecordStoreFactory,
    TypedSQLiteRecordStoreFactory,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_path(tmp_path: Path) -> Path:
    """Return a SQLite file path for testing."""
    return tmp_path / "records.sqlite"


####################################################
#     Tests for TypedSQLiteRecordStoreFactory      #
####################################################


# --- Inheritance ---


def test_typed_sqlite_record_store_factory_is_base_record_store_factory(tmp_path: Path) -> None:
    assert isinstance(TypedSQLiteRecordStoreFactory(_make_path(tmp_path)), BaseRecordStoreFactory)


# --- make_record_store ---


def test_typed_sqlite_record_store_factory_make_record_store_returns_base_record_store(
    tmp_path: Path,
) -> None:
    factory = TypedSQLiteRecordStoreFactory(_make_path(tmp_path))
    store = factory.make_record_store()
    try:
        assert isinstance(store, BaseRecordStore)
    finally:
        store.close()


def test_typed_sqlite_record_store_factory_make_record_store_returns_typed_sqlite_record_store(
    tmp_path: Path,
) -> None:
    factory = TypedSQLiteRecordStoreFactory(_make_path(tmp_path))
    store = factory.make_record_store()
    try:
        assert isinstance(store, TypedSQLiteRecordStore)
    finally:
        store.close()


def test_typed_sqlite_record_store_factory_creates_file(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = TypedSQLiteRecordStoreFactory(path)
    factory.make_record_store().close()
    assert path.exists()


def test_typed_sqlite_record_store_factory_applies_metadata_schema(tmp_path: Path) -> None:
    factory = TypedSQLiteRecordStoreFactory(
        _make_path(tmp_path), metadata_schema={"year": "INTEGER"}
    )
    store = factory.make_record_store()
    try:
        columns = [row[1] for row in store._conn.execute("PRAGMA table_info(records)").fetchall()]
        assert "year" in columns
    finally:
        store.close()


def test_typed_sqlite_record_store_factory_read_only_requires_existing_file(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    with pytest.raises(Exception):  # noqa: B017, PT011
        TypedSQLiteRecordStoreFactory(path, read_only=True).make_record_store()


def test_typed_sqlite_record_store_factory_in_memory_path_left_unchanged() -> None:
    factory = TypedSQLiteRecordStoreFactory(":memory:")
    assert factory._path == ":memory:"
    factory.make_record_store().close()


# --- _get_repr_kwargs ---


def test_typed_sqlite_record_store_factory_get_repr_kwargs_no_extra_kwargs(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    factory = TypedSQLiteRecordStoreFactory(path)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "metadata_schema": None, "read_only": False},
    )


def test_typed_sqlite_record_store_factory_get_repr_kwargs_with_metadata_schema(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    metadata_schema = {"year": "INTEGER"}
    factory = TypedSQLiteRecordStoreFactory(path, metadata_schema=metadata_schema)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "metadata_schema": metadata_schema, "read_only": False},
    )


def test_typed_sqlite_record_store_factory_get_repr_kwargs_read_only(tmp_path: Path) -> None:
    path = _make_path(tmp_path)
    factory = TypedSQLiteRecordStoreFactory(path, read_only=True)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "metadata_schema": None, "read_only": True},
    )


def test_typed_sqlite_record_store_factory_get_repr_kwargs_with_extra_kwargs(
    tmp_path: Path,
) -> None:
    path = _make_path(tmp_path)
    factory = TypedSQLiteRecordStoreFactory(path, timeout=5.0)
    assert objects_are_equal(
        factory._get_repr_kwargs(),
        {"path": path, "metadata_schema": None, "read_only": False, "timeout": 5.0},
    )


# --- __repr__ and __str__ ---


def test_typed_sqlite_record_store_factory_repr_starts_with_class_name(tmp_path: Path) -> None:
    factory = TypedSQLiteRecordStoreFactory(_make_path(tmp_path))
    assert repr(factory).startswith("TypedSQLiteRecordStoreFactory(")


def test_typed_sqlite_record_store_factory_str_starts_with_class_name(tmp_path: Path) -> None:
    factory = TypedSQLiteRecordStoreFactory(_make_path(tmp_path))
    assert str(factory).startswith("TypedSQLiteRecordStoreFactory(")


def test_typed_sqlite_record_store_factory_repr_contains_path(tmp_path: Path) -> None:
    factory = TypedSQLiteRecordStoreFactory(_make_path(tmp_path))
    assert "path" in repr(factory)
