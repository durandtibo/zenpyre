from __future__ import annotations

import pytest

from zenpyre.record_stores import (
    BaseRecordStore,
    InMemoryRecordStore,
    resolve_record_store,
)

IN_MEMORY_RECORD_STORE_TARGET = "zenpyre.record_stores.InMemoryRecordStore"


def _make_record_store() -> InMemoryRecordStore:
    """Return an InMemoryRecordStore instance for testing."""
    return InMemoryRecordStore()


##########################################
#     Tests for resolve_record_store     #
##########################################


# --- Pass-through ---


def test_resolve_record_store_returns_base_record_store_instance() -> None:
    assert isinstance(resolve_record_store(_make_record_store()), BaseRecordStore)


def test_resolve_record_store_passthrough_returns_same_instance() -> None:
    record_store = _make_record_store()
    assert resolve_record_store(record_store) is record_store


# --- From dict ---


def test_resolve_record_store_from_dict_returns_base_record_store() -> None:
    result = resolve_record_store({"_target_": IN_MEMORY_RECORD_STORE_TARGET})
    assert isinstance(result, BaseRecordStore)


def test_resolve_record_store_from_dict_returns_correct_type() -> None:
    result = resolve_record_store({"_target_": IN_MEMORY_RECORD_STORE_TARGET})
    assert isinstance(result, InMemoryRecordStore)


# --- Invalid input ---


def test_resolve_record_store_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseRecordStore instance"):
        resolve_record_store("not-a-record-store")
