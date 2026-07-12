from __future__ import annotations

import pytest

from zenpyre.document_stores import (
    BaseDocumentStore,
    InMemoryDocumentStore,
    resolve_document_store,
)

IN_MEMORY_DOCUMENT_STORE_TARGET = "zenpyre.document_stores.InMemoryDocumentStore"


def _make_document_store() -> InMemoryDocumentStore:
    """Return an InMemoryDocumentStore instance for testing."""
    return InMemoryDocumentStore()


##################################################
#     Tests for resolve_document_store           #
##################################################


# --- Pass-through ---


def test_resolve_document_store_returns_base_document_store_instance() -> None:
    assert isinstance(resolve_document_store(_make_document_store()), BaseDocumentStore)


def test_resolve_document_store_passthrough_returns_same_instance() -> None:
    document_store = _make_document_store()
    assert resolve_document_store(document_store) is document_store


# --- From dict ---


def test_resolve_document_store_from_dict_returns_base_document_store() -> None:
    result = resolve_document_store({"_target_": IN_MEMORY_DOCUMENT_STORE_TARGET})
    assert isinstance(result, BaseDocumentStore)


def test_resolve_document_store_from_dict_returns_correct_type() -> None:
    result = resolve_document_store({"_target_": IN_MEMORY_DOCUMENT_STORE_TARGET})
    assert isinstance(result, InMemoryDocumentStore)


# --- Invalid input ---


def test_resolve_document_store_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseDocumentStore instance"):
        resolve_document_store("not-a-document-store")
