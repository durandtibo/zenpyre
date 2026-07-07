from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.document_loaders import BaseLoader

from zenpyre.document_loaders import resolve_document_loader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.documents import Document


MINIMAL_LOADER_TARGET = "tests.unit.document_loaders.test_resolve.MinimalLoader"


class MinimalLoader(BaseLoader):
    """Minimal concrete BaseLoader for testing."""

    def lazy_load(self) -> Iterator[Document]:
        return iter([])


##############################################
#     Tests for resolve_document_loader      #
##############################################


# --- Pass-through ---


def test_resolve_document_loader_returns_base_loader_instance() -> None:
    assert isinstance(resolve_document_loader(MinimalLoader()), BaseLoader)


def test_resolve_document_loader_passthrough_returns_same_instance() -> None:
    loader = MinimalLoader()
    assert resolve_document_loader(loader) is loader


# --- From dict ---


def test_resolve_document_loader_from_dict_returns_base_loader() -> None:
    result = resolve_document_loader({"_target_": MINIMAL_LOADER_TARGET})
    assert isinstance(result, BaseLoader)


def test_resolve_document_loader_from_dict_returns_correct_type() -> None:
    result = resolve_document_loader({"_target_": MINIMAL_LOADER_TARGET})
    assert isinstance(result, MinimalLoader)


# --- Invalid input ---


def test_resolve_document_loader_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match=r"Received object is not a BaseLoader instance"):
        resolve_document_loader("not-a-loader")
