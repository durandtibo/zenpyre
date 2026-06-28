"""Unit tests for resolve_embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings

from zenpyre.embeddings import resolve_embeddings

if TYPE_CHECKING:
    import pytest

MODULE = "zenpyre.embeddings.resolve"

FAKE_EMBEDDINGS_TARGET = "langchain_core.embeddings.fake.FakeEmbeddings"


############################################
#     Tests for resolve_embeddings         #
############################################


# --- Pass-through ---


def test_resolve_embeddings_returns_embeddings_instance() -> None:
    assert isinstance(resolve_embeddings(FakeEmbeddings(size=128)), Embeddings)


def test_resolve_embeddings_passthrough_returns_same_instance() -> None:
    embeddings = FakeEmbeddings(size=128)
    assert resolve_embeddings(embeddings) is embeddings


# --- From dict ---


def test_resolve_embeddings_from_dict_returns_embeddings() -> None:
    embeddings = resolve_embeddings({"_target_": FAKE_EMBEDDINGS_TARGET, "size": 64})
    assert isinstance(embeddings, Embeddings)
    assert embeddings.size == 64


def test_resolve_embeddings_from_dict_returns_correct_type() -> None:
    embeddings = resolve_embeddings({"_target_": FAKE_EMBEDDINGS_TARGET, "size": 128})
    assert isinstance(embeddings, FakeEmbeddings)
    assert embeddings.size == 128


# --- Invalid input ---


def test_resolve_embeddings_invalid_type_returns_value() -> None:
    result = resolve_embeddings("not-an-embeddings")
    assert result == "not-an-embeddings"


def test_resolve_embeddings_invalid_type_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=MODULE):
        resolve_embeddings("not-an-embeddings")
    assert any("not an Embeddings instance" in m for m in caplog.messages)
