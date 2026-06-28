"""Unit tests for resolve_text_splitter."""

from __future__ import annotations

import pytest

from zenpyre.testing.fixtures import (
    langchain_text_splitters_available,
    langchain_text_splitters_not_available,
)
from zenpyre.text_splitters import resolve_text_splitter
from zenpyre.utils.imports import is_langchain_text_splitters_available

if is_langchain_text_splitters_available():
    from langchain_text_splitters import CharacterTextSplitter, TextSplitter


CHARACTER_TEXT_SPLITTER_TARGET = "langchain_text_splitters.CharacterTextSplitter"

##############################################
#     Tests for resolve_text_splitter        #
##############################################


# --- Pass-through ---


@langchain_text_splitters_available
def test_resolve_text_splitter_returns_text_splitter_instance() -> None:
    assert isinstance(resolve_text_splitter(CharacterTextSplitter()), TextSplitter)


@langchain_text_splitters_available
def test_resolve_text_splitter_passthrough_returns_same_instance() -> None:
    splitter = CharacterTextSplitter()
    assert resolve_text_splitter(splitter) is splitter


# --- From dict ---


@langchain_text_splitters_available
def test_resolve_text_splitter_from_dict_returns_text_splitter() -> None:
    result = resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(result, TextSplitter)


@langchain_text_splitters_available
def test_resolve_text_splitter_from_dict_returns_correct_type() -> None:
    result = resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(result, CharacterTextSplitter)


@langchain_text_splitters_not_available
def test_resolve_text_splitter_without_langchain_text_splitters() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"The target object does not exist: langchain_text_splitters.CharacterTextSplitter",
    ):
        resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})


# --- Invalid input ---


@langchain_text_splitters_available
def test_resolve_embeddings_invalid_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Received object is not a TextSplitter instance"):
        resolve_text_splitter("not-a-text-splitter")
