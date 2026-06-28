"""Unit tests for resolve_text_splitter."""

from __future__ import annotations

import logging

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
        RuntimeError, match=r"'langchain_text_splitters' package is required but not installed."
    ):
        resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})


# --- Invalid input ---


@langchain_text_splitters_available
def test_resolve_text_splitter_invalid_type_returns_value() -> None:
    result = resolve_text_splitter("not-a-text-splitter")
    assert result == "not-a-text-splitter"


@langchain_text_splitters_available
def test_resolve_text_splitter_invalid_type_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolve_text_splitter("not-a-text-splitter")
    assert any("not a TextSplitter instance" in m for m in caplog.messages)
