"""Unit tests for resolve_text_splitter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_text_splitters import CharacterTextSplitter, TextSplitter

from zenpyre.text_splitters import resolve_text_splitter

if TYPE_CHECKING:
    import pytest

CHARACTER_TEXT_SPLITTER_TARGET = "langchain_text_splitters.CharacterTextSplitter"


##############################################
#     Tests for resolve_text_splitter        #
##############################################


# --- Pass-through ---


def test_resolve_text_splitter_returns_text_splitter_instance() -> None:
    assert isinstance(resolve_text_splitter(CharacterTextSplitter()), TextSplitter)


def test_resolve_text_splitter_passthrough_returns_same_instance() -> None:
    splitter = CharacterTextSplitter()
    assert resolve_text_splitter(splitter) is splitter


# --- From dict ---


def test_resolve_text_splitter_from_dict_returns_text_splitter() -> None:
    result = resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(result, TextSplitter)


def test_resolve_text_splitter_from_dict_returns_correct_type() -> None:
    result = resolve_text_splitter({"_target_": CHARACTER_TEXT_SPLITTER_TARGET})
    assert isinstance(result, CharacterTextSplitter)


# --- Invalid input ---


def test_resolve_text_splitter_invalid_type_returns_value() -> None:
    result = resolve_text_splitter("not-a-text-splitter")  # type: ignore[arg-type]
    assert result == "not-a-text-splitter"


def test_resolve_text_splitter_invalid_type_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        resolve_text_splitter("not-a-text-splitter")  # type: ignore[arg-type]
    assert any("not a TextSplitter instance" in m for m in caplog.messages)
