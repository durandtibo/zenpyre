"""Unit tests for zenpyre.display.rich.console."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from zenpyre.utils.rich import get_console, print_markdown, print_pretty, set_console

if TYPE_CHECKING:
    from collections.abc import Generator

MODULE = "zenpyre.utils.rich.console"


@pytest.fixture(autouse=True)
def reset_console() -> Generator[None]:
    """Restore the shared console instance after each test."""
    yield
    set_console(Console())


#####################################
#     Tests for get_console         #
#####################################


def test_get_console_returns_console() -> None:
    assert isinstance(get_console(), Console)


def test_get_console_returns_same_instance() -> None:
    assert get_console() is get_console()


#####################################
#     Tests for set_console         #
#####################################


def test_set_console() -> None:
    custom = Console()
    set_console(custom)
    assert get_console() is custom


def test_set_console_returns_none() -> None:
    assert set_console(Console()) is None


def test_set_console_affects_print_markdown() -> None:
    custom = MagicMock(spec=Console)
    set_console(custom)
    print_markdown("**hello**")
    custom.print.assert_called_once()


def test_set_console_affects_print_pretty() -> None:
    custom = MagicMock(spec=Console)
    set_console(custom)
    print_pretty({"key": "value"})
    custom.print.assert_called_once()


def test_set_console_overridden_by_per_call_console() -> None:
    """A per-call console takes priority over the shared instance."""
    shared = MagicMock(spec=Console)
    per_call = MagicMock(spec=Console)
    set_console(shared)
    print_markdown("**hello**", console=per_call)
    shared.print.assert_not_called()
    per_call.print.assert_called_once()
