"""Unit tests for zenpyre.display.rich.console."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

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
#     Tests for print_markdown      #
#####################################


def test_print_markdown() -> None:
    print_markdown("**hello**")


def test_print_markdown_with_title() -> None:
    print_markdown("**hello**", title="Demo")


def test_print_markdown_returns_none() -> None:
    assert print_markdown("**hello**") is None


def test_print_markdown_empty_string() -> None:
    print_markdown("")


def test_print_markdown_multiline() -> None:
    print_markdown("# Heading\n\nSome **bold** and _italic_ text.")


@pytest.mark.parametrize(
    ("msg", "title"),
    [
        pytest.param("**hello**", None, id="no-title"),
        pytest.param("**hello**", "Demo", id="with-title"),
        pytest.param("", None, id="empty-string"),
        pytest.param("# H1\n\nParagraph.", "Report", id="multiline-with-title"),
    ],
)
def test_print_markdown_renders_panel(msg: str, title: str | None) -> None:
    with patch(f"{MODULE}._state.console.print") as mock_print:
        print_markdown(msg, title=title)
    panel: Panel = mock_print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Markdown)
    assert panel.title == title


def test_print_markdown_uses_custom_console() -> None:
    custom = MagicMock(spec=Console)
    print_markdown("**hello**", console=custom)
    custom.print.assert_called_once()


def test_print_markdown_custom_console_not_shared() -> None:
    """A per-call console does not affect the shared instance."""
    custom = MagicMock(spec=Console)
    print_markdown("**hello**", console=custom)
    assert get_console() is not custom


@pytest.mark.parametrize(
    ("msg", "title"),
    [
        pytest.param("**hello**", None, id="no-title"),
        pytest.param("**hello**", "Demo", id="with-title"),
    ],
)
def test_print_markdown_custom_console_renders_panel(msg: str, title: str | None) -> None:
    custom = MagicMock(spec=Console)
    print_markdown(msg, title=title, console=custom)
    panel: Panel = custom.print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Markdown)
    assert panel.title == title


####################################
#     Tests for print_pretty       #
####################################


def test_print_pretty() -> None:
    print_pretty({"key": "value"})


def test_print_pretty_with_title() -> None:
    print_pretty({"key": "value"}, title="Demo")


def test_print_pretty_returns_none() -> None:
    assert print_pretty({"key": "value"}) is None


def test_print_pretty_empty_dict() -> None:
    print_pretty({})


@pytest.mark.parametrize(
    ("data", "title"),
    [
        pytest.param({"key": "value"}, None, id="dict-no-title"),
        pytest.param({"key": "value"}, "Demo", id="dict-with-title"),
        pytest.param([], None, id="empty-list"),
        pytest.param([1, 2, 3], "Numbers", id="list-with-title"),
        pytest.param("hello", None, id="string"),
        pytest.param(42, None, id="integer"),
        pytest.param(None, None, id="none"),
        pytest.param({}, "Empty", id="empty-dict-with-title"),
    ],
)
def test_print_pretty_renders_panel(data: object, title: str | None) -> None:
    with patch(f"{MODULE}._state.console.print") as mock_print:
        print_pretty(data, title=title)
    panel: Panel = mock_print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Pretty)
    assert panel.title == title


def test_print_pretty_uses_custom_console() -> None:
    custom = MagicMock(spec=Console)
    print_pretty({"key": "value"}, console=custom)
    custom.print.assert_called_once()


def test_print_pretty_custom_console_not_shared() -> None:
    """A per-call console does not affect the shared instance."""
    custom = MagicMock(spec=Console)
    print_pretty({"key": "value"}, console=custom)
    assert get_console() is not custom


@pytest.mark.parametrize(
    ("data", "title"),
    [
        pytest.param({"key": "value"}, None, id="no-title"),
        pytest.param({"key": "value"}, "Demo", id="with-title"),
    ],
)
def test_print_pretty_custom_console_renders_panel(data: object, title: str | None) -> None:
    custom = MagicMock(spec=Console)
    print_pretty(data, title=title, console=custom)
    panel: Panel = custom.print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Pretty)
    assert panel.title == title


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
