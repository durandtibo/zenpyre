"""Unit tests for zenpyre.utils.rich.console."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

from zenpyre.utils.rich import print_markdown, print_pretty

MODULE = "zenpyre.utils.rich.console"

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
    with patch(f"{MODULE}.console.print") as mock_print:
        print_markdown(msg, title=title)
    panel: Panel = mock_print.call_args.args[0]
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
    with patch(f"{MODULE}.console.print") as mock_print:
        print_pretty(data, title=title)
    panel: Panel = mock_print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Pretty)
    assert panel.title == title
