from __future__ import annotations

import pytest

from zenpyre.utils.rich import print_markdown, print_pretty

#####################################
#     Tests for print_markdown      #
#####################################


def test_print_markdown_returns_none() -> None:
    assert print_markdown("**hello**") is None


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
    print_markdown(msg, title=title)


@pytest.mark.parametrize("title_align", ["left", "center", "right"])
def test_print_markdown_title_align(title_align: str) -> None:
    print_markdown("**hello**", title="Demo", title_align=title_align)


@pytest.mark.parametrize("box", [True, False])
def test_print_markdown_box(box: bool) -> None:
    print_markdown("**hello**", box=box)


def test_print_markdown_max_length_no_truncation() -> None:
    assert print_markdown("**hello**", max_length=500) is None


def test_print_markdown_max_length_truncates() -> None:
    assert print_markdown("word " * 200, max_length=50) is None


def test_print_markdown_max_length_none_renders_full() -> None:
    assert print_markdown("word " * 200, max_length=None) is None


####################################
#     Tests for print_pretty       #
####################################


def test_print_pretty_returns_none() -> None:
    assert print_pretty({"key": "value"}) is None


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
    print_pretty(data, title=title)


@pytest.mark.parametrize("title_align", ["left", "center", "right"])
def test_print_pretty_title_align(title_align: str) -> None:
    print_pretty({"key": "value"}, title="Demo", title_align=title_align)


@pytest.mark.parametrize("box", [True, False])
def test_print_pretty_box(box: bool) -> None:
    print_pretty({"key": "value"}, box=box)
