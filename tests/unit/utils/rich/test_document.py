from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from rich.console import Console, Group
from rich.panel import Panel

from zenpyre.utils.rich import print_document
from zenpyre.utils.rich.document import _truncate

MODULE = "zenpyre.utils.rich.document"


def _make_doc(content: str = "Hello world", metadata: dict | None = None) -> Document:
    return Document(id="1", page_content=content, metadata=metadata or {})


####################################
#     Tests for _truncate          #
####################################


def test_truncate_shorter_than_max_length_returns_unchanged() -> None:
    assert _truncate("Hello world", 100) == ("Hello world", False)


def test_truncate_exact_max_length_returns_unchanged() -> None:
    content = "a" * 50
    assert _truncate(content, 50) == (content, False)


def test_truncate_longer_than_max_length_is_truncated() -> None:
    content, truncated = _truncate("Hello wonderful world", 10)
    assert truncated is True
    assert len(content) <= 10


def test_truncate_breaks_on_word_boundary() -> None:
    content, truncated = _truncate("Hello wonderful world", 12)
    assert truncated is True
    assert not content.endswith(" ")
    assert "wonderfu" not in content  # word wasn't cut mid-way when a boundary was close


def test_truncate_empty_string() -> None:
    assert _truncate("", 10) == ("", False)


def test_truncate_snaps_to_word_boundary_when_close_to_limit() -> None:
    """last_space (15) is within 80% of max_length (18), so the cut
    should snap back to the word boundary, exercising the `cut =
    cut[:last_space]` branch."""
    content, truncated = _truncate("Hello wonderful world", 18)
    assert content == "Hello wonderful"
    assert truncated is True


def test_truncate_does_not_snap_when_word_boundary_too_far_back() -> None:
    """last_space (5) is below 80% of max_length (12), so the cut is
    kept as-is without snapping back to the word boundary."""
    content, truncated = _truncate("Hello wonderful world", 12)
    assert content == "Hello wonder"
    assert truncated is True


####################################
#     Tests for print_document     #
####################################


def test_print_document() -> None:
    print_document(_make_doc())


def test_print_document_with_metadata() -> None:
    print_document(_make_doc(metadata={"author": "Alice"}))


def test_print_document_returns_none() -> None:
    assert print_document(_make_doc()) is None


def test_print_document_empty_content() -> None:
    print_document(_make_doc(content=""))


def test_print_document_compact_metadata() -> None:
    print_document(_make_doc(metadata={"author": "Alice", "year": 2024}), compact_metadata=True)


@pytest.mark.parametrize(
    ("content", "metadata"),
    [
        pytest.param("Hello world", {}, id="no-metadata"),
        pytest.param("Hello world", {"author": "Alice"}, id="with-metadata"),
        pytest.param("", {}, id="empty-content"),
        pytest.param("x" * 1000, {"category": "Programming"}, id="long-content-truncated"),
    ],
)
def test_print_document_renders_panel(content: str, metadata: dict) -> None:
    with patch(f"{MODULE}.get_console") as mock_get_console:
        mock_console = MagicMock(spec=Console)
        mock_get_console.return_value = mock_console
        print_document(_make_doc(content=content, metadata=metadata), max_length=200)
    panel: Panel = mock_console.print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)


def test_print_document_uses_custom_console() -> None:
    custom = MagicMock(spec=Console)
    print_document(_make_doc(), console=custom)
    custom.print.assert_called_once()


def test_print_document_custom_console_not_shared() -> None:
    """A per-call console does not affect the shared instance."""
    custom = MagicMock(spec=Console)
    with patch(f"{MODULE}.get_console") as mock_get_console:
        mock_get_console.return_value = MagicMock(spec=Console)
        print_document(_make_doc(), console=custom)
        assert mock_get_console.return_value is not custom


@pytest.mark.parametrize(
    ("metadata", "compact_metadata"),
    [
        pytest.param({"author": "Alice"}, False, id="table-mode"),
        pytest.param({"author": "Alice"}, True, id="compact-mode"),
        pytest.param({}, False, id="no-metadata-table-mode"),
        pytest.param({}, True, id="no-metadata-compact-mode"),
    ],
)
def test_print_document_custom_console_renders_panel(
    metadata: dict, compact_metadata: bool
) -> None:
    custom = MagicMock(spec=Console)
    print_document(_make_doc(metadata=metadata), console=custom, compact_metadata=compact_metadata)
    panel: Panel = custom.print.call_args.args[0]
    assert isinstance(panel, Panel)
    assert isinstance(panel.renderable, Group)


def test_print_document_title_contains_doc_id() -> None:
    custom = MagicMock(spec=Console)
    print_document(_make_doc(), console=custom)
    panel: Panel = custom.print.call_args.args[0]
    assert "1" in panel.title


def test_print_document_no_id_title_falls_back() -> None:
    custom = MagicMock(spec=Console)
    doc = Document(page_content="Hello", metadata={})
    print_document(doc, console=custom)
    panel: Panel = custom.print.call_args.args[0]
    assert "Document" in panel.title
