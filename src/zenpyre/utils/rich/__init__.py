r"""Common utilities for rich."""

from __future__ import annotations

__all__ = [
    "configure_rich_logging",
    "get_console",
    "make_progressbar",
    "make_spinner",
    "print_document",
    "print_markdown",
    "print_pretty",
    "set_console",
]

from zenpyre.utils.rich.console import get_console, set_console
from zenpyre.utils.rich.document import print_document
from zenpyre.utils.rich.logging import configure_rich_logging
from zenpyre.utils.rich.print_ import print_markdown, print_pretty
from zenpyre.utils.rich.progressbar import make_progressbar, make_spinner
