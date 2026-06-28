"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["print_markdown", "print_pretty"]

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

# Single shared Console instance so Rich detects terminal capabilities
# (colour support, width, TTY) once and output remains thread-safe.
_console = Console()


def print_markdown(msg: str, title: str | None = None) -> None:
    r"""Render a Markdown string to the console inside a Rich panel.

    Prints ``msg`` as rendered Markdown wrapped in a
    :class:`~rich.panel.Panel` using the shared
    :class:`~rich.console.Console` instance.

    Args:
        msg: The Markdown content to render.  Must be a non-empty string.
            May contain any Markdown syntax supported by
            :class:`~rich.markdown.Markdown`.
        title: Optional title displayed in the panel border.  Pass
            ``None`` (the default) to render a borderless panel with no
            title.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_markdown
        >>> print_markdown("**hello**", title="Demo")

        ```
    """
    _console.print(Panel(Markdown(msg), title=title))


def print_pretty(data: Any, title: str | None = None) -> None:
    r"""Render an arbitrary object to the console in a pretty format.

    Prints ``data`` using :class:`~rich.pretty.Pretty` inside a
    :class:`~rich.panel.Panel` via the shared
    :class:`~rich.console.Console` instance.  Works with any Python
    object — dicts, lists, dataclasses, Pydantic models, and so on.

    Args:
        data: The object to render.  Passed directly to
            :class:`~rich.pretty.Pretty`, which handles formatting.
        title: Optional title displayed in the panel border.  Pass
            ``None`` (the default) to render a borderless panel with no
            title.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_pretty
        >>> print_pretty({"key": "value"}, title="Demo")

        ```
    """
    _console.print(Panel(Pretty(data), title=title))
