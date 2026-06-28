"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["print_markdown", "print_pretty"]

from typing import TYPE_CHECKING, Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

from zenpyre.utils.rich.console import get_console

if TYPE_CHECKING:
    from rich import Console


def print_markdown(
    msg: str,
    title: str | None = None,
    console: Console | None = None,
) -> None:
    r"""Render a Markdown string to the console inside a Rich panel.

    Prints ``msg`` as rendered Markdown wrapped in a
    :class:`~rich.panel.Panel`.  Uses the provided ``console`` if given,
    otherwise falls back to the shared instance (see :func:`set_console`).

    Args:
        msg: The Markdown content to render.  May contain any Markdown
            syntax supported by :class:`~rich.markdown.Markdown`.
            An empty string renders a blank panel.
        title: Optional title displayed in the panel border.  Pass
            ``None`` (the default) to render a borderless panel with no
            title.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_markdown
        >>> print_markdown("**hello**", title="Demo")

        ```
    """
    (console or get_console()).print(Panel(Markdown(msg), title=title))


def print_pretty(
    data: Any,
    title: str | None = None,
    console: Console | None = None,
) -> None:
    r"""Render an arbitrary object to the console in a pretty format.

    Prints ``data`` using :class:`~rich.pretty.Pretty` inside a
    :class:`~rich.panel.Panel`.  Uses the provided ``console`` if given,
    otherwise falls back to the shared instance (see :func:`set_console`).
    Works with any Python object — dicts, lists, dataclasses, Pydantic
    models, and so on.

    Args:
        data: The object to render.  Passed directly to
            :class:`~rich.pretty.Pretty`, which handles formatting.
        title: Optional title displayed in the panel border.  Pass
            ``None`` (the default) to render a borderless panel with no
            title.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_pretty
        >>> print_pretty({"key": "value"}, title="Demo")

        ```
    """
    (console or get_console()).print(Panel(Pretty(data), title=title))
