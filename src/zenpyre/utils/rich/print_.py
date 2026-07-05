"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["print_markdown", "print_pretty"]

from typing import TYPE_CHECKING, Any

from rich import get_console
from rich.box import MINIMAL, ROUNDED
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

if TYPE_CHECKING:
    from rich.align import AlignMethod
    from rich.console import Console


def _print_in_panel(
    renderable: Any,
    *,
    title: str | None,
    title_align: AlignMethod,
    box: bool,
    console: Console | None,
) -> None:
    r"""Render a Rich renderable to the console inside a panel.

    Shared implementation for :func:`print_markdown` and
    :func:`print_pretty`, factored out to keep their panel-construction
    logic (border style, title alignment, console fallback) in one
    place.

    Args:
        renderable: The Rich renderable to wrap in a panel (e.g. a
            :class:`~rich.markdown.Markdown` or
            :class:`~rich.pretty.Pretty` instance).
        title: Optional title displayed on the panel border.  Pass
            ``None`` to render a panel with no title.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).
        box: If ``True``, render the panel with a visible rounded
            border.  If ``False``, render with a minimal
            (near-invisible) border.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Pass ``None`` to use the shared instance.
    """
    (console or get_console()).print(
        Panel(renderable, title=title, title_align=title_align, box=ROUNDED if box else MINIMAL)
    )


def print_markdown(
    msg: str,
    *,
    title: str | None = None,
    title_align: AlignMethod = "left",
    box: bool = True,
    console: Console | None = None,
) -> None:
    r"""Render a Markdown string to the console inside a Rich panel.

    Prints ``msg`` as rendered Markdown wrapped in a
    :class:`~rich.panel.Panel`.  Uses the provided ``console`` if given,
    otherwise falls back to the shared instance.

    Args:
        msg: The Markdown content to render.  May contain any Markdown
            syntax supported by :class:`~rich.markdown.Markdown`.
            An empty string renders a blank panel.
        title: Optional title displayed on the panel border.  Pass
            ``None`` (the default) to render a panel with no title.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).  Defaults
            to ``"left"``.
        box: If ``True`` (the default), render the panel with a visible
            rounded border.  If ``False``, render with a minimal
            (near-invisible) border.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_markdown
        >>> print_markdown("**hello**", title="Demo")
        >>> print_markdown("**hello**", title="Demo", box=False)

        ```
    """
    _print_in_panel(Markdown(msg), title=title, title_align=title_align, box=box, console=console)


def print_pretty(
    data: Any,
    *,
    title: str | None = None,
    title_align: AlignMethod = "left",
    box: bool = True,
    console: Console | None = None,
) -> None:
    r"""Render an arbitrary object to the console in a pretty format.

    Prints ``data`` using :class:`~rich.pretty.Pretty` inside a
    :class:`~rich.panel.Panel`.  Uses the provided ``console`` if given,
    otherwise falls back to the shared instance.  Works with any Python
    object — dicts, lists, dataclasses, Pydantic models, and so on.

    Args:
        data: The object to render.  Passed directly to
            :class:`~rich.pretty.Pretty`, which handles formatting.
        title: Optional title displayed on the panel border.  Pass
            ``None`` (the default) to render a panel with no title.
        title_align: Horizontal alignment of ``title`` along the panel
            border (``"left"``, ``"center"``, or ``"right"``).  Defaults
            to ``"left"``.
        box: If ``True`` (the default), render the panel with a visible
            rounded border.  If ``False``, render with a minimal
            (near-invisible) border.
        console: Optional :class:`~rich.console.Console` instance to use
            for this call.  Overrides the shared default for this
            invocation only.  Pass ``None`` (the default) to use the
            shared instance.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import print_pretty
        >>> print_pretty({"key": "value"}, title="Demo")
        >>> print_pretty({"key": "value"}, title="Demo", box=False)

        ```
    """
    _print_in_panel(Pretty(data), title=title, title_align=title_align, box=box, console=console)
