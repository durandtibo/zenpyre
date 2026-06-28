"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["get_console", "print_markdown", "print_pretty", "set_console"]

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.pretty import Pretty

# Single shared Console instance so Rich detects terminal capabilities
# (colour support, width, TTY) once and output remains thread-safe.


# Define a mutable state to avoid using a global statement
@dataclass
class _State:
    console: Console = field(default_factory=Console)


_state = _State()


def get_console() -> Console:
    """Return the shared :class:`~rich.console.Console` instance.

    Returns:
        The current shared :class:`~rich.console.Console` instance used
        by :func:`print_markdown` and :func:`print_pretty`.

    Example:
        ```pycon
        >>> from zenpyre.utils.rich import get_console
        >>> from rich.console import Console
        >>> isinstance(get_console(), Console)
        True

        ```
    """
    return _state.console


def set_console(console: Console) -> None:
    """Replace the shared :class:`~rich.console.Console` instance.

    Replaces the module-level console used by :func:`print_markdown` and
    :func:`print_pretty` when no explicit ``console`` argument is passed.
    Call this once at application startup to apply a custom configuration
    (e.g. a different theme, width, or output file) globally.

    Args:
        console: The :class:`~rich.console.Console` instance to use as
            the new shared default.

    Example:
        ```pycon
        >>> from rich.console import Console
        >>> from zenpyre.utils.rich import set_console, get_console
        >>> set_console(Console(width=120))
        >>> get_console().width
        120

        ```
    """
    _state.console = console


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
