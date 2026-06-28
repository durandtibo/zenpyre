"""Provide a shared Rich console and console rendering utilities."""

from __future__ import annotations

__all__ = ["get_console", "set_console"]

from dataclasses import dataclass, field

from rich.console import Console

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
