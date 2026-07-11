r"""Rich-based pretty printing for metadata statistics reports."""

from __future__ import annotations

__all__ = ["print_metadata_stats_report"]

from typing import Any

from rich import get_console
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _format_pct(part: int, total: int) -> str:
    """Format a count as a percentage string of a total.

    Args:
        part: The count representing part of the whole.
        total: The total count to compute the percentage against.

    Returns:
        A string like ``"12 (40.0%)"``, or just ``str(part)`` if
        ``total`` is ``0`` (to avoid a division by zero).
    """
    if total == 0:
        return str(part)
    return f"{part} ({100 * part / total:.1f}%)"


def _format_value(value: Any) -> str:
    """Format a single sampled metadata value for display.

    Args:
        value: The value to format, as stored in a key's
            ``unique_values_sample`` list.

    Returns:
        ``"<empty string>"`` for ``""``, ``"<None>"`` for ``None``,
        otherwise ``str(value)``.
    """
    if value == "" and isinstance(value, str):
        return "<empty string>"
    if value is None:
        return "<None>"
    return str(value)


def print_metadata_stats_report(
    stats: dict[str, Any],
    *,
    title: str = "Metadata Report",
    console: Console | None = None,
) -> None:
    """Pretty-print a metadata statistics report using rich.

        Renders the dict returned by ``compute_doc_metadata_stats`` or
        ``compute_record_metadata_stats`` (they share the same shape) as a
        human-readable summary panel followed by a per-key table, including
        inline data-quality warnings (missing keys, mixed types,
        None/empty values, truncated value samples).

    Args:
            stats: The statistics dict to render, as returned by
                ``compute_doc_metadata_stats`` or
                ``compute_record_metadata_stats``.
            title: Title shown at the top of the summary panel. Defaults
                to ``"Metadata Report"``.
            console: A ``rich.console.Console`` instance to print to. If
                ``None``, the current default console (via
                ``rich.console.get_console``) is used.

    Example:
    ```pycon
    >>> from zenpyre.documents.stats import compute_doc_metadata_stats
    >>> from zenpyre.documents.stats_report import print_metadata_stats_report
    >>> stats = compute_doc_metadata_stats(docs)  # doctest: +SKIP
    >>> print_metadata_stats_report(stats)  # doctest: +SKIP

    ```
    """
    console = console or get_console()

    count = stats.get("count", 0)

    if count == 0:
        console.print(Panel(Text("No documents/records processed.", style="italic"), title=title))
        return

    missing_metadata = stats["missing_metadata_count"]

    summary_lines = [
        f"[bold]Total documents/records:[/bold] {count}",
        f"[bold]With no metadata:[/bold] "
        f"{_format_pct(missing_metadata, count)}"
        + (" [red]⚠[/red]" if missing_metadata > 0 else ""),
        f"[bold]Distinct keys seen:[/bold] {stats['distinct_keys_seen']}",
        f"[bold]Keys per doc/record:[/bold] "
        f"avg={stats['avg_keys']:.2f}, min={stats['min_keys']}, max={stats['max_keys']}",
    ]
    summary_panel = Panel(Text.from_markup("\n".join(summary_lines)), title=title, expand=False)

    per_key: dict[str, dict[str, Any]] = stats.get("per_key", {})

    if not per_key:
        console.print(Group(summary_panel))
        return

    table = Table(title="Metadata Keys", expand=False, show_lines=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Present", justify="right")
    table.add_column("Types")
    table.add_column("None/Empty", justify="right")
    table.add_column("Sample values")
    table.add_column("Flags")

    for key in sorted(per_key):
        info = per_key[key]

        present = info["present_in_docs"]
        missing = info["missing_in_docs"]
        value_types = info["value_types"]
        none_count = info["none_or_empty_count"]
        sample = info["unique_values_sample"]
        truncated = info["unique_values_sample_truncated"]

        sample_str = ", ".join(_format_value(v) for v in sample)
        if truncated:
            sample_str += ", …" if sample_str else "…"

        flags = []
        if missing > 0:
            flags.append("missing values")
        if len(value_types) > 1:
            flags.append("mixed types")
        if none_count > 0:
            flags.append("has None/empty")
        flags_str = f"[yellow]{', '.join(flags)}[/yellow]" if flags else "[green]ok[/green]"

        table.add_row(
            key,
            _format_pct(present, count),
            ", ".join(value_types),
            _format_pct(none_count, count),
            sample_str or "-",
            flags_str,
        )

    console.print(Group(summary_panel, table))
