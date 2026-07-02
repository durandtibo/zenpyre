r"""Define a processor that applies another processor to each item in a
sequence."""

from __future__ import annotations

__all__ = ["SequenceProcessor"]

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

from coola.display import MultilineDisplayMixin

from zenpyre.data_processors.base import BaseProcessor
from zenpyre.utils.rich import make_progressbar

U = TypeVar("U")
T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


class SequenceProcessor(BaseProcessor[Sequence[U], list[T]], MultilineDisplayMixin):
    """Processor that applies another processor to each item in a
    sequence and returns the list of results.

    Useful for applying any :class:`~zenpyre.data_processors.base.BaseProcessor`
    to every element of a sequence without writing a dedicated class.
    Unlike :class:`~zenpyre.data_processors.LambdaProcessor`, which
    applies ``processor`` to the whole input at once, this class maps
    ``processor`` over each item individually.

    Args:
        processor: A :class:`~zenpyre.data_processors.base.BaseProcessor`
            that accepts a single item of type ``U`` and returns a
            value of type ``T``.
        progress_description: The description shown on the progress bar
            while processing.  Defaults to ``"Processing items..."``.
        raise_on_error: If ``True`` (default), an exception raised while
            processing an item is propagated and processing stops
            immediately. If ``False``, items that fail are logged and
            skipped, and processing continues with the remaining items.

    Example:
        ```pycon
        >>> from zenpyre.data_processors import LambdaProcessor, SequenceProcessor
        >>> processor = SequenceProcessor(processor=LambdaProcessor(fn=str.upper))
        >>> processor.process(["hello", "world"])
        ['HELLO', 'WORLD']
        >>> processor = SequenceProcessor(processor=LambdaProcessor(fn=len))
        >>> processor.process(["a", "bb", "ccc"])
        [1, 2, 3]

        ```
    """

    def __init__(
        self,
        processor: BaseProcessor[U, T],
        progress_description: str = "Processing items...",
        raise_on_error: bool = True,
    ) -> None:
        self._processor = processor
        self._progress_description = progress_description
        self._raise_on_error = raise_on_error

    def process(self, data: Sequence[U]) -> list[T]:
        """Apply ``processor`` to each item in ``data`` and return the
        results.

        Args:
            data: The sequence of items to process.

        Returns:
            A list of results in the same order as the successfully
            processed items in ``data``. If ``raise_on_error`` is
            ``False``, items that raise an exception are skipped and
            omitted from the returned list.
        """
        results = []
        n_errors = 0

        with make_progressbar() as progress:
            task = progress.add_task(self._progress_description, total=len(data))
            for item in data:
                success, result = self._process_item(item)
                if success:
                    results.append(result)
                else:
                    n_errors += 1
                progress.advance(task)

        if n_errors:
            logger.warning("Skipped %s item(s) that failed to process", f"{n_errors:,}")
        logger.info("Processed %s item(s)", f"{len(results):,}")
        return results

    def _process_item(self, item: U) -> tuple[bool, T | None]:
        """Process a single item, handling errors according to
        ``raise_on_error``.

        Args:
            item: The item to process.

        Returns:
            A ``(success, result)`` tuple. ``result`` is ``None`` when
            ``success`` is ``False``.
        """
        try:
            return True, self._processor.process(item)
        except Exception:
            if self._raise_on_error:
                raise
            logger.exception("Failed to process item: %r", item)
            return False, None

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {
            "processor": self._processor,
            "progress_description": self._progress_description,
            "raise_on_error": self._raise_on_error,
        }
