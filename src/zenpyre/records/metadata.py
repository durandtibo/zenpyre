from __future__ import annotations

__all__ = ["analyze_record_metadata"]

from collections import defaultdict
from collections.abc import Iterable

from zenpyre.records import Record


def analyze_record_metadata(records: Iterable[Record], n_sample_values: int = 3) -> dict:
    """Analyze metadata across a (possibly huge) stream of LangChain Records.

    Designed to work when records cannot all fit into memory: this does a
    single pass over ``records`` (an iterable/generator) and only keeps
    small, bounded per-key aggregates - no list of all records, no full
    sets of unique values.

    Args:
        records: Iterable of LangChain ``Record`` objects (anything with
            a ``.metadata`` dict). Can be a generator so the full collection
            never needs to be materialized in memory.
        n_sample_values: Max number of unique sample values to keep per key.
            Defaults to 3.

    Returns:
        dict: Raw metadata aggregates with the following structure:

            * ``total_records`` (int): Number of records processed.
            * ``records_with_no_metadata`` (int): Count of records whose
              ``metadata`` dict is empty or missing.
            * ``keys_per_record`` (dict): ``avg``, ``min``, ``max`` number
              of metadata keys per record.
            * ``distinct_keys_seen`` (int): Number of distinct metadata keys
              observed across all records.
            * ``per_key`` (dict): Mapping of key name to its stats
              (``present_in_docs``, ``missing_in_docs``, ``value_types``,
              ``none_or_empty_count``, ``unique_values_sample``,
              ``unique_values_sample_truncated``).

        Percentages are intentionally omitted - compute them later from the
        raw counts (e.g. ``key_count / total_records``) since
        ``total_records`` is only known once the stream is exhausted.
    """
    total_docs = 0
    docs_no_metadata = 0

    key_counts = defaultdict(int)          # key -> number of docs containing it
    key_values = defaultdict(set)          # key -> bounded sample of unique values
    key_value_truncated = defaultdict(bool)  # key -> whether sampling was capped/skipped
    key_types = defaultdict(set)           # key -> set of value types seen
    key_none_count = defaultdict(int)      # key -> number of docs where value is None/empty string

    n_keys_sum = 0
    n_keys_min = None
    n_keys_max = None

    for doc in records:
        total_docs += 1

        metadata = doc.metadata or {}

        if not metadata:
            docs_no_metadata += 1

        n_keys = len(metadata)
        n_keys_sum += n_keys
        n_keys_min = n_keys if n_keys_min is None else min(n_keys_min, n_keys)
        n_keys_max = n_keys if n_keys_max is None else max(n_keys_max, n_keys)

        for key, value in metadata.items():
            key_counts[key] += 1
            key_types[key].add(type(value).__name__)

            if value is None or value == "":
                key_none_count[key] += 1

            if not key_value_truncated[key]:
                try:
                    if isinstance(value, (list, dict, set)):
                        # unhashable/complex -> stop trying to sample this key
                        key_value_truncated[key] = True
                    else:
                        key_values[key].add(value)
                        if len(key_values[key]) > n_sample_values:
                            # drop back down to the cap and mark as truncated
                            key_values[key] = set(list(key_values[key])[:n_sample_values])
                            key_value_truncated[key] = True
                except TypeError:
                    key_value_truncated[key] = True

    if total_docs == 0:
        return {"total_records": 0, "message": "No records provided."}

    per_key_report = {}
    for key in sorted(key_counts.keys()):
        per_key_report[key] = {
            "present_in_docs": key_counts[key],
            "missing_in_docs": total_docs - key_counts[key],
            "value_types": sorted(key_types[key]),
            "none_or_empty_count": key_none_count[key],
            "unique_values_sample": sorted(key_values[key], key=str),
            "unique_values_sample_truncated": key_value_truncated[key],
        }

    return {
        "total_records": total_docs,
        "records_with_no_metadata": docs_no_metadata,
        "keys_per_record": {
            "avg": round(n_keys_sum / total_docs, 2),
            "min": n_keys_min,
            "max": n_keys_max,
        },
        "distinct_keys_seen": len(key_counts),
        "per_key": per_key_report,
    }