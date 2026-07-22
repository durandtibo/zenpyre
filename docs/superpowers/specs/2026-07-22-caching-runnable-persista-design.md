# CachingRunnable: switch from pickle-file cache to `persista.cache.Cache`

## Context

`CachingRunnable` (`src/zenpyre/runnables/cache.py`) currently caches the
wrapped runnable's output by pickling it to a file under `cache_dir`, keyed
by `key_fn(input)`. This duplicates a chunk of ad hoc TTL/store logic that
`persista.cache.Cache` already provides in a more general, pluggable form
(in-memory, disk, Redis, etc. backing stores; per-entry TTL; `ignore_none`).
`persista` is already an optional dependency of `zenpyre`
(`pyproject.toml` extra `persista`), and `zenpyre.utils.imports.persista`
already provides `is_persista_available` / `check_persista` /
`raise_persista_missing_error` for guarding optional usage elsewhere in the
codebase.

## Goal

Replace `CachingRunnable`'s disk-pickle cache with a `persista.cache.Cache`
instance supplied by the caller, without making `persista` a hard
dependency of `zenpyre.runnables` (which is imported unconditionally by
other core modules such as `chat_models/cache.py` and
`agents/factory/*.py`).

## API changes

### Constructor

- Replace `cache_dir: Path | str | None = None` with
  `cache: Cache | None = None`.
  - `None` disables caching entirely (same short-circuit behavior as
    `cache_dir=None` today): every call goes straight to the wrapped
    runnable.
  - A `Cache` instance is used as-is; the caller is responsible for
    configuring its backing store, `default_ttl`, and `ignore_none`.
- Drop the `ignore_none: bool = False` constructor parameter. This
  responsibility now belongs entirely to the `Cache` instance the caller
  passes in (`Cache(ignore_none=True)`), avoiding two flags that control
  overlapping behavior.
- `key_fn: Callable[[Input], str] | None = None` is unchanged in signature
  and still defaults to `hash_object`. Its return value is used directly as
  the `Cache` key (no `.pkl` suffix appended, no filesystem-safety
  requirement, since `Cache` keys are plain store keys, not filenames).

### `invoke` / `ainvoke`

- Replace `_cache_path` + `_load_cached`/`_save_cache` (file I/O) with
  `cache.get(key)` / `cache.set(key, result)`.
- `ainvoke` calls these same sync `Cache` methods directly — no
  `asyncio.to_thread` wrapping — since `persista.cache.Cache` is a
  synchronous interface end-to-end (unlike the old pickle I/O, which
  benefited from thread offload for blocking disk access).
- The broad `try/except Exception` around cache load/save is removed:
  `Cache.get`/`Cache.set` don't raise for a normal miss, and there's no
  longer a "corrupt pickle file" failure mode to guard against. If the
  underlying store itself raises (e.g. a network error from a Redis-backed
  store), that propagates normally — it is the caller's responsibility to
  configure a `Cache`/store combination they trust, same as any other
  dependency.

### `batch` / `abatch`

- Same restructuring: look up each input's key via `cache.get`, collect
  cache misses, call the wrapped runnable's own `batch`/`abatch` for the
  misses only, then `cache.set` each newly computed result. Control flow is
  unchanged from today — only the storage calls change.

### `_get_repr_kwargs`

- Replace `"cache_dir": self._cache_dir` with `"cache": self._cache`.

## Optional-dependency handling

`persista.cache.cache.Cache` must not be imported unconditionally at module
load time, since `zenpyre.runnables` (and therefore `persista`) would then
become a hard dependency for every `zenpyre` user, including those who
never construct a `CachingRunnable` with a real cache.

- Under `TYPE_CHECKING`, import `from persista.cache.cache import Cache`
  for type hints only.
- At runtime, `CachingRunnable.__init__` only touches `persista` when
  `cache is not None`: call `check_persista()` (from
  `zenpyre.utils.imports`) before storing `self._cache = cache`, so a
  caller without the `persista` extra installed gets a clear
  `RuntimeError` immediately at construction time — not a confusing
  `AttributeError` on first `invoke`.
  - Note: since `cache` is already a live `Cache` instance by the time it
    reaches `CachingRunnable`, the `persista` package must already be
    importable in the caller's environment for that instance to exist.
    `check_persista()` here is a defense-in-depth guard for consistency
    with the rest of the codebase, not the primary enforcement point.

## Non-goals

- `CachingChatModel` (`src/zenpyre/chat_models/cache.py`) keeps its
  existing disk-pickle caching logic unchanged. It is structurally similar
  but out of scope for this change.
- No change to `agents/factory/cache.py` or `chat_models/factory/cache.py`
  beyond whatever follows mechanically from `CachingRunnable`'s new
  constructor signature (to be checked during implementation).

## Testing

`tests/unit/runnables/test_cache.py` is updated to:

- Construct `persista.cache.cache.Cache()` instances (in-memory store by
  default) instead of using `tmp_path` for a `cache_dir`.
- Drop the `ignore_none`-on-`CachingRunnable` constructor tests (that
  parameter no longer exists); if `ignore_none` behavior needs coverage
  here, it's exercised by passing a `Cache(ignore_none=True)` and checking
  end-to-end `invoke` behavior, not a `CachingRunnable` attribute.
- Drop tests exercising corrupt-pickle-file / cache-read-failure handling,
  since that failure mode no longer exists with `Cache` as the backend.
- Keep the existing coverage shape for: cache-disabled behavior
  (`cache=None`), cache hit/miss on `invoke`/`ainvoke`, partial-hit
  `batch`/`abatch` (only misses go to the wrapped runnable's own
  batch call), `repr` containing the class name, and default `key_fn`
  being `hash_object`.
