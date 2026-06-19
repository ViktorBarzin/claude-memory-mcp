"""Single, process-wide serialized SQLite writer for the local memory cache.

SQLite permits only one writer at a time. The MCP server's store path and the
background sync engine used to open *separate* connections to the *same* file;
under heavy concurrent ``memory_store`` calls those two writers fought over the
single SQLite write lock, blew past ``busy_timeout``, and surfaced
``sqlite3.OperationalError: database is locked`` — which made the tool slow and
eventually dropped the session.

``LocalStore`` fixes this structurally: it owns ONE connection (opened with
``check_same_thread=False``) guarded by ONE re-entrant lock. Every component that
needs to touch the local DB shares the same ``LocalStore`` instance, so all
writes serialize cleanly through the in-process lock and queue instead of racing
the SQLite writer. On the rare residual lock (e.g. another OS process touching
the file), writes retry with bounded exponential backoff rather than failing the
caller. WAL stays on for concurrent reads.

Uses only stdlib — no pip install required.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Bounded retry window for the rare residual "database is locked" — handles a
# lock held by a *different OS process* (the in-process lock already serializes
# this process's own writers). Total worst-case wait ≈ 0.05+0.1+0.2+0.4+0.8 ≈ 1.55s.
_MAX_RETRIES = 5
_BASE_BACKOFF_S = 0.05
_BUSY_TIMEOUT_MS = 30000


def _is_locked_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "database is locked" in msg or "database is busy" in msg


class LocalStore:
    """Owns the single shared SQLite connection + lock for local memory writes."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        # Re-entrant so a transaction callback may itself call ``execute``/``write``
        # without dead-locking on the same thread.
        self.lock = threading.RLock()

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def open(cls, db_path: str) -> "LocalStore":
        """Open (creating parent dirs) a WAL connection safe for cross-thread use."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return cls(conn)

    # ── serialized access ───────────────────────────────────────────

    def transaction(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn(conn)`` holding the shared lock, with bounded retry on lock errors.

        ``fn`` is responsible for issuing its own ``COMMIT`` (call ``conn.commit()``)
        when it performs writes. The whole callback runs under the process-wide lock,
        so concurrent callers queue rather than collide on the SQLite writer.
        """
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(_MAX_RETRIES):
            with self.lock:
                try:
                    return fn(self.conn)
                except sqlite3.OperationalError as exc:
                    if not _is_locked_error(exc):
                        raise
                    last_exc = exc
                    # Roll back any partial txn so the retry starts clean and the
                    # connection isn't left mid-transaction holding locks.
                    try:
                        self.conn.rollback()
                    except sqlite3.Error:
                        pass
            # Back off *outside* the lock so other writers can make progress.
            backoff = _BASE_BACKOFF_S * (2 ** attempt)
            logger.warning(
                "SQLite locked (attempt %d/%d) — backing off %.3fs", attempt + 1, _MAX_RETRIES, backoff
            )
            time.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Run a single read query under the shared lock (no implicit commit)."""
        with self.lock:
            return self.conn.execute(sql, params)

    def write(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """Run a single write statement + commit, serialized and retry-guarded."""

        def _do(conn: sqlite3.Connection) -> sqlite3.Cursor:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

        return self.transaction(_do)

    def close(self) -> None:
        with self.lock:
            self.conn.close()
