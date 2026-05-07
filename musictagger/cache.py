"""File-level cache for the music tagger pipeline.

Closely based on acoustag/cache.py — same core pattern:
  stat() + mtime + size to detect changes, SQLite + WAL for storage.

Extensions over the acoustag version:
  - has_{tag} columns driven by the TAGS registry
  - Auto-migration: new TagDefs add columns on startup
  - mark_changed() resets has_* to NULL when scanner detects a file change
  - needs_inspection() / needs_work() queries for the inspector and worker
  - A threading lock around writes for safety with concurrent workers
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from loguru import logger

from musictagger.tags import TAGS, TagDef

# Base schema — tag columns are added dynamically via migration
_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    filepath          TEXT PRIMARY KEY,
    mtime             REAL NOT NULL,
    size              INTEGER NOT NULL,
    processing_status TEXT,
    last_error        TEXT,
    processed_at      REAL
);
"""

# Extra columns added by _migrate() that are not driven by TAGS.
# Each entry is (column_name, sqlite_type).
_EXTRA_COLUMNS: list[tuple[str, str]] = [
    # SHA-256 hex of the Acoustid/Chromaprint fingerprint string read from the
    # file's acoustid_fingerprint tag during inspection.  NULL when the tag is
    # absent or has not yet been inspected.  Used as the key into embeddings.db
    # so the worker can retrieve cached EffNet embeddings without re-running fpcalc.
    ("fingerprint_hash", "TEXT"),
]


def _col(tag: TagDef) -> str:
    return f"has_{tag.name}"


# Run a PASSIVE WAL checkpoint every this many flush() calls.  A PASSIVE
# checkpoint copies WAL frames to the main database file without blocking
# readers or writers, which keeps the WAL file from growing without bound and
# prevents the multi-second automatic checkpoint stalls that would otherwise
# freeze the stats-refresh thread.  Every ~50 flushes ≈ every ~250 worker
# file writes ≈ roughly once per 5 batches, which is frequent enough to cap
# WAL size while cheap enough not to add meaningful overhead.
_FLUSH_CHECKPOINT_INTERVAL: int = 50


class FileCache:
    """SQLite-backed cache of scanned files and their tag status.

    Column meaning for has_* fields:
      NULL  — not yet inspected (new or recently changed file)
      0     — inspected, tag is absent → needs work
      1     — inspected, tag is present → nothing to do
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Disable SQLite's automatic WAL checkpoint (default threshold: 1000
        # pages).  When the threshold is hit SQLite runs a checkpoint inside
        # the writer's COMMIT, which can block for 2–3 s under write pressure.
        # That hold propagates through our threading.Lock to stats() calls,
        # starving the stats-refresh thread and freezing the TUI display.
        # We run explicit PASSIVE checkpoints in flush() instead; PASSIVE
        # never blocks readers or writers and drains the WAL incrementally.
        self._conn.execute("PRAGMA wal_autocheckpoint=0")
        self._conn.executescript(_BASE_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._closed = False
        # Counter used to throttle the PASSIVE checkpoint in flush() to once
        # every _FLUSH_CHECKPOINT_INTERVAL calls rather than every commit.
        self._flush_count: int = 0
        self._migrate()

    # ── Schema migration ───────────────────────────────────────────────────────

    def _migrate(self) -> None:
        """Add any has_* columns and extra columns that aren't in the DB yet.

        Running this on every startup means adding a TagDef to tags.py
        is all it takes — no manual migration needed.  Extra non-tag columns
        are declared in _EXTRA_COLUMNS.
        """
        existing_cols = {
            row[1] for row in self._conn.execute("PRAGMA table_info(processed)")
        }
        for tag in TAGS:
            col = _col(tag)
            if col not in existing_cols:
                logger.info("Schema migration: adding column {}", col)
                self._conn.execute(f"ALTER TABLE processed ADD COLUMN {col} INTEGER")
        for col, col_type in _EXTRA_COLUMNS:
            if col not in existing_cols:
                logger.info("Schema migration: adding column {}", col)
                self._conn.execute(f"ALTER TABLE processed ADD COLUMN {col} {col_type}")
        # Partial indexes for the inspector's and worker's hot queries.
        # idx_needs_inspection: covers needs_inspection() — files with any
        #   NULL has_* column that haven't been inspected yet.
        # idx_needs_work: covers needs_work() and the stats() queue count —
        #   files that are queued/unqueued and have at least one has_* = 0.
        #   The partial WHERE clause keeps the index small (only actionable rows).
        null_check = " OR ".join(f"{_col(t)} IS NULL" for t in TAGS)
        zero_check = " OR ".join(f"{_col(t)} = 0" for t in TAGS)
        self._conn.executescript(f"""
            CREATE INDEX IF NOT EXISTS idx_needs_inspection
                ON processed(filepath)
                WHERE {null_check};
            CREATE INDEX IF NOT EXISTS idx_needs_work
                ON processed(processing_status)
                WHERE ({zero_check})
                  AND (processing_status = 'queued' OR processing_status IS NULL);
            CREATE INDEX IF NOT EXISTS idx_processing_status
                ON processed(processing_status);
            CREATE INDEX IF NOT EXISTS idx_fingerprint_hash
                ON processed(fingerprint_hash)
                WHERE fingerprint_hash IS NOT NULL;
        """)
        self._conn.commit()

    # ── Scanner interface ──────────────────────────────────────────────────────
    # Scanner calls only these two methods — stat() only, never opens files.

    def is_unchanged(self, filepath: Path) -> bool:
        """Return True if the file matches our cached mtime + size.

        Pure stat() check — no file open, NFS friendly.
        This is the scanner's inner-loop hot path.
        """
        try:
            st = filepath.stat()
        except OSError:
            return False

        with self._lock:
            row = self._conn.execute(
                "SELECT mtime, size FROM processed WHERE filepath = ?",
                (str(filepath),),
            ).fetchone()

        if row is None:
            return False
        return row[0] == st.st_mtime and row[1] == st.st_size

    def mark_changed(self, filepath: Path) -> None:
        """Record a new or changed file, resetting has_* to NULL.

        NULL signals the inspector that this file needs a look.
        Using INSERT OR REPLACE so new files and updates share one path.
        """
        try:
            st = filepath.stat()
        except OSError:
            return

        tag_cols = ", ".join(_col(t) for t in TAGS)
        tag_nulls = ", ".join("NULL" for _ in TAGS)
        # On conflict (existing row) update only mtime, size, processing_status,
        # and the has_* columns so that rowid, last_error, and processed_at are
        # preserved.  INSERT OR REPLACE would delete + reinsert, assigning a new
        # rowid and wiping those columns — which would break ORDER BY rowid ASC
        # in needs_work() and lose error history.
        tag_updates = ", ".join(f"{_col(t)} = NULL" for t in TAGS)

        with self._lock:
            self._conn.execute(
                f"""INSERT INTO processed
                    (filepath, mtime, size, processing_status, {tag_cols})
                    VALUES (?, ?, ?, NULL, {tag_nulls})
                    ON CONFLICT(filepath) DO UPDATE SET
                        mtime = excluded.mtime,
                        size  = excluded.size,
                        processing_status = NULL,
                        {tag_updates}""",
                (str(filepath), st.st_mtime, st.st_size),
            )

    # ── Inspector interface ────────────────────────────────────────────────────

    def needs_inspection(
        self,
        limit: int = 100,
        enabled_tags: list[TagDef] | None = None,
    ) -> list[str]:
        """Files where any enabled has_* is NULL — inspector hasn't visited yet.

        *enabled_tags* restricts the NULL check to a subset of TAGS.  Disabled
        tags are intentionally left NULL by the inspector, so they must not be
        included here or those files would spin in the inspection queue forever.
        When None (the default) all tags are considered.
        """
        active = enabled_tags if enabled_tags is not None else TAGS
        if not active:
            return []
        null_check = " OR ".join(f"{_col(t)} IS NULL" for t in active)
        # Exclude error rows: a file with processing_status='error' and NULL
        # has_* columns (e.g. marked by the worker before inspection completed)
        # would otherwise spin in the inspection queue forever.  Error rows
        # must be manually requeued via requeue_errors() before re-inspection.
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT filepath FROM processed
                    WHERE ({null_check})
                      AND (processing_status != 'error' OR processing_status IS NULL)
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [row[0] for row in rows]

    def set_fingerprint_hash(self, filepath: Path, fp_hash: str) -> None:
        """Store the SHA-256 fingerprint hash for *filepath*.

        Called by the inspector after reading the acoustid_fingerprint tag.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE processed SET fingerprint_hash = ? WHERE filepath = ?",
                (fp_hash, str(filepath)),
            )

    def get_fingerprint_hash(self, filepath_str: str) -> str | None:
        """Return the stored fingerprint hash for *filepath_str*, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT fingerprint_hash FROM processed WHERE filepath = ?",
                (filepath_str,),
            ).fetchone()
        return row[0] if row else None

    def mark_inspected(self, filepath: Path, tag_results: dict[str, bool]) -> None:
        """Set has_* flags after the inspector reads a file.

        tag_results maps tag name → True (present) / False (absent).
        Also sets processing_status to 'queued' if anything is missing,
        or 'done' if the file is fully tagged.
        """
        sets = ", ".join(f"has_{name} = ?" for name in tag_results)
        values = list(int(v) for v in tag_results.values())
        status = "queued" if any(not v for v in tag_results.values()) else "done"

        with self._lock:
            self._conn.execute(
                f"UPDATE processed SET {sets}, processing_status = ? WHERE filepath = ?",
                (*values, status, str(filepath)),
            )

    # ── Worker interface ───────────────────────────────────────────────────────

    def needs_work(
        self,
        limit: int = 50,
        enabled_tags: list[TagDef] | None = None,
    ) -> list[str]:
        """Files where any enabled has_* = 0 and status is queued.

        *enabled_tags* restricts the check to a subset of TAGS.  When None
        (the default) all tags are considered, preserving backwards compatibility.
        """
        active = enabled_tags if enabled_tags is not None else TAGS
        if not active:
            return []
        zero_check = " OR ".join(f"{_col(t)} = 0" for t in active)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT filepath FROM processed
                    WHERE ({zero_check})
                      AND (processing_status = 'queued' OR processing_status IS NULL)
                    ORDER BY rowid ASC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [row[0] for row in rows]

    def get_tag_states(self, filepath_str: str) -> tuple[object, ...] | None:
        """Return the raw has_* column values for a single file.

        Returns a tuple of values in TAGS order, or None if the row is absent.
        Used by the worker to decide which tags still need to be computed
        without bypassing the threading lock.
        """
        tag_cols = ", ".join(f"has_{t.name}" for t in TAGS)
        with self._lock:
            return self._conn.execute(
                f"SELECT {tag_cols} FROM processed WHERE filepath = ?",
                (filepath_str,),
            ).fetchone()

    def mark_working(self, filepath: Path) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                "UPDATE processed SET processing_status = 'working' WHERE filepath = ?",
                (str(filepath),),
            )

    def mark_done(self, filepath: Path) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                """UPDATE processed
                   SET processing_status = 'done', processed_at = ?, last_error = NULL
                   WHERE filepath = ?""",
                (time.time(), str(filepath)),
            )

    def refresh_stat(self, filepath: Path) -> None:
        """Update the cached mtime and size to match the file's current stat.

        The worker writes tags to audio files, which changes their mtime on
        disk.  Without this call the scanner treats every worker-written file
        as 'changed' on its next pass and re-queues it for inspection —
        creating an infinite overwrite loop when overwrite=True is configured.
        Call this immediately after a successful tag write so the cache
        baseline stays in sync with the on-disk state.

        Failures are silently ignored: if the stat fails the cache retains
        the old values and the scanner will re-detect the file on its next
        pass, which is the safe fallback (re-inspect, re-work) rather than
        a crash.
        """
        try:
            st = filepath.stat()
        except OSError:
            return
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                "UPDATE processed SET mtime = ?, size = ? WHERE filepath = ?",
                (st.st_mtime, st.st_size, str(filepath)),
            )

    def mark_done_with_tags(self, filepath: Path, written_tags: list[str]) -> None:
        """Mark a file done and record which tag columns were successfully written.

        The worker calls this instead of mark_done() so that has_* columns are
        set to 1 for every tag that was written.  Without this the columns stay
        at whatever value the inspector left them (0 for absent, or NULL if the
        file was never inspected), which causes the inspector to re-visit the
        file on every pass and the stats to show incorrect counts.

        Tags not in *written_tags* are left unchanged — the inspector is
        responsible for setting those.
        """
        if not written_tags:
            self.mark_done(filepath)
            return

        valid_cols = {_col(t) for t in TAGS}
        cols = [f"has_{name}" for name in written_tags if f"has_{name}" in valid_cols]
        if not cols:
            self.mark_done(filepath)
            return

        set_clause = ", ".join(f"{col} = 1" for col in cols)
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                f"""UPDATE processed
                    SET processing_status = 'done', processed_at = ?,
                        last_error = NULL, {set_clause}
                    WHERE filepath = ?""",
                (time.time(), str(filepath)),
            )

    def mark_error(self, filepath: Path, error: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.execute(
                """UPDATE processed
                   SET processing_status = 'error', last_error = ?
                   WHERE filepath = ?""",
                (error[:500], str(filepath)),
            )

    def requeue_errors(self) -> int:
        """Reset all error-status rows that still have missing tags back to queued.

        Only rows where at least one has_* = 0 are touched — if every tag is
        already present the row stays as-is (the next worker pass will mark it
        done naturally).

        Returns the number of rows requeued so callers can log the result.
        """
        zero_check = " OR ".join(f"{_col(t)} = 0" for t in TAGS)
        with self._lock:
            cur = self._conn.execute(
                f"""UPDATE processed
                    SET processing_status = 'queued', last_error = NULL
                    WHERE processing_status = 'error'
                      AND ({zero_check})""",
            )
            return cur.rowcount

    def requeue_working(self) -> int:
        """Reset all 'working' rows back to 'queued'.

        A row is left in 'working' status only if the process was killed while
        processing that file.  On the next startup those rows would be silently
        excluded from needs_work() forever because only 'queued'/NULL rows are
        selected.  This method is called once at startup to recover from such
        crashes.

        Returns the number of rows reset so callers can log the result.
        """
        with self._lock:
            cur = self._conn.execute(
                """UPDATE processed
                   SET processing_status = 'queued', last_error = 'recovered from interrupted working state'
                   WHERE processing_status = 'working'"""
            )
            return cur.rowcount

    def requeue_done_missing_tags(
        self,
        enabled_tags: list[TagDef] | None = None,
    ) -> int:
        """Reset 'done' rows where any enabled tag column is still 0 or NULL.

        Catches two distinct bad states:
          - has_* = 0: inspector flagged the tag absent but worker marked done
            without writing it (e.g. process_file() returned an empty dict).
          - has_* = NULL: worker wrote the tag but called plain mark_done()
            instead of mark_done_with_tags(), leaving the column uninspected.

        *enabled_tags* restricts the check to tags that are active in the
        current config.  When None all tags are considered.

        Returns the number of rows reset.
        """
        active = enabled_tags if enabled_tags is not None else TAGS
        if not active:
            return 0
        missing_check = " OR ".join(
            f"({_col(t)} = 0 OR {_col(t)} IS NULL)" for t in active
        )
        with self._lock:
            cur = self._conn.execute(
                f"""UPDATE processed
                    SET processing_status = 'queued', last_error = 'recovered: done with missing tags'
                    WHERE processing_status = 'done'
                      AND ({missing_check})"""
            )
            return cur.rowcount

    # ── Overview stats ─────────────────────────────────────────────────────────

    def library_size_bytes(self) -> int:
        """Return the sum of all tracked file sizes in bytes.

        Uses the ``size`` column already stored by the scanner so no extra
        filesystem I/O is needed.  Returns 0 when the table is empty or closed.
        """
        with self._lock:
            if self._closed:
                return 0
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM processed"
            ).fetchone()
        return int(row[0])

    def error_summary(self, limit: int = 3) -> list[tuple[str, int]]:
        """Return the most common error messages across all error rows.

        Groups ``last_error`` values and returns up to *limit* entries as
        ``(message, count)`` tuples, ordered by count descending.  Rows
        with NULL ``last_error`` are excluded.  Used by the overview panel
        to surface recurring failure patterns without exposing individual
        file paths.  Returns an empty list when the connection is closed.
        """
        with self._lock:
            if self._closed:
                return []
            rows = self._conn.execute(
                """SELECT last_error, COUNT(*) AS n
                   FROM processed
                   WHERE processing_status = 'error'
                     AND last_error IS NOT NULL
                   GROUP BY last_error
                   ORDER BY n DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [(str(row[0]), int(row[1])) for row in rows]

    def fingerprinted_count(self) -> int:
        """Return the number of tracked files that have a stored fingerprint hash."""
        with self._lock:
            if self._closed:
                return 0
            row = self._conn.execute(
                "SELECT COUNT(*) FROM processed WHERE fingerprint_hash IS NOT NULL"
            ).fetchone()
        return int(row[0]) if row else 0

    # ── Cleanup interface ──────────────────────────────────────────────────────

    def all_filepaths(self) -> list[str]:
        with self._lock:
            if self._closed:
                return []
            rows = self._conn.execute("SELECT filepath FROM processed").fetchall()
        return [row[0] for row in rows]

    def remove(self, filepath: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.execute("DELETE FROM processed WHERE filepath = ?", (filepath,))

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(
        self,
        enabled_tags: list[TagDef] | None = None,
    ) -> dict[str, int]:
        """Return pipeline counters.

        *enabled_tags* restricts the NULL / zero checks to the tags that are
        actually active in the current config.  When None (the default) all tags
        are used, which is correct for callers that have no config context but
        will overcount when some tags are disabled.
        """
        active = enabled_tags if enabled_tags is not None else TAGS

        # Initialise every output variable to a safe zero so that if any
        # individual query raises (e.g. the connection was closed during
        # shutdown) the caller receives a coherent all-zero snapshot rather
        # than an UnboundLocalError propagating up through the TUI.
        total: int = 0
        needs_insp: int = 0
        needs_work: int = 0
        in_progress: int = 0
        errors: int = 0
        done: int = 0
        per_tag: dict[str, int] = {t.name: 0 for t in TAGS}

        # Return the zeroed snapshot immediately if the connection is already
        # closed — background refresh threads may call stats() while teardown
        # is in progress; we prefer a stale zero over a raised exception.
        if self._closed:
            return {
                "total": total,
                "needs_inspection": needs_insp,
                "needs_work": needs_work,
                "in_progress": in_progress,
                "errors": errors,
                "done": done,
                "per_tag": per_tag,
            }

        # Run each aggregate in its own short lock window rather than holding
        # the lock across all four queries.  The counts are used only for
        # display and do not need to be mutually consistent, so the brief
        # unlock gaps between queries are acceptable.  This prevents the
        # stats refresh (called every second) from blocking the worker's
        # post-inference DB writes for hundreds of milliseconds at a time.
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM processed").fetchone()[0]

        if active:
            null_check = " OR ".join(f"{_col(t)} IS NULL" for t in active)
            with self._lock:
                needs_insp = self._conn.execute(
                    f"SELECT COUNT(*) FROM processed WHERE {null_check}"
                ).fetchone()[0]

        # Count files the worker still needs to finish: queued/NULL rows (waiting
        # to be picked up) plus 'working' rows (already dequeued but not yet
        # committed as done/error).  Including 'working' ensures the displayed
        # count decrements smoothly as files finish rather than appearing frozen
        # while a batch is in flight.  Error-status rows are still excluded so
        # the count stays consistent with what needs_work() actually returns.
        if active:
            zero_check = " OR ".join(f"{_col(t)} = 0" for t in active)
            with self._lock:
                needs_work = self._conn.execute(
                    f"""SELECT COUNT(*) FROM processed
                        WHERE ({zero_check})
                          AND (processing_status = 'queued'
                               OR processing_status IS NULL
                               OR processing_status = 'working')""",
                ).fetchone()[0]

        # Collapse in_progress, errors, done, and all per-tag counts into one scan.
        per_tag_exprs = ", ".join(
            f"SUM(CASE WHEN {_col(t)} = 1 THEN 1 ELSE 0 END) AS {_col(t)}" for t in TAGS
        )
        with self._lock:
            agg_row = self._conn.execute(
                f"""SELECT
                    SUM(CASE WHEN processing_status = 'working' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN processing_status = 'error'   THEN 1 ELSE 0 END),
                    SUM(CASE WHEN processing_status = 'done'    THEN 1 ELSE 0 END),
                    {per_tag_exprs}
                FROM processed"""
            ).fetchone()
        in_progress = agg_row[0] or 0
        errors = agg_row[1] or 0
        done = agg_row[2] or 0
        per_tag = {t.name: (agg_row[3 + i] or 0) for i, t in enumerate(TAGS)}

        return {
            "total": total,
            "needs_inspection": needs_insp,
            "needs_work": needs_work,
            "in_progress": in_progress,
            "errors": errors,
            "done": done,
            "per_tag": per_tag,
        }

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """Commit pending writes and periodically run a PASSIVE WAL checkpoint.

        The checkpoint is non-blocking: it copies WAL frames to the main
        database file only when no reader holds a snapshot that overlaps those
        frames.  Running it here prevents the WAL from growing until SQLite
        triggers its own auto-checkpoint inside a COMMIT (which can stall
        writers for several seconds and freeze the stats-refresh thread).
        """
        with self._lock:
            if self._closed:
                return
            self._conn.commit()
            self._flush_count += 1
            if self._flush_count % _FLUSH_CHECKPOINT_INTERVAL == 0:
                # wal_checkpoint(PASSIVE) never blocks; it returns immediately
                # if any frame is still needed by an open reader.
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.commit()
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
