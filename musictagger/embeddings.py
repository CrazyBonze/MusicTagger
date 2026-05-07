"""Content-addressed cache for heavy-to-compute audio embeddings.

Embeddings are keyed by a SHA-256 hash of the Chromaprint fingerprint string
for the audio content.  Because Chromaprint operates on chroma features
derived from downsampled PCM it is robust to container/codec differences —
the same recording encoded as MP3 or FLAC produces (nearly) the same
fingerprint and therefore the same cache key.

The fingerprint itself is read from the ``acoustid_fingerprint`` tag already
written to the file by the acoustag tool.  This avoids re-running ``fpcalc``
on every worker pass; the inspector reads the tag once and stores the hash in
the main ``cache.db``.  If the tag is absent the worker simply skips the cache
and recomputes embeddings as before.

Schema (``embeddings.db``)
--------------------------
One table — ``embeddings`` — with a composite primary key of
``(fingerprint_hash, model)``.  This means a single database file can store
embeddings from multiple models without any schema migration when a new model
is introduced.

    fingerprint_hash  TEXT    — 64-char lowercase SHA-256 hex of the raw
                               Chromaprint fingerprint string
    model             TEXT    — model filename, e.g. "discogs-effnet-bs64-1.pb"
    computed_at       REAL    — Unix timestamp of when this row was written
    n_patches         INTEGER — first dimension of the stored array
    n_dims            INTEGER — second dimension of the stored array
    encoding          TEXT    — blob encoding: "raw_f32" (legacy) or "zlib_f16"
    data              BLOB    — encoded bytes of the array (see encoding column)

Encoding schemes
----------------
``raw_f32``  (legacy)
    Raw little-endian float32 bytes, as originally stored.  No compression.
    Retained for backward-compatible reads of pre-migration rows.

``zlib_f16``  (current)
    Array cast to float16, serialised as little-endian bytes, then compressed
    with ``zlib.compress()``.  Typically ~6-8x smaller than ``raw_f32`` for
    EffNet embeddings (float16 halves precision bytes; zlib reduces further by
    ~3-4x on the resulting data).  The public ``get()`` API always returns
    float32 regardless of encoding, so callers are unaffected.

Migration
---------
Call ``EmbeddingCache.migrate_encoding()`` (or the
``musictagger-migrate-embeddings`` CLI script) to rewrite all legacy
``raw_f32`` rows in-place to ``zlib_f16``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
import zlib
from pathlib import Path

import numpy as np
from loguru import logger

# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    fingerprint_hash  TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    computed_at       REAL    NOT NULL,
    n_patches         INTEGER NOT NULL,
    n_dims            INTEGER NOT NULL,
    encoding          TEXT    NOT NULL DEFAULT 'raw_f32',
    data              BLOB    NOT NULL,
    PRIMARY KEY (fingerprint_hash, model)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_hash
    ON embeddings(fingerprint_hash);
"""

# Migration adds the encoding column to pre-existing databases that were
# created before this column was introduced.
_MIGRATE_ADD_ENCODING = (
    "ALTER TABLE embeddings ADD COLUMN encoding TEXT NOT NULL DEFAULT 'raw_f32'"
)

_ENCODING_RAW_F32 = "raw_f32"
_ENCODING_ZLIB_F16 = "zlib_f16"

# Batch size for the migration loop — keeps transactions small enough that the
# DB remains responsive to reads while the migration is running.
_MIGRATION_BATCH = 200


# ── Module-level helper ────────────────────────────────────────────────────────


def fingerprint_hash(raw_fingerprint: str) -> str:
    """Return the SHA-256 hex digest of *raw_fingerprint*.

    This is the canonical cache key for the embeddings database.  Using a hash
    rather than the raw fingerprint string keeps keys short (64 chars) and
    uniform regardless of fingerprint length.

    The input must be the raw Chromaprint fingerprint string as returned by
    ``fpcalc`` or ``acoustid.fingerprint_file()`` — not a base64-decoded byte
    sequence.
    """
    return hashlib.sha256(raw_fingerprint.encode()).hexdigest()


# ── Encoding helpers ───────────────────────────────────────────────────────────


def _encode(arr: np.ndarray) -> tuple[str, bytes]:
    """Encode *arr* as zlib-compressed float16 bytes.

    Returns ``(encoding_name, compressed_bytes)``.
    """
    f16 = np.asarray(arr, dtype=np.float16)
    return _ENCODING_ZLIB_F16, zlib.compress(f16.tobytes(), level=6)


def _decode(encoding: str, raw: bytes, n_patches: int, n_dims: int) -> np.ndarray:
    """Decode *raw* bytes back to a float32 ndarray given *encoding*.

    Always returns float32 so callers remain unaffected by the storage format.
    """
    if encoding == _ENCODING_ZLIB_F16:
        data = zlib.decompress(raw)
        return (
            np.frombuffer(data, dtype=np.float16)
            .reshape(n_patches, n_dims)
            .astype(np.float32)
        )
    if encoding == _ENCODING_RAW_F32:
        return np.frombuffer(raw, dtype=np.float32).reshape(n_patches, n_dims)
    raise ValueError(f"Unknown embedding encoding: {encoding!r}")


# ── EmbeddingCache ─────────────────────────────────────────────────────────────


class EmbeddingCache:
    """SQLite-backed store for audio embedding arrays.

    Thread-safe via an internal ``threading.Lock``.  Use as a context manager
    or call ``close()`` explicitly when done.

    Typical usage in the worker::

        cache = EmbeddingCache(config.embeddings_db_path)
        fp_hash = ...                         # from FileCache
        embeddings = cache.get(fp_hash, model_filename)
        if embeddings is None:
            embeddings = compute_embeddings(filepath)
            cache.put(fp_hash, model_filename, embeddings)
            cache.flush()
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_encoding_column()
        self._conn.commit()
        self._lock = threading.Lock()
        self._closed = False
        logger.debug("EmbeddingCache opened: {}", db_path)

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _ensure_encoding_column(self) -> None:
        """Add the ``encoding`` column if it is missing (pre-migration DBs)."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "encoding" not in cols:
            try:
                self._conn.execute(_MIGRATE_ADD_ENCODING)
                logger.info(
                    "EmbeddingCache: added 'encoding' column to existing database"
                )
            except sqlite3.OperationalError:
                # Column already exists — race or repeated open; safe to ignore.
                pass

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, fp_hash: str, model: str) -> np.ndarray | None:
        """Return the cached embedding array, or None if not present.

        Reconstructs a ``float32`` array from the stored bytes using the
        ``n_patches``, ``n_dims``, and ``encoding`` columns.  The returned
        dtype is always ``float32`` regardless of how the row was stored.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT n_patches, n_dims, encoding, data FROM embeddings"
                " WHERE fingerprint_hash = ? AND model = ?",
                (fp_hash, model),
            ).fetchone()

        if row is None:
            return None

        n_patches, n_dims, encoding, raw = row
        try:
            arr = _decode(encoding, raw, n_patches, n_dims)
        except Exception as exc:
            # Corrupted or unrecognised blob — log and treat as a cache miss.
            logger.warning(
                "EmbeddingCache: unreadable blob for hash={} model={} encoding={}: {}"
                " — treating as miss",
                fp_hash[:12],
                model,
                encoding,
                exc,
            )
            return None

        logger.debug(
            "EmbeddingCache hit: hash={} model={} shape=({}, {}) encoding={}",
            fp_hash[:12],
            model,
            n_patches,
            n_dims,
            encoding,
        )
        return arr

    def put(self, fp_hash: str, model: str, embeddings: np.ndarray) -> None:
        """Store *embeddings* for the given fingerprint hash and model.

        The array is encoded as zlib-compressed float16 bytes before storage.
        Any existing row for the same ``(fp_hash, model)`` pair is replaced.

        Call ``flush()`` periodically to commit accumulated writes.
        """
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:  # noqa: PLR2004
            raise ValueError(
                f"embeddings must be 2-D (n_patches × n_dims), got shape {arr.shape}"
            )
        n_patches, n_dims = arr.shape
        encoding, raw = _encode(arr)

        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO embeddings
                   (fingerprint_hash, model, computed_at, n_patches, n_dims, encoding, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (fp_hash, model, time.time(), n_patches, n_dims, encoding, raw),
            )

        logger.debug(
            "EmbeddingCache put: hash={} model={} shape=({}, {}) encoding={}",
            fp_hash[:12],
            model,
            n_patches,
            n_dims,
            encoding,
        )

    def migrate_encoding(
        self,
        progress_cb: object | None = None,
    ) -> tuple[int, int]:
        """Rewrite all ``raw_f32`` rows in-place to ``zlib_f16`` encoding.

        Processes rows in batches of ``_MIGRATION_BATCH`` to avoid holding a
        large write transaction.  The lock is acquired per batch, not for the
        entire migration, so the cache remains readable by other threads while
        work is in progress.

        Parameters
        ----------
        progress_cb:
            Optional callable invoked after each batch with signature
            ``progress_cb(done: int, total: int)``.  Useful for CLI progress
            display.  Must be safe to call without the lock.

        Returns
        -------
        tuple[int, int]
            ``(rows_migrated, total_legacy_rows)`` — number of rows that were
            re-encoded and the total number of legacy rows found at the start.
        """
        # Count legacy rows first (outside the lock — reads are fine).
        with self._lock:
            total: int = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE encoding = ?",
                (_ENCODING_RAW_F32,),
            ).fetchone()[0]

        if total == 0:
            logger.info("EmbeddingCache.migrate_encoding: nothing to migrate")
            return 0, 0

        logger.info(
            "EmbeddingCache.migrate_encoding: {} legacy rows to re-encode", total
        )

        done = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT fingerprint_hash, model, n_patches, n_dims, data"
                    " FROM embeddings WHERE encoding = ? LIMIT ?",
                    (_ENCODING_RAW_F32, _MIGRATION_BATCH),
                ).fetchall()

            if not rows:
                break

            updates: list[tuple[str, bytes, str, str]] = []
            for fp_hash, model, n_patches, n_dims, raw in rows:
                try:
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(
                        n_patches, n_dims
                    )
                    enc, new_raw = _encode(arr)
                    updates.append((enc, new_raw, fp_hash, model))
                except Exception as exc:
                    # Skip corrupt blobs — leave them as-is.
                    logger.warning(
                        "EmbeddingCache.migrate_encoding: skipping corrupt row"
                        " hash={} model={}: {}",
                        fp_hash[:12],
                        model,
                        exc,
                    )

            with self._lock:
                self._conn.executemany(
                    "UPDATE embeddings SET encoding = ?, data = ?"
                    " WHERE fingerprint_hash = ? AND model = ?",
                    updates,
                )
                self._conn.commit()

            done += len(updates)
            if progress_cb is not None:
                progress_cb(done, total)  # type: ignore[operator]

        logger.info("EmbeddingCache.migrate_encoding: migrated {}/{} rows", done, total)
        return done, total

    def stats(self) -> dict[str, int]:
        """Return lightweight aggregate statistics about the cache contents.

        Executes cheap COUNT queries under the lock and returns::

            {
                "total_embeddings": int,    # total rows in the embeddings table
                "unique_fingerprints": int, # distinct fingerprint_hash values
                "legacy_rows": int,         # rows still using raw_f32 encoding
            }

        All values are 0 when the table is empty.  No BLOBs are read.
        """
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            unique = self._conn.execute(
                "SELECT COUNT(DISTINCT fingerprint_hash) FROM embeddings"
            ).fetchone()[0]
            legacy = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE encoding = ?",
                (_ENCODING_RAW_F32,),
            ).fetchone()[0]
        return {
            "total_embeddings": int(total),
            "unique_fingerprints": int(unique),
            "legacy_rows": int(legacy),
        }

    def flush(self) -> None:
        """Commit pending writes to disk."""
        with self._lock:
            if self._closed:
                return
            self._conn.commit()

    def close(self) -> None:
        """Flush and close the database connection.  Safe to call multiple times."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.commit()
            self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
