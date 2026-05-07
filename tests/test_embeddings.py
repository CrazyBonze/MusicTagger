"""Tests for musictagger.embeddings — EmbeddingCache and fingerprint_hash."""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pytest

from musictagger.embeddings import (
    EmbeddingCache,
    _ENCODING_RAW_F32,
    _ENCODING_ZLIB_F16,
    fingerprint_hash,
)


# ── fingerprint_hash ──────────────────────────────────────────────────────────


def test_fingerprint_hash_returns_sha256_hex() -> None:
    raw = "AQADtEkUSYmSD_2RL3iO"
    expected = hashlib.sha256(raw.encode()).hexdigest()
    assert fingerprint_hash(raw) == expected


def test_fingerprint_hash_is_64_chars() -> None:
    assert len(fingerprint_hash("anything")) == 64


def test_fingerprint_hash_same_input_same_output() -> None:
    fp = "AQADtEkUSYmSD_2RL3iO"
    assert fingerprint_hash(fp) == fingerprint_hash(fp)


def test_fingerprint_hash_different_inputs_differ() -> None:
    assert fingerprint_hash("abc") != fingerprint_hash("def")


# ── EmbeddingCache ────────────────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_path):
    db = tmp_path / "embeddings.db"
    ec = EmbeddingCache(db)
    try:
        yield ec
    finally:
        ec.close()


def test_cache_miss_returns_none(cache: EmbeddingCache) -> None:
    result = cache.get("nonexistent" * 4, "some-model.pb")
    assert result is None


def test_cache_put_then_get_round_trips_array(cache: EmbeddingCache) -> None:
    fp = fingerprint_hash("test-fingerprint")
    model = "discogs-effnet-bs64-1.pb"
    arr = np.random.default_rng(0).random((42, 1280), dtype=np.float64)

    cache.put(fp, model, arr)
    cache.flush()
    result = cache.get(fp, model)

    assert result is not None
    assert result.shape == (42, 1280)
    assert result.dtype == np.float32
    # The encoding path is float64 → float32 → float16 → float32.
    # Compare against the exact same conversion sequence to avoid rounding
    # boundary mismatches caused by the intermediate float32 step.
    expected = arr.astype(np.float32).astype(np.float16).astype(np.float32)
    np.testing.assert_array_equal(result, expected)


def test_cache_put_stores_zlib_f16_encoding(cache: EmbeddingCache, tmp_path) -> None:
    """New rows must be stored with zlib_f16 encoding."""
    fp = fingerprint_hash("encoding-check")
    model = "model.pb"
    cache.put(fp, model, np.ones((5, 1280), dtype=np.float32))
    cache.flush()

    row = cache._conn.execute(
        "SELECT encoding FROM embeddings WHERE fingerprint_hash = ? AND model = ?",
        (fp, model),
    ).fetchone()
    assert row is not None
    assert row[0] == _ENCODING_ZLIB_F16


def test_cache_different_models_stored_independently(cache: EmbeddingCache) -> None:
    fp = fingerprint_hash("shared-fingerprint")
    arr_a = np.ones((10, 1280), dtype=np.float32)
    arr_b = np.zeros((20, 512), dtype=np.float32)

    cache.put(fp, "model-a.pb", arr_a)
    cache.put(fp, "model-b.pb", arr_b)
    cache.flush()

    result_a = cache.get(fp, "model-a.pb")
    result_b = cache.get(fp, "model-b.pb")

    assert result_a is not None and result_a.shape == (10, 1280)
    assert result_b is not None and result_b.shape == (20, 512)


def test_cache_put_replaces_existing_row(cache: EmbeddingCache) -> None:
    fp = fingerprint_hash("fp-to-replace")
    model = "model.pb"
    original = np.ones((5, 1280), dtype=np.float32)
    updated = np.full((5, 1280), 2.0, dtype=np.float32)

    cache.put(fp, model, original)
    cache.put(fp, model, updated)
    cache.flush()

    result = cache.get(fp, model)
    assert result is not None
    np.testing.assert_allclose(result, updated.astype(np.float16).astype(np.float32))


def test_cache_different_fingerprints_stored_independently(
    cache: EmbeddingCache,
) -> None:
    fp_a = fingerprint_hash("song-a")
    fp_b = fingerprint_hash("song-b")
    model = "model.pb"
    arr_a = np.ones((3, 1280), dtype=np.float32)
    arr_b = np.full((3, 1280), 9.0, dtype=np.float32)

    cache.put(fp_a, model, arr_a)
    cache.put(fp_b, model, arr_b)
    cache.flush()

    assert cache.get(fp_a, model) is not None
    assert cache.get(fp_b, model) is not None
    np.testing.assert_allclose(
        cache.get(fp_a, model), arr_a.astype(np.float16).astype(np.float32)
    )
    np.testing.assert_allclose(
        cache.get(fp_b, model), arr_b.astype(np.float16).astype(np.float32)
    )


def test_cache_put_rejects_non_2d_array(cache: EmbeddingCache) -> None:
    fp = fingerprint_hash("bad-shape")
    with pytest.raises(ValueError, match="2-D"):
        cache.put(fp, "model.pb", np.ones((1280,), dtype=np.float32))


def test_cache_persists_across_reopen(tmp_path) -> None:
    db = tmp_path / "embeddings.db"
    fp = fingerprint_hash("persistence-test")
    model = "model.pb"
    arr = np.full((7, 1280), 3.14, dtype=np.float32)

    with EmbeddingCache(db) as ec:
        ec.put(fp, model, arr)
        ec.flush()

    with EmbeddingCache(db) as ec2:
        result = ec2.get(fp, model)

    assert result is not None
    np.testing.assert_allclose(result, arr.astype(np.float16).astype(np.float32))


def test_cache_creates_parent_directory(tmp_path) -> None:
    db = tmp_path / "nested" / "dir" / "embeddings.db"
    with EmbeddingCache(db) as ec:
        ec.put(fingerprint_hash("x"), "m.pb", np.zeros((1, 1), dtype=np.float32))
        ec.flush()
    assert db.exists()


# ── Legacy raw_f32 backward-compatibility ─────────────────────────────────────


def _insert_raw_f32_row(
    conn: sqlite3.Connection,
    fp_hash: str,
    model: str,
    arr: np.ndarray,
) -> None:
    """Insert a legacy raw_f32 row directly, bypassing EmbeddingCache.put()."""
    f32 = np.asarray(arr, dtype=np.float32)
    n_patches, n_dims = f32.shape
    conn.execute(
        """INSERT OR REPLACE INTO embeddings
           (fingerprint_hash, model, computed_at, n_patches, n_dims, encoding, data)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (fp_hash, model, 0.0, n_patches, n_dims, _ENCODING_RAW_F32, f32.tobytes()),
    )
    conn.commit()


def test_cache_reads_legacy_raw_f32_row(tmp_path) -> None:
    """get() must decode legacy raw_f32 blobs and return float32 arrays."""
    db = tmp_path / "embeddings.db"
    arr = np.full((10, 1280), 1.5, dtype=np.float32)
    fp = fingerprint_hash("legacy-song")
    model = "model.pb"

    with EmbeddingCache(db) as ec:
        _insert_raw_f32_row(ec._conn, fp, model, arr)
        result = ec.get(fp, model)

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (10, 1280)
    np.testing.assert_array_equal(result, arr)


def test_cache_reads_legacy_db_without_encoding_column(tmp_path) -> None:
    """Opening a DB that has no encoding column should add it and read rows."""
    db = tmp_path / "old.db"

    # Build a pre-migration DB without the encoding column.
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            fingerprint_hash  TEXT    NOT NULL,
            model             TEXT    NOT NULL,
            computed_at       REAL    NOT NULL,
            n_patches         INTEGER NOT NULL,
            n_dims            INTEGER NOT NULL,
            data              BLOB    NOT NULL,
            PRIMARY KEY (fingerprint_hash, model)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_hash
            ON embeddings(fingerprint_hash);
    """)
    arr = np.full((3, 1280), 2.0, dtype=np.float32)
    fp = fingerprint_hash("old-db-song")
    model = "model.pb"
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?)",
        (fp, model, 0.0, 3, 1280, arr.tobytes()),
    )
    conn.commit()
    conn.close()

    # Opening via EmbeddingCache must add the column and read the row.
    with EmbeddingCache(db) as ec:
        result = ec.get(fp, model)

    assert result is not None
    assert result.shape == (3, 1280)
    np.testing.assert_array_equal(result, arr)


# ── migrate_encoding ──────────────────────────────────────────────────────────


def test_migrate_encoding_converts_legacy_rows(tmp_path) -> None:
    """migrate_encoding() rewrites raw_f32 rows to zlib_f16."""
    db = tmp_path / "embeddings.db"
    arr = np.random.default_rng(1).random((15, 1280)).astype(np.float32)
    fp = fingerprint_hash("migrate-me")
    model = "model.pb"

    with EmbeddingCache(db) as ec:
        _insert_raw_f32_row(ec._conn, fp, model, arr)

        done, total = ec.migrate_encoding()

        assert done == 1
        assert total == 1

        # Row should now be zlib_f16.
        row = ec._conn.execute(
            "SELECT encoding FROM embeddings WHERE fingerprint_hash = ? AND model = ?",
            (fp, model),
        ).fetchone()
        assert row[0] == _ENCODING_ZLIB_F16

        # Data should still be recoverable and close to original.
        result = ec.get(fp, model)
    assert result is not None
    np.testing.assert_allclose(result, arr.astype(np.float16).astype(np.float32))


def test_migrate_encoding_skips_already_migrated_rows(tmp_path) -> None:
    """migrate_encoding() reports 0 when all rows are already zlib_f16."""
    db = tmp_path / "embeddings.db"
    fp = fingerprint_hash("already-new")
    model = "model.pb"

    with EmbeddingCache(db) as ec:
        ec.put(fp, model, np.ones((5, 1280), dtype=np.float32))
        ec.flush()

        done, total = ec.migrate_encoding()

    assert done == 0
    assert total == 0


def test_migrate_encoding_calls_progress_cb(tmp_path) -> None:
    """progress_cb is invoked at least once during a migration."""
    db = tmp_path / "embeddings.db"
    calls: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        calls.append((done, total))

    with EmbeddingCache(db) as ec:
        for i in range(3):
            fp = fingerprint_hash(f"song-{i}")
            _insert_raw_f32_row(ec._conn, fp, "m.pb", np.ones((2, 8), dtype=np.float32))

        ec.migrate_encoding(progress_cb=cb)

    assert len(calls) >= 1
    assert calls[-1][0] == 3


def test_migrate_encoding_mixed_db(tmp_path) -> None:
    """Only raw_f32 rows are re-encoded; existing zlib_f16 rows are untouched."""
    db = tmp_path / "embeddings.db"
    fp_new = fingerprint_hash("already-new")
    fp_old = fingerprint_hash("needs-migration")
    model = "m.pb"

    with EmbeddingCache(db) as ec:
        # Write one new-format row via the public API.
        ec.put(fp_new, model, np.ones((4, 8), dtype=np.float32))
        ec.flush()
        # Insert one legacy row directly.
        _insert_raw_f32_row(
            ec._conn, fp_old, model, np.full((4, 8), 3.0, dtype=np.float32)
        )

        done, total = ec.migrate_encoding()

    assert done == 1
    assert total == 1


def test_stats_includes_legacy_rows_count(tmp_path) -> None:
    """stats() must report legacy_rows correctly."""
    db = tmp_path / "embeddings.db"
    fp_old = fingerprint_hash("old")
    fp_new = fingerprint_hash("new")
    model = "m.pb"

    with EmbeddingCache(db) as ec:
        _insert_raw_f32_row(ec._conn, fp_old, model, np.ones((2, 8), dtype=np.float32))
        ec.put(fp_new, model, np.ones((2, 8), dtype=np.float32))
        ec.flush()

        s = ec.stats()

    assert s["total_embeddings"] == 2
    assert s["legacy_rows"] == 1


# ── FileCache fingerprint_hash column ─────────────────────────────────────────


def test_file_cache_fingerprint_hash_roundtrip(tmp_path) -> None:
    """FileCache.set_fingerprint_hash / get_fingerprint_hash store and retrieve."""
    from pathlib import Path

    from musictagger.cache import FileCache

    db = tmp_path / "cache.db"
    cache = FileCache(db)
    try:
        fp = Path(tmp_path / "song.mp3")
        fp.write_bytes(b"")
        cache.mark_changed(fp)
        cache.flush()

        fp_hash = fingerprint_hash("AQADtEkUSYmSD_2RL3iO")
        cache.set_fingerprint_hash(fp, fp_hash)
        cache.flush()

        assert cache.get_fingerprint_hash(str(fp)) == fp_hash
    finally:
        cache.close()


def test_file_cache_fingerprint_hash_missing_returns_none(tmp_path) -> None:
    from musictagger.cache import FileCache

    db = tmp_path / "cache.db"
    cache = FileCache(db)
    try:
        assert cache.get_fingerprint_hash("/nonexistent/file.mp3") is None
    finally:
        cache.close()
