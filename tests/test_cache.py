"""Unit tests for the SQLite file cache."""

from __future__ import annotations

from pathlib import Path

from musictagger.cache import FileCache
from musictagger.tags import TAGS


def _all_tag_results(value: bool) -> dict[str, bool]:
    return {tag.name: value for tag in TAGS}


def _make_audio_file(tmp_path: Path, name: str = "track.mp3") -> Path:
    filepath = tmp_path / name
    filepath.write_bytes(b"test-audio-data")
    return filepath


def test_mark_changed_adds_file_to_inspection_queue(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        assert cache.is_unchanged(filepath) is False

        cache.mark_changed(filepath)
        cache.flush()

        assert cache.is_unchanged(filepath) is True
        assert cache.needs_inspection() == [str(filepath)]

        columns = {
            row[1] for row in cache._conn.execute("PRAGMA table_info(processed)")
        }
        assert {f"has_{tag.name}" for tag in TAGS}.issubset(columns)
    finally:
        cache.close()


def test_mark_inspected_done_updates_stats_and_clears_queues(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        stats = cache.stats()

        assert cache.needs_inspection() == []
        assert cache.needs_work() == []
        assert stats["total"] == 1
        assert stats["done"] == 1
        assert stats["needs_work"] == 0
        assert stats["needs_inspection"] == 0
        assert stats["errors"] == 0
        assert stats["per_tag"] == {tag.name: 1 for tag in TAGS}
    finally:
        cache.close()


def test_mark_inspected_with_missing_tags_queues_file_for_work(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        stats = cache.stats()

        assert cache.needs_work() == [str(filepath)]
        assert stats["done"] == 0
        assert stats["needs_work"] == 1
        assert stats["per_tag"]["bpm"] == 0
    finally:
        cache.close()


def test_mark_error_removes_file_from_worker_queue(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.mark_error(filepath, "worker failed")
        cache.flush()

        stats = cache.stats()

        assert cache.needs_work() == []
        assert stats["needs_work"] == 0
        assert stats["errors"] == 1
    finally:
        cache.close()


def test_mark_changed_resets_completed_file_back_to_inspection(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        filepath.write_bytes(b"updated-audio-data")

        cache.mark_changed(filepath)
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status, has_bpm FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        assert cache.needs_inspection() == [str(filepath)]
        assert row == (None, None)
    finally:
        cache.close()


# ── refresh_stat ──────────────────────────────────────────────────────────────
#
# refresh_stat() must keep the cached mtime/size in sync with the file after
# the worker writes tags to it.  Without this the scanner re-detects the file
# as 'changed' on its next pass (because writing updates the on-disk mtime),
# calls mark_changed(), and resets all has_* to NULL — creating an infinite
# re-inspect/re-work loop when overwrite=True is configured.


def test_refresh_stat_makes_is_unchanged_return_true_after_write(
    tmp_path: Path,
) -> None:
    """Happy path: after refresh_stat() the scanner sees the file as unchanged.

    Simulates the worker writing new content (mtime advances), then calling
    refresh_stat() to update the cache baseline.  is_unchanged() must then
    return True so the scanner does not re-queue the file.
    """
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        # Simulate the worker writing to the file (mtime advances on disk).
        filepath.write_bytes(b"updated-after-tag-write")

        # Before refresh_stat: cache holds the old mtime → scanner fires.
        assert cache.is_unchanged(filepath) is False

        # After refresh_stat: cache baseline matches post-write stat.
        cache.refresh_stat(filepath)
        assert cache.is_unchanged(filepath) is True
    finally:
        cache.close()


def test_refresh_stat_on_deleted_file_does_not_raise(tmp_path: Path) -> None:
    """Unhappy path: file deleted between the write and refresh_stat().

    The stat() call will fail with OSError.  refresh_stat() must swallow it
    silently — the cache keeps its old mtime/size values and the scanner will
    re-detect the file as changed on its next pass, which is the correct safe
    fallback.
    """
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        # Record the mtime/size before deletion so we can verify they are
        # left unchanged after the failed refresh.
        row_before = cache._conn.execute(
            "SELECT mtime, size FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        filepath.unlink()

        # Must not raise even though the file no longer exists.
        cache.refresh_stat(filepath)

        row_after = cache._conn.execute(
            "SELECT mtime, size FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        # Cache row must be untouched — old values preserved.
        assert row_after == row_before
    finally:
        cache.close()


def test_refresh_stat_does_not_touch_tag_columns_or_status(tmp_path: Path) -> None:
    """Guard: refresh_stat() updates only mtime/size, nothing else.

    has_* columns and processing_status must be completely unaffected so that
    the pipeline state is not corrupted by the stat refresh.
    """
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False  # One tag absent to give status='queued'.

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        # Snapshot every column except mtime/size before the refresh.
        row_before = cache._conn.execute(
            "SELECT processing_status, "
            + ", ".join(f"has_{t.name}" for t in TAGS)
            + " FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        # Simulate a write (advances mtime) then refresh the stat.
        filepath.write_bytes(b"post-write-content")
        cache.refresh_stat(filepath)

        row_after = cache._conn.execute(
            "SELECT processing_status, "
            + ", ".join(f"has_{t.name}" for t in TAGS)
            + " FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        # Status and every tag column must be identical before and after.
        assert row_after == row_before
    finally:
        cache.close()


def test_requeue_errors_puts_failed_files_back_in_work_queue(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.mark_error(filepath, "model failed")
        cache.flush()

        # Sanity check: error row is excluded from work queue.
        assert cache.needs_work() == []
        assert cache.stats()["errors"] == 1

        count = cache.requeue_errors()
        cache.flush()

        assert count == 1
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["errors"] == 0
        assert cache.stats()["needs_work"] == 1

        row = cache._conn.execute(
            "SELECT processing_status, last_error FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("queued", None)
    finally:
        cache.close()


def test_requeue_errors_ignores_rows_with_no_missing_tags(tmp_path: Path) -> None:
    """A row in error state but with all tags present should not be requeued."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        # Force the row into error state directly.
        with cache._lock:
            cache._conn.execute(
                "UPDATE processed SET processing_status = 'error' WHERE filepath = ?",
                (str(filepath),),
            )
        cache.flush()

        count = cache.requeue_errors()
        cache.flush()

        assert count == 0
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("error",)
    finally:
        cache.close()


def test_requeue_errors_returns_correct_count_for_multiple_files(
    tmp_path: Path,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    tag_results_missing = _all_tag_results(True)
    tag_results_missing["bpm"] = False

    filepaths = [_make_audio_file(tmp_path, f"track{i}.mp3") for i in range(3)]

    try:
        for fp in filepaths:
            cache.mark_changed(fp)
            cache.mark_inspected(fp, tag_results_missing)
            cache.mark_error(fp, "transient failure")
        cache.flush()

        assert cache.stats()["errors"] == 3
        count = cache.requeue_errors()
        cache.flush()

        assert count == 3
        assert cache.stats()["errors"] == 0
        assert cache.stats()["needs_work"] == 3
    finally:
        cache.close()


# ── requeue_working ────────────────────────────────────────────────────────────


def test_requeue_working_resets_stuck_working_rows_to_queued(tmp_path: Path) -> None:
    """Rows left in 'working' by a prior crash should be reset on startup."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.mark_working(filepath)
        cache.flush()

        # Simulate crash state: 'working' row is excluded from the queue-fetch
        # method so it is not double-dispatched, but IS counted in the display
        # stat so the TUI counter doesn't appear frozen.
        assert cache.needs_work() == []
        assert cache.stats()["needs_work"] == 1

        count = cache.requeue_working()
        cache.flush()

        assert count == 1
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["needs_work"] == 1
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("queued",)
    finally:
        cache.close()


def test_requeue_working_returns_zero_when_no_stuck_rows(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        count = cache.requeue_working()
        assert count == 0
    finally:
        cache.close()


def test_requeue_working_does_not_touch_done_or_error_rows(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    done_fp = _make_audio_file(tmp_path, "done.mp3")
    error_fp = _make_audio_file(tmp_path, "error.mp3")
    tag_results_missing = _all_tag_results(True)
    tag_results_missing["bpm"] = False

    try:
        cache.mark_changed(done_fp)
        cache.mark_inspected(done_fp, _all_tag_results(True))  # status = done

        cache.mark_changed(error_fp)
        cache.mark_inspected(error_fp, tag_results_missing)
        cache.mark_error(error_fp, "some error")
        cache.flush()

        count = cache.requeue_working()
        cache.flush()

        assert count == 0
        assert cache.stats()["done"] == 1
        assert cache.stats()["errors"] == 1
    finally:
        cache.close()


# ── requeue_done_missing_tags ──────────────────────────────────────────────────


def test_requeue_done_missing_tags_resets_incorrectly_completed_rows(
    tmp_path: Path,
) -> None:
    """A 'done' row where a tag column is still 0 should be requeued."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        # Force to 'done' without actually writing the bpm tag — reproduces
        # the bug where process_file() returned {} and mark_done was called.
        cache.mark_done(filepath)
        cache.flush()

        assert cache.stats()["done"] == 1
        assert cache.stats()["needs_work"] == 0

        count = cache.requeue_done_missing_tags()
        cache.flush()

        assert count == 1
        assert cache.stats()["done"] == 0
        assert cache.stats()["needs_work"] == 1
        row = cache._conn.execute(
            "SELECT processing_status, last_error FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row[0] == "queued"
        assert "missing tags" in row[1]
    finally:
        cache.close()


def test_requeue_done_missing_tags_catches_null_has_columns(
    tmp_path: Path,
) -> None:
    """A 'done' row with NULL has_* (worker called plain mark_done) must be caught."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        # Skip mark_inspected — has_* stay NULL — then force to 'done'.
        # This reproduces the scenario where the worker wrote tags but called
        # plain mark_done() instead of mark_done_with_tags().
        cache.mark_done(filepath)
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status, has_bpm FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("done", None)

        count = cache.requeue_done_missing_tags()
        cache.flush()

        assert count == 1
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("queued",)
    finally:
        cache.close()


# ── mark_done_with_tags ────────────────────────────────────────────────────────


def test_mark_done_with_tags_sets_written_columns_to_one(tmp_path: Path) -> None:
    """mark_done_with_tags should set has_* = 1 for every written tag."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False
    tag_results["key"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)  # has_bpm=0, has_key=0
        cache.mark_done_with_tags(filepath, ["bpm", "key"])
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status, has_bpm, has_key, has_mood_happy FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row[0] == "done"
        assert row[1] == 1  # bpm written
        assert row[2] == 1  # key written
        assert row[3] == 1  # mood_happy was already 1, should be untouched
    finally:
        cache.close()


def test_mark_done_with_tags_falls_back_to_mark_done_on_empty_list(
    tmp_path: Path,
) -> None:
    """Calling mark_done_with_tags with no tags behaves like plain mark_done."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.mark_done_with_tags(filepath, [])
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("done",)
    finally:
        cache.close()


def test_mark_done_with_tags_ignores_unknown_tag_names(tmp_path: Path) -> None:
    """Unknown tag names in written_tags should be silently ignored."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        # "nonexistent_tag" is not a valid column — should not raise.
        cache.mark_done_with_tags(filepath, ["bpm", "nonexistent_tag"])
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status, has_bpm FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row[0] == "done"
        assert row[1] == 1
    finally:
        cache.close()


def test_requeue_done_missing_tags_ignores_fully_tagged_done_rows(
    tmp_path: Path,
) -> None:
    """A 'done' row with all tags present must not be touched."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        count = cache.requeue_done_missing_tags()
        assert count == 0
        assert cache.stats()["done"] == 1
    finally:
        cache.close()


def test_requeue_done_missing_tags_respects_enabled_tags_filter(
    tmp_path: Path,
) -> None:
    """With a restricted enabled_tags list only those tags trigger requeue."""
    from musictagger.tags import TAGS

    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    # Mark all tags present except 'key', then force to done.
    tag_results = _all_tag_results(True)
    tag_results["key"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.mark_done(filepath)
        cache.flush()

        # When 'key' is not in the enabled set, the row should NOT be requeued.
        enabled_without_key = [t for t in TAGS if t.name != "key"]
        count = cache.requeue_done_missing_tags(enabled_tags=enabled_without_key)
        assert count == 0

        # When 'key' is included, it should be requeued.
        count = cache.requeue_done_missing_tags(enabled_tags=TAGS)
        assert count == 1
    finally:
        cache.close()


# ── Regression tests: working-status visibility in stats() ────────────────────


def test_stats_counts_working_rows_in_needs_work(tmp_path: Path) -> None:
    """A file in 'working' status must still appear in needs_work so the TUI
    counter decrements smoothly rather than appearing frozen while a batch is
    in flight."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(False)  # all tags absent → needs work

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        # Simulate the worker dequeuing the file: status transitions to 'working'.
        cache.mark_working(filepath)
        cache.flush()

        stats = cache.stats()

        # The file is in 'working' status and has zero tags — it should be
        # counted in needs_work so the TUI shows the correct remaining total.
        assert stats["needs_work"] == 1, (
            "working rows must be included in the needs_work display count"
        )
        # needs_work() (the queue-fetch method) must NOT return the file — it is
        # already being processed and should not be double-dispatched.
        assert cache.needs_work() == [], (
            "needs_work() must not return rows already in 'working' status"
        )
    finally:
        cache.close()


def test_stats_includes_in_progress_count(tmp_path: Path) -> None:
    """stats() must expose an 'in_progress' key counting rows in 'working'
    status so the TUI can display how many files are currently mid-batch."""
    cache = FileCache(tmp_path / "cache.db")
    filepath_a = _make_audio_file(tmp_path, "a.mp3")
    filepath_b = _make_audio_file(tmp_path, "b.mp3")
    tag_results = _all_tag_results(False)

    try:
        cache.mark_changed(filepath_a)
        cache.mark_changed(filepath_b)
        cache.mark_inspected(filepath_a, tag_results)
        cache.mark_inspected(filepath_b, tag_results)

        # Neither file has been picked up yet.
        stats_before = cache.stats()
        assert stats_before["in_progress"] == 0

        # Pick up one file.
        cache.mark_working(filepath_a)
        cache.flush()

        stats_mid = cache.stats()
        assert stats_mid["in_progress"] == 1

        # Finish that file.
        cache.mark_done_with_tags(filepath_a, [t.name for t in TAGS])
        cache.flush()

        stats_after = cache.stats()
        assert stats_after["in_progress"] == 0
    finally:
        cache.close()


def test_stats_in_progress_key_present_in_zeroed_closed_snapshot(
    tmp_path: Path,
) -> None:
    """The zeroed snapshot returned when the connection is closed must include
    the 'in_progress' key so callers never hit a KeyError on shutdown."""
    cache = FileCache(tmp_path / "cache.db")
    cache.close()

    snapshot = cache.stats()
    assert "in_progress" in snapshot
    assert snapshot["in_progress"] == 0
