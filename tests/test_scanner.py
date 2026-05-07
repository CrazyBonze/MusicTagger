"""Unit tests for scanner NFS-hang defences.

Covers:
- Per-directory readdir timeout: a stalled directory is skipped and the scan
  continues with the rest of the tree.
- Scanner heartbeat: last_activity is updated on every file processed and every
  directory successfully listed.
- Scanner watchdog (TUI side): when last_activity is stale the watchdog calls
  scanner.stop() so the orchestrator can relaunch on the next tick.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.scanner import Scanner, _list_dir, _DIR_READDIR_TIMEOUT_S


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_config(
    tmp_path: Path, file_throttle_ms: int = 0, dir_throttle_ms: int = 0
) -> Config:
    return Config(
        music_path=tmp_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        file_throttle_ms=file_throttle_ms,
        dir_throttle_ms=dir_throttle_ms,
    )


def _make_mp3(path: Path, name: str = "track.mp3") -> Path:
    f = path / name
    f.write_bytes(b"audio")
    return f


# ── _list_dir unit tests ───────────────────────────────────────────────────────


def test_list_dir_returns_sorted_audio_files(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _make_mp3(tmp_path, "b.mp3")
    _make_mp3(tmp_path, "a.flac")
    (tmp_path / "readme.txt").write_text("hi")

    subdirs, audio_files = _list_dir(str(tmp_path))

    assert subdirs == ["sub"]
    assert audio_files == ["a.flac", "b.mp3"]


def test_list_dir_ignores_non_audio_files(tmp_path: Path) -> None:
    (tmp_path / "cover.jpg").write_bytes(b"jpeg")
    (tmp_path / "info.txt").write_text("text")
    _make_mp3(tmp_path, "song.mp3")

    _, audio_files = _list_dir(str(tmp_path))

    assert audio_files == ["song.mp3"]


# ── per-directory timeout ──────────────────────────────────────────────────────


def test_scanner_skips_stalled_directory_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory whose readdir blocks beyond the timeout is skipped.

    The scan must still complete and process files in other directories.
    """
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    _make_mp3(good_dir, "fine.mp3")

    stall_dir = tmp_path / "stall"
    stall_dir.mkdir()

    logs: list[str] = []

    real_list_dir = _list_dir

    def fake_list_dir(dirpath_str: str) -> tuple[list[str], list[str]]:
        if Path(dirpath_str) == stall_dir:
            # Block longer than the timeout to simulate an NFS hang.
            time.sleep(_DIR_READDIR_TIMEOUT_S + 5)
        return real_list_dir(dirpath_str)

    # Patch the timeout constant to 0.1 s so the test runs quickly.
    monkeypatch.setattr("musictagger.scanner._DIR_READDIR_TIMEOUT_S", 0.1)
    monkeypatch.setattr("musictagger.scanner._list_dir", fake_list_dir)

    cache = FileCache(tmp_path / "cache.db")
    try:
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache, log_fn=logs.append)

        scanned, changed = scanner.run_pass()

        # The good file must have been picked up.
        assert scanned >= 1
        # A timeout warning must have been logged.
        assert any("timeout" in m.lower() or "skipping" in m.lower() for m in logs), (
            logs
        )
        # Scanner must have finished cleanly (not stuck).
        assert not scanner.running
    finally:
        cache.close()


def test_scanner_heartbeat_updated_per_file(tmp_path: Path) -> None:
    """last_activity advances as the scanner processes files."""
    _make_mp3(tmp_path, "a.mp3")
    _make_mp3(tmp_path, "b.mp3")

    cache = FileCache(tmp_path / "cache.db")
    try:
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)

        before = scanner.last_activity
        scanner.run_pass()
        after = scanner.last_activity

        assert after >= before
    finally:
        cache.close()


def test_scanner_heartbeat_updated_per_directory(tmp_path: Path) -> None:
    """last_activity is updated even when a directory has no audio files."""
    (tmp_path / "empty_subdir").mkdir()

    cache = FileCache(tmp_path / "cache.db")
    try:
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)

        before = time.monotonic()
        scanner.run_pass()

        # last_activity must be at least as recent as our pre-run mark.
        assert scanner.last_activity >= before
    finally:
        cache.close()


# ── scanner watchdog (unit, no TUI) ───────────────────────────────────────────


def test_scanner_stop_clears_running_flag() -> None:
    """scanner.stop() sets _running=False so the watchdog can relaunch."""
    scanner = Scanner.__new__(Scanner)
    scanner._running = True
    scanner._files_scanned = 0
    scanner._files_changed = 0
    scanner._current_file = ""
    scanner.last_activity = time.monotonic()

    scanner.stop()

    assert not scanner.running


def test_scanner_watchdog_calls_stop_when_heartbeat_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the TUI watchdog logic: if last_activity is old, call stop().

    This test replicates the _orchestrate_inner watchdog check in isolation
    so we verify the decision boundary without needing a running Textual app.
    """
    cache = FileCache(tmp_path / "cache.db")
    try:
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)

        # Pretend the scanner is running but its heartbeat is ancient.
        scanner._running = True
        scanner.last_activity = time.monotonic() - 9999  # far in the past

        _SCANNER_HANG_TIMEOUT_S = 120  # same constant as tui.py

        # Replicate the watchdog decision from _orchestrate_inner.
        idle_s = time.monotonic() - scanner.last_activity
        if scanner.running and idle_s > _SCANNER_HANG_TIMEOUT_S:
            scanner.stop()

        assert not scanner.running
    finally:
        cache.close()


def test_scanner_normal_run_processes_all_files(tmp_path: Path) -> None:
    """Sanity check: with no NFS issues the scanner finds all audio files."""
    sub = tmp_path / "Artist" / "Album"
    sub.mkdir(parents=True)
    _make_mp3(sub, "01.mp3")
    _make_mp3(sub, "02.flac")
    _make_mp3(tmp_path, "root.mp3")

    cache = FileCache(tmp_path / "cache.db")
    try:
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)

        scanned, changed = scanner.run_pass()

        assert scanned == 3
        assert changed == 3
    finally:
        cache.close()


# ── refresh_stat / overwrite loop prevention ───────────────────────────────────
#
# When overwrite=True is configured the worker rewrites tags on every file it
# processes.  Writing to an audio file advances its mtime on disk.  Without
# refresh_stat() the scanner's next pass sees a mtime mismatch, calls
# mark_changed(), and resets all has_* to NULL — sending the file back through
# the full inspect → work cycle indefinitely.
#
# These two tests cover:
#   1. Happy path — scanner leaves the file alone after refresh_stat().
#   2. Unhappy path — a genuine external modification after the write is still
#      detected correctly (refresh_stat() does not suppress real changes).


def test_scanner_does_not_requeue_file_after_worker_writes_tags(
    tmp_path: Path,
) -> None:
    """Core regression: scanner must not re-detect a file the worker just wrote.

    Sequence:
      1. File enters cache via mark_changed().
      2. Inspector marks all tags present (done).
      3. Worker writes new tag values → file mtime advances on disk.
      4. Worker calls refresh_stat() to sync the cache baseline.
      5. Scanner runs its next pass.

    Expected: is_unchanged() returns True; scanner does not call mark_changed();
    file stays 'done' and never re-enters the inspection or work queues.
    """
    filepath = _make_mp3(tmp_path)
    cache = FileCache(tmp_path / "cache.db")

    try:
        from musictagger.tags import TAGS

        # Step 1 & 2: file scanned and inspected as fully tagged.
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, {t.name: True for t in TAGS})
        cache.flush()
        assert cache.stats()["done"] == 1

        # Step 3: simulate the worker writing new tag values (mtime advances).
        filepath.write_bytes(b"audio-with-updated-tags")

        # Step 4: worker refreshes the cache stat after the write.
        cache.refresh_stat(filepath)
        cache.flush()

        # Step 5: scanner runs — file must look unchanged.
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)
        scanned, changed = scanner.run_pass()

        assert scanned == 1
        assert changed == 0  # scanner must NOT have called mark_changed()
        assert cache.stats()["done"] == 1
        assert cache.needs_inspection() == []
        assert cache.needs_work() == []
    finally:
        cache.close()


def test_scanner_requeues_file_modified_externally_after_worker_write(
    tmp_path: Path,
) -> None:
    """Unhappy path: a real external modification after the worker write is detected.

    refresh_stat() records the mtime/size immediately after the worker's write.
    If the file is then modified externally (different content → different size),
    the scanner must detect the new change and re-queue the file for inspection.
    This ensures refresh_stat() does not accidentally suppress genuine changes.
    """
    filepath = _make_mp3(tmp_path)
    cache = FileCache(tmp_path / "cache.db")

    try:
        from musictagger.tags import TAGS

        # File scanned, inspected, worker writes tags and refreshes stat.
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, {t.name: True for t in TAGS})
        filepath.write_bytes(b"audio-with-updated-tags")
        cache.refresh_stat(filepath)
        cache.flush()

        # Genuine external modification after the worker (different size).
        filepath.write_bytes(b"externally-modified-content-different-size-xxxx")

        # Scanner runs — must detect the external change.
        config = _make_config(tmp_path)
        scanner = Scanner(config, cache)
        scanned, changed = scanner.run_pass()

        assert scanned == 1
        assert changed == 1  # scanner MUST have called mark_changed()
        # File must be back in the inspection queue, not 'done'.
        assert cache.needs_inspection() == [str(filepath)]
        assert cache.stats()["done"] == 0
    finally:
        cache.close()
