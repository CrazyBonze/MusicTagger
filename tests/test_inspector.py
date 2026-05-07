"""Unit tests for inspector edge cases.

Overwrite behaviour (scenarios reference the matrix in the design docs):
  overwrite=False, all tags present   → marked done, not queued        (sc.1)
  overwrite=True,  all tags present   → queued for worker to rewrite   (sc.2)
  overwrite=False, some tags missing  → queued normally                (sc.3)
  overwrite=True,  some tags missing  → queued (missing + overwrite)   (sc.4)
  overwrite=True,  cache already done → NOT re-inspected, stays done   (sc.6)
  overwrite=True,  file modified, all tags present → queued again      (sc.8)
"""

from __future__ import annotations

from pathlib import Path

import pytest

import musictagger.inspector as inspector_module
from musictagger.cache import FileCache
from musictagger.config import Config, TagConfig
from musictagger.inspector import Inspector
from musictagger.tags import TAGS


def _make_config(tmp_path: Path, tag_configs: dict | None = None) -> Config:
    return Config(
        music_path=tmp_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        inspector_throttle_ms=0,
        inspector_batch_size=10,
        tag_configs=tag_configs or {},
    )


def _make_audio_file(tmp_path: Path, name: str = "track.mp3") -> Path:
    filepath = tmp_path / name
    filepath.write_bytes(b"audio")
    return filepath


def test_inspector_marks_missing_file_as_needing_work(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        filepath.unlink()

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 0
        assert inspector.errors == 0
        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["done"] == 0
    finally:
        cache.close()


# ── Inspector.stop() / threading.Event ───────────────────────────────────────


def test_inspector_stop_sets_stop_event() -> None:
    """stop() must set the threading.Event so the file loop exits cleanly."""
    import musictagger.inspector as inspector_mod

    inspector = inspector_mod.Inspector.__new__(inspector_mod.Inspector)
    inspector._stop_event = __import__("threading").Event()
    inspector._running = False

    assert not inspector._stop_event.is_set()
    inspector.stop()
    assert inspector._stop_event.is_set()


def test_inspector_reset_clears_stop_event() -> None:
    """reset() must clear the stop event so the inspector can be relaunched."""
    import musictagger.inspector as inspector_mod

    inspector = inspector_mod.Inspector.__new__(inspector_mod.Inspector)
    inspector._stop_event = __import__("threading").Event()
    inspector._stop_event.set()

    inspector.reset()
    assert not inspector._stop_event.is_set()


# ── Inspector.run() — drains the queue across multiple passes ─────────────────


def test_inspector_run_loops_until_queue_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspector.run() must call run_pass() repeatedly until it returns 0.

    This is the counterpart to test_worker_loop_continues_across_batches:
    the inspector must not stop after one batch when there is more work.
    """
    import musictagger.inspector as inspector_mod

    cache = FileCache(tmp_path / "cache.db")
    config = _make_config(tmp_path)
    inspector = inspector_mod.Inspector(config, cache)

    return_values = [3, 3, 3, 0]
    call_count = [0]

    def _fake_run_pass(self: object) -> int:
        result = return_values[call_count[0]]
        call_count[0] += 1
        inspector._running = False  # mirrors real run_pass() behaviour
        return result

    monkeypatch.setattr(inspector_mod.Inspector, "run_pass", _fake_run_pass)

    inspector.run()

    assert call_count[0] == len(return_values), (
        f"run_pass() called {call_count[0]} time(s); expected {len(return_values)}. "
        "Inspector.run() is not looping until the queue is empty."
    )
    cache.close()


def test_inspector_run_stops_when_stop_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inspector.run() must exit cleanly when stop() is called between passes."""
    import musictagger.inspector as inspector_mod

    cache = FileCache(tmp_path / "cache.db")
    config = _make_config(tmp_path)
    inspector = inspector_mod.Inspector(config, cache)

    call_count = [0]

    def _fake_run_pass(self: object) -> int:
        call_count[0] += 1
        inspector.stop()
        inspector._running = False
        return 3  # non-zero — queue not empty

    monkeypatch.setattr(inspector_mod.Inspector, "run_pass", _fake_run_pass)

    inspector.run()

    assert call_count[0] == 1, (
        f"run_pass() called {call_count[0]} time(s); expected 1. "
        "Inspector.run() did not respect stop()."
    )
    cache.close()


def test_inspector_marks_unrecognised_format_as_needing_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        monkeypatch.setattr(
            inspector_module.mutagen, "File", lambda *_args, **_kwargs: None
        )

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 0
        assert inspector.errors == 0
        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
    finally:
        cache.close()


def test_inspector_records_read_errors_and_marks_file_as_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file that raises during mutagen.File() must be marked error, not left
    with NULL has_* columns.  Leaving it NULL caused an infinite re-inspection
    loop where needs_inspection() returned the same file on every pass."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(inspector_module.mutagen, "File", _raise)

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 0
        assert inspector.errors == 1
        # File must be out of the inspection queue — not returned by needs_inspection()
        assert cache.needs_inspection() == []
        # File must not be in the work queue either
        assert cache.needs_work() == []
        # File must be in error state so the user can see it and requeue manually
        stats = cache.stats()
        assert stats["errors"] == 1
    finally:
        cache.close()


# ── Overwrite behaviour ────────────────────────────────────────────────────────
#
# The inspector is the single point where overwrite=True is enforced.  It
# records overwrite-enabled tags as absent (has_*=0) so mark_inspected() sets
# processing_status='queued' and needs_work() returns the file to the worker.
#
# Overwrite only fires during inspection.  Files whose cache row is already
# 'done' are not re-inspected (the scanner did not detect a change), so they
# are never rewritten unnecessarily.


def _fake_mutagen_all_present(
    monkeypatch: pytest.MonkeyPatch,
    inspector_module: object,
) -> None:
    """Patch mutagen.File so every tag check_fn returns True."""

    class _FakeFile:
        def __getitem__(self, key: str) -> list[str]:
            return ["value"]

        def __contains__(self, key: str) -> bool:
            return True

    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: _FakeFile())


def test_overwrite_false_all_tags_present_marks_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 1 — overwrite=False, all tags present → done, not queued."""
    _fake_mutagen_all_present(monkeypatch, inspector_module)
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(_make_config(tmp_path), cache)
        assert inspector.run_pass() == 1

        assert cache.needs_inspection() == []
        assert cache.needs_work() == []
        assert cache.stats()["done"] == 1
    finally:
        cache.close()


def test_overwrite_true_all_tags_present_queues_for_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 2 — overwrite=True, all tags present → queued so worker rewrites.

    This is the primary bug fix: previously such files were silently marked
    'done' and the worker never saw them.
    """
    _fake_mutagen_all_present(monkeypatch, inspector_module)

    # Enable overwrite for every tag.
    tag_configs = {tag.name: TagConfig(enabled=True, overwrite=True) for tag in TAGS}
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(_make_config(tmp_path, tag_configs=tag_configs), cache)
        assert inspector.run_pass() == 1

        # File must be in the work queue, not marked done.
        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["done"] == 0
        # has_* columns must all be 0 (treated as absent) so needs_work() finds them.
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row[0] == "queued"
    finally:
        cache.close()


def test_overwrite_false_some_tags_missing_queues_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 3 — overwrite=False, some tags missing → queued as normal."""

    class _FakeFile:
        """Only the first tag is present; all others are absent."""

        def __getitem__(self, key: str) -> list[str]:
            if key in ("TBPM", "bpm", "BPM", "tmpo"):
                return ["120"]
            raise KeyError(key)

        def __contains__(self, key: str) -> bool:
            return key in ("TBPM", "bpm", "BPM", "tmpo")

    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: _FakeFile())
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(_make_config(tmp_path), cache)
        assert inspector.run_pass() == 1

        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["done"] == 0
    finally:
        cache.close()


def test_overwrite_true_some_tags_missing_queues_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 4 — overwrite=True, some tags missing → queued (missing + overwrite tags)."""

    class _FakeFile:
        """Only bpm is present; everything else is absent."""

        def __getitem__(self, key: str) -> list[str]:
            if key in ("TBPM", "bpm", "BPM", "tmpo"):
                return ["120"]
            raise KeyError(key)

        def __contains__(self, key: str) -> bool:
            return key in ("TBPM", "bpm", "BPM", "tmpo")

    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: _FakeFile())

    # Enable overwrite for every tag (including bpm which is present).
    tag_configs = {tag.name: TagConfig(enabled=True, overwrite=True) for tag in TAGS}
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(_make_config(tmp_path, tag_configs=tag_configs), cache)
        assert inspector.run_pass() == 1

        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
        # bpm is present in the file but overwrite=True forces has_bpm=0.
        bpm_val = cache._conn.execute(
            "SELECT has_bpm FROM processed WHERE filepath = ?", (str(filepath),)
        ).fetchone()[0]
        assert bpm_val == 0
    finally:
        cache.close()


def test_overwrite_true_done_file_not_requeued_without_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 6 — overwrite=True, cache already 'done', file unchanged → stays done.

    The scanner detects no mtime/size change so mark_changed() is never called,
    the inspector is never triggered, and the file remains 'done' in the cache.
    This prevents unnecessary work on unmodified files.
    """
    _fake_mutagen_all_present(monkeypatch, inspector_module)

    tag_configs = {tag.name: TagConfig(enabled=True, overwrite=True) for tag in TAGS}
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        # Simulate a file that was already fully processed in a prior run.
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, {tag.name: True for tag in TAGS})
        cache.flush()

        # Sanity-check: the file is 'done' and not in any queue.
        assert cache.needs_inspection() == []
        assert cache.needs_work() == []
        assert cache.stats()["done"] == 1

        # Run the inspector — it should find nothing to do (scanner did not
        # call mark_changed() because the file is unchanged).
        inspector = Inspector(_make_config(tmp_path, tag_configs=tag_configs), cache)
        assert inspector.run_pass() == 0

        # File must still be 'done'.
        assert cache.needs_inspection() == []
        assert cache.needs_work() == []
        assert cache.stats()["done"] == 1
    finally:
        cache.close()


def test_overwrite_true_modified_file_all_tags_present_requeued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scenario 8 — overwrite=True, file modified but all tags survive → requeued.

    When the scanner detects an mtime/size change it calls mark_changed(),
    which resets all has_* to NULL.  The inspector then re-opens the file and,
    because overwrite=True, records the tags as absent so the worker rewrites them.
    """
    _fake_mutagen_all_present(monkeypatch, inspector_module)

    tag_configs = {tag.name: TagConfig(enabled=True, overwrite=True) for tag in TAGS}
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        # First pass: file processed and marked 'done'.
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, {tag.name: True for tag in TAGS})
        cache.flush()
        assert cache.stats()["done"] == 1

        # Simulate the scanner detecting a file modification: it calls
        # mark_changed() which resets has_* to NULL and clears processing_status.
        cache.mark_changed(filepath)
        cache.flush()

        assert cache.needs_inspection() == [str(filepath)]

        # Inspector re-opens the file; overwrite forces all tags to absent.
        inspector = Inspector(_make_config(tmp_path, tag_configs=tag_configs), cache)
        assert inspector.run_pass() == 1

        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["done"] == 0
    finally:
        cache.close()
