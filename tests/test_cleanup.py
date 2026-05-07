"""Unit tests for orphaned cache entry cleanup."""

from __future__ import annotations

from pathlib import Path

from musictagger.cache import FileCache
from musictagger.cleanup import Cleanup
from musictagger.config import Config


def _make_config(tmp_path: Path) -> Config:
    return Config(
        music_path=tmp_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
    )


def _make_audio_file(tmp_path: Path, name: str) -> Path:
    filepath = tmp_path / name
    filepath.write_bytes(b"audio")
    return filepath


def test_cleanup_removes_only_missing_cached_files(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    config = _make_config(tmp_path)
    existing = _make_audio_file(tmp_path, "existing.mp3")
    deleted = _make_audio_file(tmp_path, "deleted.mp3")

    try:
        cache.mark_changed(existing)
        cache.mark_changed(deleted)
        deleted.unlink()

        cleanup = Cleanup(config, cache)

        assert cleanup.run() == 1
        assert cleanup.last_removed == 1
        assert cache.all_filepaths() == [str(existing)]
    finally:
        cache.close()


def test_cleanup_noops_when_all_cached_files_exist(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    config = _make_config(tmp_path)
    existing = _make_audio_file(tmp_path, "track.mp3")

    try:
        cache.mark_changed(existing)

        cleanup = Cleanup(config, cache)

        assert cleanup.run() == 0
        assert cleanup.last_removed == 0
        assert cache.all_filepaths() == [str(existing)]
    finally:
        cache.close()


# ── Cleanup.stop() / threading.Event ─────────────────────────────────────────


def test_cleanup_stop_sets_stop_event() -> None:
    """stop() must set the threading.Event so the check loop exits cleanly."""
    from musictagger.cleanup import Cleanup

    cleanup = Cleanup.__new__(Cleanup)
    cleanup._running = False
    cleanup._stop_event = __import__("threading").Event()

    assert not cleanup._stop_event.is_set()
    cleanup.stop()
    assert cleanup._stop_event.is_set()


def test_cleanup_reset_clears_stop_event() -> None:
    """reset() must clear the stop event so cleanup can be relaunched."""
    from musictagger.cleanup import Cleanup

    cleanup = Cleanup.__new__(Cleanup)
    cleanup._stop_event = __import__("threading").Event()
    cleanup._stop_event.set()

    cleanup.reset()
    assert not cleanup._stop_event.is_set()
