"""Tests for per-tag enabled/overwrite configuration.

Covers:
  - Config.load() parsing [tags.*] TOML sections
  - Config.tag_cfg() fallback to defaults
  - cache.needs_work() respects enabled_tags parameter
  - Inspector skips disabled tags (leaves has_* as NULL)
  - Worker._missing_tags() skips disabled tags and honours overwrite flag
"""

from __future__ import annotations

from pathlib import Path


import pytest

import musictagger.inspector as inspector_module
from musictagger.cache import FileCache
from musictagger.config import Config, TagConfig
from musictagger.inspector import Inspector
from musictagger.tags import TAGS


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_config(
    tmp_path: Path,
    tag_configs: dict[str, TagConfig] | None = None,
) -> Config:
    return Config(
        music_path=tmp_path,
        db_path=tmp_path / "cache.db",
        embeddings_db_path=tmp_path / "embeddings.db",
        log_path=tmp_path / "musictagger.log",
        inspector_throttle_ms=0,
        inspector_batch_size=10,
        tag_configs=tag_configs or {},
    )


def _all_tag_results(value: bool) -> dict[str, bool]:
    return {tag.name: value for tag in TAGS}


def _make_audio_file(tmp_path: Path, name: str = "track.mp3") -> Path:
    filepath = tmp_path / name
    filepath.write_bytes(b"audio")
    return filepath


# ── Config loading ─────────────────────────────────────────────────────────────


def test_config_load_parses_tags_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'music_path = "/music"',
                "[tags.bpm]",
                "enabled = true",
                "overwrite = true",
                "[tags.mood_happy]",
                "enabled = false",
                "overwrite = false",
            ]
        )
    )

    config = Config.load(config_path=config_path)

    assert config.tag_cfg("bpm") == TagConfig(enabled=True, overwrite=True)
    assert config.tag_cfg("mood_happy") == TagConfig(enabled=False, overwrite=False)


def test_config_tag_cfg_defaults_for_unlisted_tag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('music_path = "/music"\n')

    config = Config.load(config_path=config_path)

    assert config.tag_cfg("bpm") == TagConfig(enabled=True, overwrite=False)
    assert config.tag_cfg("tonality") == TagConfig(enabled=True, overwrite=False)


def test_config_tag_cfg_partial_section_uses_defaults(tmp_path: Path) -> None:
    """A section with only one key uses the default for the other."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'music_path = "/music"',
                "[tags.bpm]",
                "overwrite = true",
                # enabled not specified — should default to True
            ]
        )
    )

    config = Config.load(config_path=config_path)

    assert config.tag_cfg("bpm") == TagConfig(enabled=True, overwrite=True)


# ── cache.stats() with enabled_tags ───────────────────────────────────────────


def test_stats_does_not_overcount_needs_inspection_for_disabled_tags(
    tmp_path: Path,
) -> None:
    """Regression: stats() must not count disabled-tag NULL columns as needing
    inspection, or the TUI orchestrator will spin the inspector forever."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        # Simulate inspector finishing with bpm disabled: only non-bpm tags written.
        results_without_bpm = {t.name: True for t in TAGS if t.name != "bpm"}
        cache.mark_inspected(filepath, results_without_bpm)
        cache.flush()

        enabled_tags = [t for t in TAGS if t.name != "bpm"]

        # With all tags: wrongly counts the file as needing inspection.
        assert cache.stats()["needs_inspection"] == 1

        # With bpm excluded: correctly reports zero.
        assert cache.stats(enabled_tags=enabled_tags)["needs_inspection"] == 0
    finally:
        cache.close()


def test_stats_does_not_overcount_needs_work_for_disabled_tags(
    tmp_path: Path,
) -> None:
    """Regression: stats() must not count disabled-tag zero columns in the
    needs_work figure, or the TUI orchestrator will spin the worker forever."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    # All tags present except bpm (which is disabled and absent)
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        enabled_tags = [t for t in TAGS if t.name != "bpm"]

        # With all tags: wrongly counts the file as needing work.
        assert cache.stats()["needs_work"] == 1

        # With bpm excluded: correctly reports zero.
        assert cache.stats(enabled_tags=enabled_tags)["needs_work"] == 0
    finally:
        cache.close()


# ── cache.needs_work() with enabled_tags ──────────────────────────────────────


def test_needs_work_respects_enabled_tags_filter(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    # bpm absent, everything else present
    tag_results = _all_tag_results(True)
    tag_results["bpm"] = False

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        bpm_tag = next(t for t in TAGS if t.name == "bpm")
        other_tags = [t for t in TAGS if t.name != "bpm"]

        # When only non-bpm tags are enabled, this file should NOT need work
        assert cache.needs_work(enabled_tags=other_tags) == []

        # When bpm is among enabled tags, it should
        assert cache.needs_work(enabled_tags=[bpm_tag]) == [str(filepath)]
    finally:
        cache.close()


# ── cache.needs_inspection() with enabled_tags ────────────────────────────────


def test_needs_inspection_ignores_disabled_tag_null_columns(tmp_path: Path) -> None:
    """Regression: disabled tags stay NULL, so needs_inspection must not flag them.

    Before the fix, a file with any disabled tag (has_* = NULL) would perpetually
    appear in needs_inspection even after the inspector had finished with it,
    causing an infinite inspection loop.
    """
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    try:
        cache.mark_changed(filepath)
        # Simulate inspector running with bpm disabled: only non-bpm tags written.
        results_without_bpm = {t.name: True for t in TAGS if t.name != "bpm"}
        cache.mark_inspected(filepath, results_without_bpm)
        cache.flush()

        # has_bpm is still NULL (never written).
        # With all tags: file appears to need inspection.
        assert cache.needs_inspection() == [str(filepath)]

        # With bpm excluded (disabled): file should NOT need inspection.
        enabled_tags = [t for t in TAGS if t.name != "bpm"]
        assert cache.needs_inspection(enabled_tags=enabled_tags) == []
    finally:
        cache.close()


def test_inspector_does_not_loop_when_a_tag_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: inspector must not re-queue a file indefinitely because a
    disabled tag left its has_* column as NULL."""
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    # Fake mutagen — all tags would be present if checked.
    fake_file: dict[str, list[str]] = {"TBPM": ["120"]}
    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: fake_file)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=False, overwrite=False)},
    )

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(config, cache)

        # First pass: inspector runs, skips bpm (disabled), marks rest.
        count = inspector.run_pass()
        cache.flush()
        assert count == 1

        # Second pass: no files should need inspection — the loop is broken.
        count = inspector.run_pass()
        assert count == 0
    finally:
        cache.close()


# ── Inspector: disabled tags stay NULL ────────────────────────────────────────


def test_inspector_skips_disabled_tags_leaving_them_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    # Fake mutagen so all check_fns would return True if called
    fake_file: dict[str, list[str]] = {
        "TBPM": ["120"],
        "TXXX:MOOD_HAPPY": ["0.9"],
    }
    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: fake_file)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=False, overwrite=False)},
    )

    try:
        cache.mark_changed(filepath)
        inspector = Inspector(config, cache)
        inspector.run_pass()
        cache.flush()

        row = cache._conn.execute(
            "SELECT has_bpm FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        # bpm was disabled — column should still be NULL
        assert row[0] is None
    finally:
        cache.close()


def test_inspector_processes_enabled_tags_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    fake_file: dict[str, list[str]] = {"TBPM": ["120"]}
    monkeypatch.setattr(inspector_module.mutagen, "File", lambda *a, **kw: fake_file)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=True, overwrite=False)},
    )

    try:
        cache.mark_changed(filepath)
        Inspector(config, cache).run_pass()
        cache.flush()

        row = cache._conn.execute(
            "SELECT has_bpm FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row[0] == 1
    finally:
        cache.close()


# ── Worker._missing_tags() ─────────────────────────────────────────────────────


def _make_worker(config: Config, cache: FileCache):
    """Import lazily to avoid heavyweight worker imports at module load."""
    from musictagger.worker import Worker

    return Worker(config, cache)


def test_worker_missing_tags_excludes_disabled_tags(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    # All tags absent
    tag_results = _all_tag_results(False)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=False, overwrite=False)},
    )

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        worker = _make_worker(config, cache)
        missing = worker._missing_tags(str(filepath))

        assert "bpm" not in missing
        worker.close()
    finally:
        cache.close()


def test_worker_missing_tags_includes_overwrite_tags_even_when_present(
    tmp_path: Path,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    # All tags present — bpm = 1
    tag_results = _all_tag_results(True)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=True, overwrite=True)},
    )

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        worker = _make_worker(config, cache)
        missing = worker._missing_tags(str(filepath))

        assert "bpm" in missing
        worker.close()
    finally:
        cache.close()


def test_worker_missing_tags_normal_tag_not_included_when_present(
    tmp_path: Path,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)
    tag_results = _all_tag_results(True)

    config = _make_config(tmp_path)  # no overrides

    try:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results)
        cache.flush()

        worker = _make_worker(config, cache)
        missing = worker._missing_tags(str(filepath))

        assert missing == []
        worker.close()
    finally:
        cache.close()


# ── overwrite=True does not re-queue unchanged files ──────────────────────────


def test_overwrite_tag_does_not_requeue_unchanged_done_file(tmp_path: Path) -> None:
    """Regression: a fully-tagged file must not be re-queued just because its
    tag has overwrite=True, if the file has not changed on disk.

    The old startup reset (_apply_overwrite_resets) would flip has_*=1 back to
    0 on every launch, causing the entire library to be reprocessed even for
    unchanged files.  After removing that reset, a done file with overwrite=True
    stays out of the work queue until the scanner detects an mtime/size change.
    """
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_audio_file(tmp_path)

    config = _make_config(
        tmp_path,
        tag_configs={"bpm": TagConfig(enabled=True, overwrite=True)},
    )

    try:
        # Simulate a completed first run: file inspected and all tags written.
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, _all_tag_results(True))
        cache.flush()

        # File is done — should not be in the work queue.
        assert cache.needs_work() == []

        # Simulate a second startup with overwrite=True still set.
        # Previously _apply_overwrite_resets() would flip has_bpm back to 0 here.
        # Now there is no such reset, so the file stays done.
        enabled_tags = [t for t in TAGS if config.tag_cfg(t.name).enabled]
        assert cache.needs_work(enabled_tags=enabled_tags) == []
        assert cache.needs_inspection(enabled_tags=enabled_tags) == []
    finally:
        cache.close()
