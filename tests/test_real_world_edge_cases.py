"""Focused edge-case tests for messy real-world libraries."""

from __future__ import annotations

from pathlib import Path

import pytest

import musictagger.inspector as inspector_module
from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.inspector import Inspector
from musictagger.scanner import Scanner
from musictagger.tags import TAGS

FULL_TAG_PAYLOAD = {
    "TBPM": ["120"],
    "TXXX:MOOD_HAPPY": ["1"],
    "TXXX:MOOD_SAD": ["1"],
    "TXXX:MOOD_RELAXED": ["1"],
    "TXXX:MOOD_AGGRESSIVE": ["1"],
    "TXXX:MOOD_PARTY": ["1"],
    "MOOD_DANCEABILITY": ["1"],
    "TMOO": ["Calm; Happy"],
    "TXXX:THEME": ["Travel; Nature"],
    "TXXX:ELECTRONIC": ["1"],
    "TXXX:ACOUSTIC": ["1"],
    "TXXX:INSTRUMENTAL": ["1"],
    "TIMBRE_BRIGHTNESS": ["1"],
    "TONALITY": ["1"],
    "TKEY": ["C major"],
}


def _make_config(tmp_path: Path, *, music_path: Path | None = None) -> Config:
    return Config(
        music_path=music_path or tmp_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        file_throttle_ms=0,
        dir_throttle_ms=0,
        inspector_throttle_ms=0,
        inspector_batch_size=20,
    )


def _make_file(path: Path, data: bytes = b"audio") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_scanner_handles_mixed_case_hidden_and_odd_filenames(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache.db")
    music_path = tmp_path / "library"
    expected_files = [
        _make_file(music_path / ".hidden.FLAC"),
        _make_file(music_path / "Artist" / "Album" / "01 - Artist's [Mix], Pt. 1.Mp3"),
        _make_file(music_path / "Artist" / "Album" / "02 - afterhours.m4A"),
    ]
    _make_file(music_path / "Artist" / "Album" / "cover.jpg", b"jpg")
    _make_file(music_path / "Artist" / "Album" / "track.bin", b"not-audio")

    try:
        scanner = Scanner(_make_config(tmp_path, music_path=music_path), cache)

        assert scanner.run_pass() == (len(expected_files), len(expected_files))
        assert set(cache.all_filepaths()) == {str(path) for path in expected_files}
    finally:
        cache.close()


def test_inspector_queues_file_with_only_partial_tags_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_file(tmp_path / "partial.mp3")

    try:
        cache.mark_changed(filepath)
        monkeypatch.setattr(
            inspector_module.mutagen,
            "File",
            lambda *_args, **_kwargs: {
                "TBPM": ["120"],
                "TXXX:MOOD_HAPPY": ["1"],
            },
        )

        inspector = Inspector(_make_config(tmp_path), cache)
        stats_before = cache.stats()

        assert inspector.run_pass() == 1

        stats_after = cache.stats()
        assert inspector.queued == 1
        assert inspector.errors == 0
        assert stats_before["needs_inspection"] == 1
        assert stats_after["needs_inspection"] == 0
        assert stats_after["needs_work"] == 1
        assert stats_after["done"] == 0
        assert stats_after["per_tag"]["bpm"] == 1
        assert stats_after["per_tag"]["mood_happy"] == 1
        assert stats_after["per_tag"]["tonality"] == 0
    finally:
        cache.close()


def test_inspector_treats_empty_tag_values_as_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_file(tmp_path / "empty-values.mp3")

    try:
        cache.mark_changed(filepath)
        monkeypatch.setattr(
            inspector_module.mutagen,
            "File",
            lambda *_args, **_kwargs: {
                "TBPM": "",
                "TXXX:MOOD_HAPPY": [],
                "TONALITY": None,
            },
        )

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 1
        assert inspector.errors == 0
        assert cache.needs_work() == [str(filepath)]
        assert cache.stats()["per_tag"] == {tag.name: 0 for tag in TAGS}
    finally:
        cache.close()


def test_inspector_handles_zero_byte_files_as_unrecognised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_file(tmp_path / "empty.mp3", b"")

    try:
        cache.mark_changed(filepath)

        def _fake_mutagen_file(filepath_str: str, easy: bool = False) -> object:
            del easy
            if Path(filepath_str).stat().st_size == 0:
                return None
            return dict(FULL_TAG_PAYLOAD)

        monkeypatch.setattr(inspector_module.mutagen, "File", _fake_mutagen_file)

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 0
        assert inspector.errors == 0
        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(filepath)]
    finally:
        cache.close()


def test_inspector_continues_past_corrupted_file_in_mixed_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    good = _make_file(tmp_path / "good.mp3")
    bad = _make_file(tmp_path / "bad.mp3")
    partial = _make_file(tmp_path / "partial.mp3")

    try:
        for filepath in [good, bad, partial]:
            cache.mark_changed(filepath)

        def _fake_mutagen_file(filepath_str: str, easy: bool = False) -> object:
            del easy
            name = Path(filepath_str).name
            if name == "good.mp3":
                return dict(FULL_TAG_PAYLOAD)
            if name == "bad.mp3":
                raise RuntimeError("corrupted file")
            return {"TBPM": ["120"]}

        monkeypatch.setattr(inspector_module.mutagen, "File", _fake_mutagen_file)

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 2
        assert inspector.errors == 1
        # bad.mp3 must be in error state, not stuck in the inspection queue
        assert cache.needs_inspection() == []
        assert cache.needs_work() == [str(partial)]
        assert cache.stats()["done"] == 1
        assert cache.stats()["errors"] == 1
    finally:
        cache.close()


def test_inspector_survives_one_tag_checker_failing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_file(tmp_path / "tag-checker-error.mp3")
    original_check_fn = TAGS[0].check_fn

    try:
        cache.mark_changed(filepath)
        monkeypatch.setattr(
            inspector_module.mutagen,
            "File",
            lambda *_args, **_kwargs: dict(FULL_TAG_PAYLOAD),
        )

        object.__setattr__(
            TAGS[0],
            "check_fn",
            lambda _f: (_ for _ in ()).throw(RuntimeError("bad tag check")),
        )

        inspector = Inspector(_make_config(tmp_path), cache)

        assert inspector.run_pass() == 1
        assert inspector.errors == 0
        assert inspector.queued == 1
        assert cache.stats()["per_tag"][TAGS[0].name] == 0
        assert cache.needs_work() == [str(filepath)]
    finally:
        object.__setattr__(TAGS[0], "check_fn", original_check_fn)
        cache.close()


def test_mark_changed_resets_partial_results_for_reinspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = FileCache(tmp_path / "cache.db")
    filepath = _make_file(tmp_path / "rescan-me.mp3")

    try:
        cache.mark_changed(filepath)
        monkeypatch.setattr(
            inspector_module.mutagen,
            "File",
            lambda *_args, **_kwargs: {"TBPM": ["120"]},
        )

        inspector = Inspector(_make_config(tmp_path), cache)
        assert inspector.run_pass() == 1
        assert cache.needs_work() == [str(filepath)]

        filepath.write_bytes(b"changed")
        cache.mark_changed(filepath)
        cache.flush()

        row = cache._conn.execute(
            "SELECT processing_status, has_bpm, has_tonality FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()

        assert cache.needs_inspection() == [str(filepath)]
        assert cache.needs_work() == []
        assert row == (None, None, None)
    finally:
        cache.close()
