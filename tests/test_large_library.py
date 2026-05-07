"""Tests that model messier real-world and larger-library scenarios."""

from __future__ import annotations

import time
from pathlib import Path

import musictagger.inspector as inspector_module

from musictagger.cache import FileCache
from musictagger.cleanup import Cleanup
from musictagger.config import Config
from musictagger.inspector import Inspector
from musictagger.scanner import Scanner

FULL_TAG_PAYLOAD = {
    "TBPM": ["120"],
    "TXXX:MOOD_HAPPY": ["1"],
    "TXXX:MOOD_SAD": ["1"],
    "TXXX:MOOD_RELAXED": ["1"],
    "TXXX:MOOD_AGGRESSIVE": ["1"],
    "TXXX:MOOD_PARTY": ["1"],
    "MOOD_DANCEABILITY": ["1"],
    "TMOO": ["happy"],
    "TXXX:THEME": ["film"],
    "TXXX:ELECTRONIC": ["1"],
    "TXXX:ACOUSTIC": ["1"],
    "TXXX:INSTRUMENTAL": ["1"],
    "TIMBRE_BRIGHTNESS": ["1"],
    "TONALITY": ["1"],
    "TKEY": ["C major"],
}


def _make_config(tmp_path: Path, *, inspector_batch_size: int = 100) -> Config:
    return Config(
        music_path=tmp_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        file_throttle_ms=0,
        dir_throttle_ms=0,
        inspector_throttle_ms=0,
        inspector_batch_size=inspector_batch_size,
    )


def _create_large_library(root: Path) -> list[Path]:
    albums = [
        ("Artist 001", "Album One"),
        ("Artist 002", "Artist's Mix [Live]"),
        ("Various Artists", "Compilation 2024 Disc 1"),
        ("DJ Example", "Singles and Remixes"),
    ]
    suffixes = [".mp3", ".flac", ".m4a", ".MP3"]
    filepaths: list[Path] = []

    for album_index, (artist, album) in enumerate(albums):
        album_dir = root / artist / album
        disc_dir = album_dir / "Disc 1"
        disc_dir.mkdir(parents=True, exist_ok=True)

        (album_dir / "cover.jpg").write_bytes(b"jpg")
        (album_dir / "booklet.txt").write_text("notes")
        (album_dir / ".DS_Store").write_bytes(b"mac")
        (disc_dir / "album.cue").write_text("cue data")

        for track_index in range(30):
            suffix = suffixes[(album_index + track_index) % len(suffixes)]
            filename = (
                f"{track_index + 1:02d} - Track {track_index:03d} "
                f"(mix {album_index}){suffix}"
            )
            filepath = disc_dir / filename
            filepath.write_bytes(f"audio-{album_index}-{track_index}".encode())
            filepaths.append(filepath)

    return sorted(filepaths)


def _inspect_all(inspector: Inspector) -> tuple[int, int]:
    total_inspected = 0
    passes = 0

    while True:
        inspected = inspector.run_pass()
        if inspected == 0:
            return total_inspected, passes
        total_inspected += inspected
        passes += 1


def test_scanner_handles_large_messy_library_and_incremental_changes(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    cache = FileCache(tmp_path / "cache.db")
    expected_files = _create_large_library(tmp_path)

    try:
        scanner = Scanner(config, cache)

        assert scanner.run_pass() == (len(expected_files), len(expected_files))
        assert set(cache.all_filepaths()) == {str(path) for path in expected_files}

        assert scanner.run_pass() == (len(expected_files), 0)

        changed_path = expected_files[10]
        new_path = (
            tmp_path / "New Artist" / "Fresh Album" / "Disc 1" / "01 - New Song.flac"
        )
        new_path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(0.01)
        changed_path.write_bytes(b"audio-updated")
        new_path.write_bytes(b"brand-new-audio")

        assert scanner.run_pass() == (len(expected_files) + 1, 2)
    finally:
        cache.close()


def test_inspector_drains_large_queue_in_multiple_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _make_config(tmp_path, inspector_batch_size=7)
    cache = FileCache(tmp_path / "cache.db")

    try:
        filepaths = []
        for index in range(25):
            filepath = tmp_path / f"track_{index:02d}.mp3"
            filepath.write_bytes(f"audio-{index}".encode())
            cache.mark_changed(filepath)
            filepaths.append(filepath)

        def _fake_mutagen_file(
            filepath_str: str, easy: bool = False
        ) -> dict[str, list[str]]:
            del easy
            index = int(Path(filepath_str).stem.split("_")[1])
            payload = dict(FULL_TAG_PAYLOAD)
            if index % 4 == 0:
                payload.pop("TBPM")
                payload.pop("TONALITY")
            return payload

        monkeypatch.setattr(inspector_module.mutagen, "File", _fake_mutagen_file)

        inspector = Inspector(config, cache)
        total_inspected, passes = _inspect_all(inspector)
        expected_missing = sum(1 for index in range(len(filepaths)) if index % 4 == 0)

        stats = cache.stats()

        assert passes == 4
        assert total_inspected == len(filepaths)
        assert inspector.inspected == len(filepaths)
        assert inspector.queued == expected_missing
        assert inspector.errors == 0
        assert stats["needs_inspection"] == 0
        assert stats["needs_work"] == expected_missing
        assert stats["done"] == len(filepaths) - expected_missing
    finally:
        cache.close()


def test_cleanup_handles_large_library_reorganization(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    cache = FileCache(tmp_path / "cache.db")
    original_files = _create_large_library(tmp_path)

    try:
        scanner = Scanner(config, cache)
        assert scanner.run_pass() == (len(original_files), len(original_files))

        deleted_files = original_files[:3]
        renamed_files = original_files[3:7]
        moved_files = original_files[7:12]

        for filepath in deleted_files:
            filepath.unlink()

        for filepath in renamed_files:
            filepath.rename(filepath.with_name(f"renamed - {filepath.name}"))

        for index, filepath in enumerate(moved_files):
            new_path = tmp_path / "Reorganized" / f"Set {index}" / filepath.name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            filepath.rename(new_path)

        current_files = sorted(path for path in tmp_path.rglob("*") if path.is_file())
        current_audio_files = sorted(
            path
            for path in current_files
            if path.suffix.lower() in {".mp3", ".flac", ".m4a"}
        )

        rescanned, changed = scanner.run_pass()
        cleanup = Cleanup(config, cache)

        assert rescanned == len(current_audio_files)
        assert changed == 9
        assert cleanup.run() == 12
        assert set(cache.all_filepaths()) == {str(path) for path in current_audio_files}
    finally:
        cache.close()
