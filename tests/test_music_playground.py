"""Integration tests against the local music playground library.

These tests use the real sample albums in the repository's ``music/`` folder
to validate the scanner and inspector end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import mutagen
import pytest

from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.inspector import Inspector
from musictagger.scanner import AUDIO_EXTENSIONS, Scanner
from musictagger.tags import TAGS

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1] / "music"

pytestmark = pytest.mark.skipif(
    not PLAYGROUND_ROOT.exists(),
    reason="local music playground not available",
)


@pytest.fixture
def music_path() -> Path:
    return PLAYGROUND_ROOT


@pytest.fixture
def cache(tmp_path: Path) -> FileCache:
    file_cache = FileCache(tmp_path / "cache.db")
    yield file_cache
    file_cache.close()


@pytest.fixture
def config(tmp_path: Path, music_path: Path) -> Config:
    return Config(
        music_path=music_path,
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        file_throttle_ms=0,
        dir_throttle_ms=0,
        inspector_throttle_ms=0,
        inspector_batch_size=7,
    )


def _audio_filepaths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and _is_audio_file(path)
    )


def _is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def _inspect_all(inspector: Inspector) -> int:
    total = 0
    while True:
        inspected = inspector.run_pass()
        if inspected == 0:
            return total
        total += inspected


def _direct_tag_results(filepath: Path) -> dict[str, bool]:
    audio_file = mutagen.File(filepath, easy=False)
    if audio_file is None:
        return {tag.name: False for tag in TAGS}

    results: dict[str, bool] = {}
    for tag in TAGS:
        try:
            results[tag.name] = bool(tag.check_fn(audio_file))
        except Exception:
            results[tag.name] = False
    return results


def test_scanner_tracks_only_audio_files(config: Config, cache: FileCache) -> None:
    expected_files = _audio_filepaths(config.music_path)

    scanner = Scanner(config, cache)

    assert scanner.run_pass() == (len(expected_files), len(expected_files))
    assert set(cache.all_filepaths()) == {str(path) for path in expected_files}


def test_scanner_second_pass_reports_no_changes(
    config: Config,
    cache: FileCache,
) -> None:
    expected_files = _audio_filepaths(config.music_path)
    scanner = Scanner(config, cache)

    scanner.run_pass()

    assert scanner.run_pass() == (len(expected_files), 0)


def test_inspector_matches_direct_mutagen_results(
    config: Config,
    cache: FileCache,
) -> None:
    expected_files = _audio_filepaths(config.music_path)
    scanner = Scanner(config, cache)

    scanner.run_pass()

    inspector = Inspector(config, cache)
    total_inspected = _inspect_all(inspector)

    expected_done = 0
    expected_needs_work = 0
    expected_per_tag = {tag.name: 0 for tag in TAGS}

    for filepath in expected_files:
        results = _direct_tag_results(filepath)
        if all(results.values()):
            expected_done += 1
        else:
            expected_needs_work += 1

        for tag_name, present in results.items():
            if present:
                expected_per_tag[tag_name] += 1

    stats = cache.stats()

    assert total_inspected == len(expected_files)
    assert inspector.errors == 0
    assert stats["total"] == len(expected_files)
    assert stats["needs_inspection"] == 0
    assert stats["needs_work"] == expected_needs_work
    assert stats["done"] == expected_done
    assert stats["errors"] == 0
    assert stats["per_tag"] == expected_per_tag
