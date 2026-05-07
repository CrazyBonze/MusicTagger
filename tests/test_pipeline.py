"""Tests for the Pipeline orchestration class.

Pipeline owns the full stage lifecycle: launching, stopping, watchdog,
cron scheduling, and stats.  These tests drive it without a TUI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from musictagger.cache import FileCache
from musictagger.config import Config


def _make_config(tmp_path: Path) -> Config:
    return Config(
        music_path=tmp_path / "music",
        db_path=tmp_path / "cache.db",
        embeddings_db_path=tmp_path / "embeddings.db",
        log_path=tmp_path / "musictagger.log",
        models_dir=tmp_path / "models",
        inspector_throttle_ms=0,
        inspector_batch_size=5,
        worker_batch_size=5,
    )


def _make_queued_file(
    cache: FileCache, tmp_path: Path, name: str = "track.mp3"
) -> Path:
    """Write a fake audio file and register it as needing inspection."""
    filepath = tmp_path / name
    filepath.write_bytes(b"fake-audio")
    cache.mark_changed(filepath)
    cache.flush()
    return filepath


# ── Construction ──────────────────────────────────────────────────────────────


def test_pipeline_constructs_with_config(tmp_path: Path) -> None:
    """Pipeline can be constructed from a Config without starting."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    assert not pipeline.running
    assert pipeline.scanner is not None
    assert pipeline.inspector is not None
    assert pipeline.worker is not None
    assert pipeline.cleanup is not None
    pipeline.close()


def test_pipeline_exposes_stage_objects(tmp_path: Path) -> None:
    """Pipeline exposes the four stage objects for the TUI to read state from."""
    from musictagger.pipeline import Pipeline
    from musictagger.cleanup import Cleanup
    from musictagger.inspector import Inspector
    from musictagger.scanner import Scanner
    from musictagger.worker import Worker

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    assert isinstance(pipeline.scanner, Scanner)
    assert isinstance(pipeline.inspector, Inspector)
    assert isinstance(pipeline.worker, Worker)
    assert isinstance(pipeline.cleanup, Cleanup)
    pipeline.close()


# ── Stats ─────────────────────────────────────────────────────────────────────


def test_pipeline_stats_reflects_cache_state(tmp_path: Path) -> None:
    """pipeline.stats returns a fresh snapshot of the cache — no staleness."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    # Initially empty.
    assert pipeline.stats["total"] == 0

    # Add a file to the cache directly.
    _make_queued_file(pipeline.cache, tmp_path)

    # stats must reflect the new row immediately — no polling delay.
    assert pipeline.stats["total"] == 1
    assert pipeline.stats["needs_inspection"] == 1
    pipeline.close()


# ── Cron scheduling ───────────────────────────────────────────────────────────


def test_pipeline_next_scan_is_set_on_construction(tmp_path: Path) -> None:
    """Pipeline schedules the first scan immediately (next_scan <= now)."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    # First scan fires immediately on startup.
    assert pipeline.next_scan <= time.time()
    pipeline.close()


def test_pipeline_force_scan_resets_next_scan(tmp_path: Path) -> None:
    """force_scan() makes next_scan fire on the very next orchestrator tick."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)
    # Push it far out first so we can confirm force_scan resets it.
    pipeline._next_scan = time.time() + 9999

    pipeline.force_scan()

    assert pipeline.next_scan <= time.time()
    pipeline.close()


def test_pipeline_force_inspect_resets_inspector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force_inspect() resets the inspector stop event and triggers a run."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    ran = threading.Event()

    def _fake_run(self: object) -> None:
        ran.set()

    monkeypatch.setattr(pipeline.inspector.__class__, "run", _fake_run)

    pipeline.inspector.stop()  # pre-set the stop event
    pipeline.force_inspect()

    assert not pipeline.inspector._stop_event.is_set(), (
        "force_inspect() must reset the stop event before launching"
    )
    pipeline.close()


# ── Stop / pause ──────────────────────────────────────────────────────────────


def test_pipeline_stop_sets_all_stage_stop_events(tmp_path: Path) -> None:
    """pipeline.stop() propagates to every stage."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    pipeline.stop()

    assert pipeline.scanner._stop_event.is_set()
    assert pipeline.inspector._stop_event.is_set()
    assert pipeline.worker._stop_event.is_set()
    assert pipeline.cleanup._stop_event.is_set()
    pipeline.close()


def test_pipeline_pause_stops_all_stages(tmp_path: Path) -> None:
    """pipeline.pause() stops all stages without permanently closing."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    pipeline.pause()

    assert pipeline.paused
    assert pipeline.scanner._stop_event.is_set()
    assert pipeline.inspector._stop_event.is_set()
    assert pipeline.worker._stop_event.is_set()
    assert pipeline.cleanup._stop_event.is_set()
    pipeline.close()


def test_pipeline_resume_clears_paused_flag(tmp_path: Path) -> None:
    """pipeline.resume() clears the paused flag so the loop can relaunch stages."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    pipeline.pause()
    assert pipeline.paused

    pipeline.resume()
    assert not pipeline.paused
    pipeline.close()


# ── Watchdog ─────────────────────────────────────────────────────────────────


def test_pipeline_worker_watchdog_stops_hung_worker(tmp_path: Path) -> None:
    """If the worker heartbeat is stale, the watchdog must call worker.stop()."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    # Simulate a hung worker: running=True, heartbeat ancient.
    pipeline.worker._running = True
    pipeline.worker._last_activity = time.monotonic() - 9999

    pipeline._check_watchdogs()

    assert pipeline.worker._stop_event.is_set(), (
        "Watchdog must call worker.stop() when heartbeat is stale"
    )
    pipeline.close()


def test_pipeline_scanner_watchdog_stops_hung_scanner(tmp_path: Path) -> None:
    """If the scanner heartbeat is stale, the watchdog must call scanner.stop()."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    pipeline.scanner._running = True
    pipeline.scanner.last_activity = time.monotonic() - 9999

    pipeline._check_watchdogs()

    assert pipeline.scanner._stop_event.is_set(), (
        "Watchdog must call scanner.stop() when heartbeat is stale"
    )
    pipeline.close()


# ── on_log callback ───────────────────────────────────────────────────────────


def test_pipeline_on_log_receives_stage_messages(tmp_path: Path) -> None:
    """Log messages from stages are forwarded to the on_log callback."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    received: list[tuple[str, str]] = []
    pipeline.on_log = lambda source, msg: received.append((source, msg))

    # Trigger a log message directly via the scanner's log fn.
    pipeline.scanner._log("hello from scanner")

    assert ("scanner", "hello from scanner") in received
    pipeline.close()


# ── start / join / running ────────────────────────────────────────────────────


def test_pipeline_running_is_false_before_start(tmp_path: Path) -> None:
    """Pipeline.running is False until start() is called."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)
    assert not pipeline.running
    pipeline.close()


def test_pipeline_running_is_true_after_start_and_false_after_stop(
    tmp_path: Path,
) -> None:
    """Pipeline.running flips True on start() and back to False after stop()+join()."""
    from musictagger.pipeline import Pipeline

    config = _make_config(tmp_path)
    pipeline = Pipeline(config)

    pipeline.start()
    assert pipeline.running

    pipeline.stop()
    pipeline.join(timeout=5.0)
    assert not pipeline.running
    pipeline.close()
