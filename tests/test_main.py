"""Tests for startup wiring and model download policy handling."""

from __future__ import annotations

from pathlib import Path

import pytest

import musictagger.__main__ as main_module
from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.download_models import MODELS, missing_models
from musictagger.tags import TAGS


class _FakeStdin:
    def __init__(self, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


def _make_config(tmp_path: Path, *, policy: str = "ask") -> Config:
    return Config(
        music_path=tmp_path / "music",
        db_path=tmp_path / "cache.db",
        log_path=tmp_path / "musictagger.log",
        models_dir=tmp_path / "models",
        model_download_policy=policy,
    )


def test_missing_models_reports_only_absent_files(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    present = [MODELS[0][0], MODELS[3][0]]
    for filename in present:
        (models_dir / filename).write_bytes(b"pb")

    assert missing_models(models_dir) == [
        filename for filename, _ in MODELS if filename not in present
    ]


def test_normalize_model_download_policy_rejects_invalid_values() -> None:
    with pytest.raises(SystemExit, match="Invalid worker.download_models value"):
        main_module._normalize_model_download_policy("maybe")


def test_ensure_models_available_downloads_when_policy_is_always(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, policy="always")
    calls: list[Path] = []

    monkeypatch.setattr(main_module, "download_models", lambda dest: calls.append(dest))

    main_module._ensure_models_available(config, "always")

    assert calls == [config.models_dir]


def test_ensure_models_available_respects_interactive_prompt_yes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    calls: list[Path] = []

    monkeypatch.setattr(main_module, "download_models", lambda dest: calls.append(dest))
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")

    main_module._ensure_models_available(
        config,
        "ask",
        stdin=_FakeStdin(interactive=True),
    )

    assert calls == [config.models_dir]


def test_ensure_models_available_skips_when_prompt_declined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    calls: list[Path] = []

    monkeypatch.setattr(main_module, "download_models", lambda dest: calls.append(dest))
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    main_module._ensure_models_available(
        config,
        "ask",
        stdin=_FakeStdin(interactive=True),
    )

    assert calls == []


def test_ensure_models_available_skips_prompt_in_noninteractive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    calls: list[Path] = []

    monkeypatch.setattr(main_module, "download_models", lambda dest: calls.append(dest))

    main_module._ensure_models_available(
        config,
        "ask",
        stdin=_FakeStdin(interactive=False),
    )

    assert calls == []


def test_main_honors_download_models_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, policy="never")
    ensure_calls: list[str] = []
    app_runs: list[Config] = []

    class _FakeApp:
        def __init__(self, app_config: Config) -> None:
            self._config = app_config

        def run(self) -> None:
            app_runs.append(self._config)

    monkeypatch.setattr(
        main_module,
        "Config",
        type("FakeConfigLoader", (), {"load": staticmethod(lambda **_kwargs: config)}),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _config: None)
    monkeypatch.setattr(main_module, "MusicTaggerApp", _FakeApp)
    monkeypatch.setattr(
        main_module,
        "_ensure_models_available",
        lambda _config, policy: ensure_calls.append(policy),
    )
    monkeypatch.setattr(
        "sys.argv", ["musictagger", "--download-models", str(tmp_path / "music")]
    )

    main_module.main()

    assert ensure_calls == ["always"]
    assert app_runs == [config]


def test_main_honors_no_download_models_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, policy="always")
    ensure_calls: list[str] = []

    class _FakeApp:
        def __init__(self, _config: Config) -> None:
            pass

        def run(self) -> None:
            pass

    monkeypatch.setattr(
        main_module,
        "Config",
        type("FakeConfigLoader", (), {"load": staticmethod(lambda **_kwargs: config)}),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _config: None)
    monkeypatch.setattr(main_module, "MusicTaggerApp", _FakeApp)
    monkeypatch.setattr(
        main_module,
        "_ensure_models_available",
        lambda _config, policy: ensure_calls.append(policy),
    )
    monkeypatch.setattr(
        "sys.argv", ["musictagger", "--no-download-models", str(tmp_path / "music")]
    )

    main_module.main()

    assert ensure_calls == ["never"]


# ── _recover_interrupted_rows ──────────────────────────────────────────────────


def _make_audio_file(tmp_path: Path, name: str = "track.mp3") -> Path:
    filepath = tmp_path / name
    filepath.write_bytes(b"fake-audio")
    return filepath


def test_recover_interrupted_rows_resets_working_rows(tmp_path: Path) -> None:
    """Startup recovery should move any 'working' rows back to 'queued'."""
    config = _make_config(tmp_path)
    tag_results_missing = {t.name: False for t in TAGS}

    filepath = _make_audio_file(tmp_path)

    with FileCache(config.db_path) as cache:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results_missing)
        cache.mark_working(filepath)
        cache.flush()

    # Sanity: the row is 'working' before recovery.
    with FileCache(config.db_path) as cache:
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
    assert row == ("working",)

    main_module._recover_interrupted_rows(config)

    with FileCache(config.db_path) as cache:
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("queued",)
        assert cache.needs_work() == [str(filepath)]


def test_recover_interrupted_rows_resets_done_rows_with_missing_tags(
    tmp_path: Path,
) -> None:
    """Startup recovery should requeue 'done' rows that still have missing tags."""
    config = _make_config(tmp_path)
    tag_results_missing = {t.name: False for t in TAGS}

    filepath = _make_audio_file(tmp_path)

    with FileCache(config.db_path) as cache:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, tag_results_missing)
        # Force 'done' without tags written — reproduces the Bug 1 scenario.
        cache.mark_done(filepath)
        cache.flush()

    main_module._recover_interrupted_rows(config)

    with FileCache(config.db_path) as cache:
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("queued",)


def test_recover_interrupted_rows_leaves_clean_done_rows_untouched(
    tmp_path: Path,
) -> None:
    """A properly completed file should not be disturbed by startup recovery."""
    config = _make_config(tmp_path)
    all_present = {t.name: True for t in TAGS}

    filepath = _make_audio_file(tmp_path)

    with FileCache(config.db_path) as cache:
        cache.mark_changed(filepath)
        cache.mark_inspected(filepath, all_present)
        cache.flush()

    main_module._recover_interrupted_rows(config)

    with FileCache(config.db_path) as cache:
        row = cache._conn.execute(
            "SELECT processing_status FROM processed WHERE filepath = ?",
            (str(filepath),),
        ).fetchone()
        assert row == ("done",)


def test_main_calls_recover_interrupted_rows_before_app_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_recover_interrupted_rows must be called during normal startup."""
    config = _make_config(tmp_path, policy="never")
    call_order: list[str] = []

    class _FakeApp:
        def __init__(self, _config: Config) -> None:
            pass

        def run(self) -> None:
            call_order.append("app")

    monkeypatch.setattr(
        main_module,
        "Config",
        type("FakeConfigLoader", (), {"load": staticmethod(lambda **_kwargs: config)}),
    )
    monkeypatch.setattr(main_module, "setup_logging", lambda _config: None)
    monkeypatch.setattr(main_module, "MusicTaggerApp", _FakeApp)
    monkeypatch.setattr(
        main_module,
        "_ensure_models_available",
        lambda *_: None,
    )
    monkeypatch.setattr(
        main_module,
        "_recover_interrupted_rows",
        lambda _config: call_order.append("recover"),
    )
    monkeypatch.setattr("sys.argv", ["musictagger", str(tmp_path / "music")])

    main_module.main()

    # recover must happen before the app launches.
    assert call_order == ["recover", "app"]


# ── Headless mode ─────────────────────────────────────────────────────────────


def test_headless_flag_runs_pipeline_without_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--headless must start the Pipeline and exit cleanly without the TUI."""
    import threading
    import musictagger.__main__ as main_module

    started = threading.Event()
    stopped = threading.Event()

    class _FakePipeline:
        def __init__(self, config: object) -> None:
            self._stop_event = threading.Event()

        def start(self) -> None:
            started.set()

        def stop(self) -> None:
            self._stop_event.set()
            stopped.set()

        def join(self, timeout: float = 10.0) -> None:
            pass

        def close(self) -> None:
            pass

        @property
        def running(self) -> bool:
            # Return False immediately so the wait loop exits.
            return False

    monkeypatch.setattr(main_module, "_recover_interrupted_rows", lambda _: None)
    monkeypatch.setattr(main_module, "_ensure_models_available", lambda *a, **kw: None)
    monkeypatch.setattr("musictagger.pipeline.Pipeline", _FakePipeline)

    config = _make_config(tmp_path)
    main_module._run_headless(config)

    assert started.is_set(), "_run_headless must call pipeline.start()"
    assert stopped.is_set(), "_run_headless must call pipeline.stop()"
