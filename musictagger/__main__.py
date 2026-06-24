"""Entry point: python -m musictagger [/path/to/music] [--config PATH]"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

from musictagger.cache import FileCache
from musictagger.config import DEFAULT_CONFIG_PATH, Config
from musictagger.download_models import download_models, missing_models
from musictagger.logging import setup_logging
from musictagger.tags import TAGS
from musictagger.tui import MusicTaggerApp

MODEL_DOWNLOAD_POLICIES = frozenset({"ask", "always", "never"})


def _normalize_model_download_policy(policy: str) -> str:
    normalized = policy.strip().lower()
    if normalized not in MODEL_DOWNLOAD_POLICIES:
        allowed = ", ".join(sorted(MODEL_DOWNLOAD_POLICIES))
        raise SystemExit(
            f"Invalid worker.download_models value {policy!r}. Expected one of: {allowed}"
        )
    return normalized


def _resolve_model_download_policy(
    config_policy: str,
    *,
    force_download: bool,
    skip_download: bool,
) -> str:
    if force_download:
        return "always"
    if skip_download:
        return "never"
    return _normalize_model_download_policy(config_policy)


def _ensure_models_available(
    config: Config,
    policy: str,
    *,
    stdin: object | None = None,
) -> None:
    missing = missing_models(config.models_dir)
    if not missing or policy == "never":
        return

    if policy == "ask":
        stream = sys.stdin if stdin is None else stdin
        if not stream.isatty():
            print(
                "Essentia models are missing; skipping download in non-interactive mode.",
                file=sys.stderr,
            )
            return

        response = (
            input(
                f"Essentia models are missing in {config.models_dir}. Download now? [y/N] "
            )
            .strip()
            .lower()
        )
        if response not in {"y", "yes"}:
            return

    print(f"Downloading {len(missing)} Essentia model files to {config.models_dir}...")
    download_models(config.models_dir)


def _print_info(config: Config) -> None:
    """Print a human-readable summary of the resolved config and storage paths."""
    from musictagger.download_models import MODELS

    cfg_path = DEFAULT_CONFIG_PATH

    def _exists(p: Path) -> str:
        return "exists" if p.exists() else "not found"

    def _model_status(models_dir: Path) -> str:
        present = sum(1 for name, _ in MODELS if (models_dir / name).exists())
        total = len(MODELS)
        return f"{present}/{total} present"

    lines = [
        "",
        "musictagger — configuration summary",
        "─" * 40,
        "",
        "Config file",
        f"  path     : {cfg_path}",
        f"  status   : {_exists(cfg_path)}",
        "",
        "Music library",
        f"  path     : {config.music_path}",
        f"  status   : {_exists(config.music_path)}",
        "",
        "Storage",
        f"  cache db        : {config.db_path}",
        f"  embeddings db   : {config.embeddings_db_path}",
        f"  log file        : {config.log_path}",
        f"  models dir      : {config.models_dir}",
        f"  models          : {_model_status(config.models_dir)}",
        "",
        "Pipeline",
        f"  scan cron       : {config.scan_cron}",
        f"  cleanup cron    : {config.cleanup_cron}",
        f"  file throttle   : {config.file_throttle_ms} ms",
        f"  dir throttle    : {config.dir_throttle_ms} ms",
        f"  inspector batch : {config.inspector_batch_size}",
        f"  worker batch    : {config.worker_batch_size}",
        "",
        "Worker",
        f"  bpm threshold   : {config.bpm_confidence_threshold}",
        f"  mood threshold  : {config.mood_threshold}",
        f"  mood min/max    : {config.mood_min_results} / {config.mood_max_results}",
        f"  model policy    : {config.model_download_policy}",
        "",
        "Tags",
    ]

    for tag in TAGS:
        tc = config.tag_cfg(tag.name)
        status = "enabled" if tc.enabled else "disabled"
        overwrite = ", overwrite" if tc.overwrite else ""
        lines.append(f"  {tag.name:<22} {status}{overwrite}")

    lines.append("")
    print("\n".join(lines))


def _recover_interrupted_rows(config: Config) -> None:
    """Recover rows left in inconsistent states by a prior crash or kill.

    Two recovery passes are performed each startup:

    1. 'working' rows — the previous process was killed while a file was
       being processed.  These rows are silently excluded from needs_work()
       forever because that query only selects 'queued'/NULL rows.  Reset them
       to 'queued' so the worker retries them.

    2. 'done' rows with missing tags — mark_done() was called but no tag data
       was actually written (process_file() returned an empty dict).  Reset
       them to 'queued' so the worker tries again.  Only enabled tags are
       checked so files that are legitimately done for active tags are left alone.
    """
    from loguru import logger

    enabled_tags = [t for t in TAGS if config.tag_cfg(t.name).enabled]

    with FileCache(config.db_path) as cache:
        working = cache.requeue_working()
        done_missing = cache.requeue_done_missing_tags(enabled_tags=enabled_tags)
        cache.flush()

    if working:
        logger.warning(
            "Startup recovery: reset {} 'working' row(s) back to queued "
            "(process was killed during a previous worker pass)",
            working,
        )
    if done_missing:
        logger.warning(
            "Startup recovery: reset {} 'done' row(s) that had missing tags "
            "back to queued (mark_done was called without writing tags)",
            done_missing,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="musictagger",
        description="Music library tag analysis and repair pipeline.",
    )
    parser.add_argument(
        "music_path",
        nargs="?",
        metavar="PATH",
        help="Path to the music library. Overrides music_path in the config file.",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help=f"Config file to use (default: {DEFAULT_CONFIG_PATH})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download-models",
        action="store_true",
        help="Download missing Essentia model files before startup.",
    )
    group.add_argument(
        "--no-download-models",
        action="store_true",
        help="Do not download missing Essentia model files at startup.",
    )
    parser.add_argument(
        "--requeue-errors",
        action="store_true",
        help=(
            "Reset all error-status cache rows that still have missing tags back "
            "to queued so the worker will retry them, then exit."
        ),
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print resolved config and storage paths, then exit.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run the pipeline without a TUI.  All output goes to the log file "
            "and stderr.  Send SIGINT (Ctrl-C) or SIGTERM to stop cleanly."
        ),
    )
    args = parser.parse_args()

    config = Config.load(
        music_path=args.music_path,
        config_path=Path(args.config) if args.config else None,
    )

    if args.info:
        _print_info(config)
        return

    if args.requeue_errors:
        setup_logging(config)
        with FileCache(config.db_path) as cache:
            count = cache.requeue_errors()
            cache.flush()
        from loguru import logger

        logger.complete()
        print(f"Requeued {count} error row(s). Run musictagger to process them.")
        return

    policy = _resolve_model_download_policy(
        config.model_download_policy,
        force_download=args.download_models,
        skip_download=args.no_download_models,
    )
    setup_logging(config)
    _recover_interrupted_rows(config)
    _ensure_models_available(config, policy)

    if args.headless:
        _run_headless(config)
    else:
        app = MusicTaggerApp(config)
        app.run()


def _run_headless(config: Config) -> None:
    """Run the full pipeline without a TUI.

    Starts the Pipeline, wires log callbacks so every stage message is printed
    to the terminal with a timestamp and source prefix, then blocks until a
    stop signal is received or the pipeline exits on its own.

    Output format::

        12:34:56 [scanner]  Scanning /path/to/music
        12:34:57 [worker]   Techno (0.82) — /path/to/track.flac

    A brief stats summary is also printed every 30 seconds so overall progress
    is visible without scrolling through individual file lines.

    Send SIGINT (Ctrl-C) or SIGTERM to stop cleanly.
    """
    import re

    from loguru import logger

    from musictagger.pipeline import Pipeline

    # ── Markup stripping ──────────────────────────────────────────────────────
    # Rich markup tags look like [bold], [/bold], [bold cyan], [on green], etc.
    # Strip them so markup-carrying worker lines (mood scores, BPM results) are
    # readable plain text rather than showing raw bracket tokens.
    _MARKUP_RE = re.compile(r"\[/?[a-zA-Z][^\[\]]*\]")

    def _strip_markup(text: str) -> str:
        return _MARKUP_RE.sub("", text)

    # ── Terminal log callback ─────────────────────────────────────────────────

    def _print_line(source: str, msg: str, *, markup: bool = False) -> None:
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M:%S")
        body = _strip_markup(msg) if markup else msg
        print(f"{ts} [{source}] {body}", flush=True)

    # ── Pipeline wiring ───────────────────────────────────────────────────────

    pipeline = Pipeline(config)
    pipeline.on_log = lambda source, msg: _print_line(source, msg)
    pipeline.on_log_markup = lambda source, msg: _print_line(
        source, msg, markup=True
    )

    stop_requested = threading.Event()

    def _handle_signal(signum: int, frame: object) -> None:
        logger.info("Signal {} received — stopping pipeline", signum)
        print(f"\nReceived signal {signum} — stopping…", flush=True)
        stop_requested.set()
        pipeline.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "musictagger headless mode started — library={}, db={}",
        config.music_path,
        config.db_path,
    )
    print(f"musictagger headless — library: {config.music_path}", flush=True)
    print(f"database: {config.db_path}", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    pipeline.start()

    # ── Main loop — print a stats summary every _STATS_INTERVAL_S seconds ────

    _STATS_INTERVAL_S = 30
    _last_stats_print: float = 0.0

    while pipeline.running and not stop_requested.is_set():
        stop_requested.wait(timeout=1.0)

        now = time.monotonic()
        if now - _last_stats_print >= _STATS_INTERVAL_S:
            _last_stats_print = now
            try:
                stats = pipeline.stats
                total = stats.get("total", 0)
                done = stats.get("done", 0)
                needs_work = stats.get("needs_work", 0)
                needs_inspection = stats.get("needs_inspection", 0)
                errors = stats.get("errors", 0)
                print(
                    f"[stats] total={total:,}  done={done:,}"
                    f"  needs_work={needs_work:,}"
                    f"  uninspected={needs_inspection:,}"
                    f"  errors={errors:,}",
                    flush=True,
                )
            except Exception:
                pass  # stale stats — not worth crashing the loop

    pipeline.stop()
    pipeline.join(timeout=15.0)
    pipeline.close()

    logger.info("musictagger headless mode stopped")
    print("\nmusictagger headless stopped.", flush=True)
    _flush_log_with_timeout(timeout=2.0)


def _run() -> None:
    """Entrypoint wrapper that forces an immediate process exit after main().

    PyTorch, TensorFlow, and Essentia each spin up C-level thread pools
    (OpenMP workers, ATen threads, TF inter/intra-op threads) the moment their
    models are loaded.  These are non-daemon threads that Python's interpreter
    shutdown joins before it can exit — causing the process to hang for several
    seconds after the TUI has already closed.

    All application-level cleanup (SQLite commit+close, embedding cache close)
    is done inside on_unmount() before app.run() returns, so there is nothing
    meaningful left to clean up.  os._exit() terminates the process immediately
    without running atexit handlers or joining threads.

    Tests call main() directly and do not go through _run(), so they are
    unaffected by the forced exit.
    """
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C while the TUI is running — Textual has already called
        # on_unmount and cleaned up.  Fall through to os._exit() below.
        pass

    # Best-effort flush of the loguru enqueue buffer before forcing exit.
    # enqueue=True writes records via a background thread; we give it up to
    # 2 seconds to drain.  If it takes longer we still force-exit — a clean
    # terminal is more important than preserving the last few log lines.
    # os._exit() is used rather than sys.exit() because PyTorch/TensorFlow/
    # Essentia spin up non-daemon C-level thread pools that would otherwise
    # block interpreter shutdown indefinitely.
    _flush_log_with_timeout(timeout=2.0)
    os._exit(0)


def _flush_log_with_timeout(timeout: float) -> None:
    """Try to flush the loguru async queue, giving up after *timeout* seconds."""
    import threading as _threading
    from loguru import logger as _logger

    done = _threading.Event()

    def _flush() -> None:
        try:
            _logger.complete()
        except Exception:
            pass
        finally:
            done.set()

    t = _threading.Thread(target=_flush, daemon=True)
    t.start()
    done.wait(timeout=timeout)


if __name__ == "__main__":
    _run()
