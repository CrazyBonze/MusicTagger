"""Entry point: python -m musictagger [/path/to/music] [--config PATH]"""

from __future__ import annotations

import argparse
import os
import sys
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
    app = MusicTaggerApp(config)
    try:
        app.run()
    finally:
        # Flush the loguru enqueue buffer before exit.  enqueue=True writes
        # log records via a background thread; without this the thread can be
        # killed by interpreter shutdown before the last records reach disk.
        from loguru import logger

        logger.complete()


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
    main()
    os._exit(0)


if __name__ == "__main__":
    _run()
