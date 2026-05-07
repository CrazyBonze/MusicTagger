"""Logging setup for musictagger.

Configures loguru as the single logging backend:
  - File sink  : rotating log at config.log_path (INFO by default)
  - No console : Textual owns the terminal; printing to stdout/stderr would
                 corrupt the TUI.

stdlib interop:
  Any third-party library that calls logging.getLogger(...).warning(...) is
  automatically forwarded to loguru via the InterceptHandler below.  This
  covers mutagen, deeprhythm, torch, and our own modules during the transition.
"""

from __future__ import annotations

import logging

from loguru import logger

from musictagger.config import Config


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records into loguru.

    Installed on the root stdlib logger so every library's log calls are
    captured without needing to change each one individually.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level to loguru level name.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the real caller outside of logging
        # internals so loguru reports the right file/line.
        frame, depth = logging.currentframe(), 0
        while frame and (
            frame.f_code.co_filename == logging.__file__
            or frame.f_globals.get("__name__") in ("logging", "loguru")
        ):
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(config: Config) -> None:
    """Initialise loguru sinks.  Call once at startup before the TUI starts.

    - Removes loguru's default stderr sink (would corrupt the Textual TUI).
    - Adds a rotating file sink at config.log_path.
    - Intercepts all stdlib logging so third-party libraries are captured too.
    """
    # Drop the default stderr sink loguru adds at import time.
    logger.remove()

    # Ensure the log directory exists.
    config.log_path.parent.mkdir(parents=True, exist_ok=True)

    # Single rotating file sink — async (enqueued) for thread-safe writes from
    # background scanner, inspector, and worker threads.  Using one sink avoids
    # the dual-sink pitfall where two independent rotation/retention counters
    # track the same base path; with two sinks the WARNING-only sink almost
    # never fills its rotation quota, so its retention never fires and rotated
    # files accumulate indefinitely.  loguru drains the async queue via atexit,
    # so WARNING/ERROR records are preserved on normal shutdown.
    logger.add(
        config.log_path,
        level=config.log_level,
        rotation=config.log_rotation,
        retention=config.log_retention,
        encoding="utf-8",
        # Each record on one line — easy to grep.
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        enqueue=True,  # thread-safe async write — background threads are fine
        backtrace=True,  # full tracebacks on exceptions
        diagnose=False,  # don't leak local variable values into the log file
    )

    # Bridge stdlib → loguru.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
