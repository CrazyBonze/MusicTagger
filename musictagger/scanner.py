"""Periodic directory scanner.

Walks the music library using stat() only — never opens audio files.
Throttled to be gentle on NFS. Speed is not the goal; correctness is.

The scanner's only job: tell the cache which files are new or changed.
The inspector handles everything that requires opening a file.

NFS hang protection
-------------------
NFS mounts can block indefinitely on readdir() for a single directory
while the rest of the tree remains fine.  Two complementary defences are
used:

  Per-directory timeout
    Each directory listing is performed by a short-lived daemon thread via
    ``concurrent.futures.ThreadPoolExecutor``.  If the thread doesn't
    return within ``_DIR_READDIR_TIMEOUT_S`` seconds the scanner logs a
    warning, skips that directory, and moves on.  The stuck thread is
    abandoned (it is daemon, so it won't prevent process exit).

  Heartbeat + watchdog (TUI side)
    ``last_activity`` is updated on every file and every successful
    directory entry.  The TUI orchestrator polls this timestamp and
    force-resets ``_running`` if no heartbeat is seen for
    ``_SCANNER_HANG_TIMEOUT_S`` seconds, then relaunches a fresh pass on
    the next tick.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from pathlib import Path
from typing import Callable

from loguru import logger

from musictagger.cache import FileCache
from musictagger.config import Config

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".wav",
        ".aiff",
        ".aif",
        ".wv",
        ".ape",
        ".opus",
        ".mpc",
        ".wma",
        ".alac",
    }
)

# Maximum seconds to wait for a single directory listing before skipping it.
# NFS can stall indefinitely on a single readdir(); this caps the damage to
# one directory at a time rather than freezing the entire scan pass.
_DIR_READDIR_TIMEOUT_S: int = 30


def _list_dir(dirpath_str: str) -> tuple[list[str], list[str]]:
    """Return (sorted_subdirs, sorted_audio_files) for *dirpath_str*.

    Runs in a short-lived thread so the caller can apply a timeout and
    skip directories that NFS refuses to serve promptly.
    """
    entries = os.scandir(dirpath_str)
    subdirs: list[str] = []
    audio_files: list[str] = []
    with entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                subdirs.append(entry.name)
            elif Path(entry.name).suffix.lower() in AUDIO_EXTENSIONS:
                audio_files.append(entry.name)
    subdirs.sort()
    audio_files.sort()
    return subdirs, audio_files


class Scanner:
    """Walks the music directory and updates the cache with new/changed files.

    Designed to be gentle on NFS:
      - stat() only, never opens files
      - Configurable sleep between files and between directories
      - Periodic cache flushes rather than per-file commits
      - Per-directory readdir timeout to survive stalled NFS mounts
      - Can be stopped mid-pass cleanly

    ``last_activity`` is a ``time.monotonic()`` timestamp updated on every
    processed file and every successfully listed directory.  The TUI
    watchdog uses this to detect a frozen scan thread and force-reset it.
    """

    def __init__(
        self,
        config: Config,
        cache: FileCache,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self._log = log_fn or (lambda msg: None)
        self._running = False
        # Stop signal — set by stop(), cleared by reset().
        self._stop_event = threading.Event()
        self._files_scanned = 0
        self._files_changed = 0
        self._current_file = ""
        # Heartbeat updated on every file processed and every directory listed.
        # Initialised to now so the watchdog doesn't fire before the first pass
        # has even started.
        self.last_activity: float = time.monotonic()
        # Rate tracking: wall-clock start of the current pass and how many
        # files had been scanned when timing began.  Reset at each run_pass().
        self._pass_start: float = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def files_scanned(self) -> int:
        return self._files_scanned

    @property
    def files_changed(self) -> int:
        return self._files_changed

    @property
    def current_file(self) -> str:
        return self._current_file

    @property
    def pass_rate(self) -> float:
        """Files scanned per second in the current (or last) pass.

        Returns 0.0 if the pass has not processed any files yet.
        """
        elapsed = time.monotonic() - self._pass_start
        if elapsed <= 0 or self._files_scanned == 0:
            return 0.0
        return self._files_scanned / elapsed

    def run_pass(self) -> tuple[int, int]:
        """Run one full library scan. Returns (files_scanned, files_changed).

        Blocking — run this in a thread worker.

        Each directory listing is attempted with a ``_DIR_READDIR_TIMEOUT_S``
        timeout.  If a directory stalls (common on NFS) it is skipped with a
        warning and the scan continues with the next directory.
        """
        self._running = True
        self._files_scanned = 0
        self._files_changed = 0
        self._current_file = ""
        self.last_activity = time.monotonic()
        self._pass_start = time.monotonic()

        music_path = self.config.music_path
        if not music_path.exists():
            self._log(f"Music path does not exist: {music_path}")
            logger.warning("Music path does not exist: {}", music_path)
            self._running = False
            return 0, 0

        self._log(f"Scan started → {music_path}")
        logger.info("Scan started: {}", music_path)

        file_sleep = self.config.file_throttle_ms / 1000.0
        dir_sleep = self.config.dir_throttle_ms / 1000.0
        commit_every = 500

        # Single long-lived executor for the duration of this scan pass.
        # Each directory listing is submitted as a future with a timeout so
        # that a stalled NFS readdir() doesn't freeze the whole pass.
        # The executor uses one thread; we only ever have one outstanding
        # directory listing at a time.
        executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scanner-readdir"
        )

        try:
            # Manual BFS/DFS walk so we control the directory queue and can
            # skip individual directories that time out.
            dir_queue: list[str] = [str(music_path)]

            while dir_queue and not self._stop_event.is_set():
                dirpath_str = dir_queue.pop(0)

                # Submit the directory listing to a thread with a timeout so
                # that a stalled NFS readdir() doesn't freeze the whole pass.
                logger.debug("Scanner: submitting readdir for {}", dirpath_str)
                future: Future[tuple[list[str], list[str]]] = executor.submit(
                    _list_dir, dirpath_str
                )
                try:
                    subdirs, audio_files = future.result(timeout=_DIR_READDIR_TIMEOUT_S)
                    logger.debug(
                        "Scanner: readdir OK {} — {} subdirs, {} audio files",
                        dirpath_str,
                        len(subdirs),
                        len(audio_files),
                    )
                except FuturesTimeoutError:
                    logger.warning(
                        "Scanner: readdir timeout after {}s — skipping {}",
                        _DIR_READDIR_TIMEOUT_S,
                        dirpath_str,
                    )
                    self._log(
                        f"Readdir timeout ({_DIR_READDIR_TIMEOUT_S}s) — skipping: "
                        f"{Path(dirpath_str).name}"
                    )
                    # Don't add subdirs; we can't list them either.
                    continue
                except Exception as exc:
                    logger.warning("Scanner: readdir error in {}: {}", dirpath_str, exc)
                    self._log(
                        f"Readdir error — skipping {Path(dirpath_str).name}: {exc}"
                    )
                    continue

                # Heartbeat: we successfully listed a directory.
                self.last_activity = time.monotonic()

                # Enqueue subdirectories (already sorted by _list_dir).
                # Insert at front to preserve DFS order matching the original
                # os.walk(topdown=True) behaviour.
                for subdir in reversed(subdirs):
                    dir_queue.insert(0, os.path.join(dirpath_str, subdir))

                for filename in audio_files:
                    if self._stop_event.is_set():
                        break

                    filepath = Path(dirpath_str) / filename
                    self._current_file = filepath.name
                    self._files_scanned += 1
                    self.last_activity = time.monotonic()

                    unchanged = self.cache.is_unchanged(filepath)
                    if not unchanged:
                        self.cache.mark_changed(filepath)
                        self._files_changed += 1
                        self._log(f"New/changed: {filepath.name}")
                        logger.debug("New/changed: {}", filepath)

                    if self._files_scanned % commit_every == 0:
                        self.cache.flush()
                        self._log(
                            f"  … {self._files_scanned:,} scanned, "
                            f"{self._files_changed} new/changed"
                        )
                        logger.info(
                            "Scan progress: {:,} scanned, {} new/changed",
                            self._files_scanned,
                            self._files_changed,
                        )

                    if file_sleep > 0:
                        time.sleep(file_sleep)

                if not self._stop_event.is_set() and dir_sleep > 0:
                    time.sleep(dir_sleep)

        except Exception as exc:
            logger.exception("Scanner error")
            self._log(f"Scanner error: {exc}")
        finally:
            executor.shutdown(wait=False)
            self.cache.flush()
            # Capture whether we finished normally before clearing the flag.
            finished_normally = not self._stop_event.is_set()
            self._running = False
            self._current_file = ""

        if not finished_normally:
            # Stopped by watchdog or user request — don't emit the normal summary.
            return self._files_scanned, self._files_changed

        summary = (
            f"Scan complete: {self._files_scanned:,} files scanned, "
            f"{self._files_changed} new/changed"
        )
        self._log(summary)
        logger.info(summary)
        return self._files_scanned, self._files_changed

    def stop(self) -> None:
        """Signal the running pass to stop at the next iteration."""
        self._stop_event.set()
        self._running = False

    def reset(self) -> None:
        """Clear the stop signal so the scanner can be relaunched."""
        self._stop_event.clear()
