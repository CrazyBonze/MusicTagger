"""Cleanup job — removes cache entries for files that no longer exist.

File paths can become orphaned when:
  - A file is deleted
  - A file is moved/renamed (old path becomes stale)
  - An album directory is reorganised

This is intentionally separate from the scanner hot path.
Run it infrequently (daily is plenty for most libraries).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from loguru import logger

from musictagger.cache import FileCache
from musictagger.config import Config


class Cleanup:
    """Checks all cached paths and removes entries where the file is gone."""

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
        self._last_removed = 0
        # Rate tracking: total paths in the last run and how many have been
        # checked so far, plus the monotonic start time.
        self._pass_total: int = 0
        self._pass_checked: int = 0
        self._pass_start: float = 0.0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_removed(self) -> int:
        return self._last_removed

    @property
    def pass_total(self) -> int:
        """Total paths to check in the current (or last) run."""
        return self._pass_total

    @property
    def pass_checked(self) -> int:
        """Paths checked so far in the current (or last) run."""
        return self._pass_checked

    @property
    def pass_rate(self) -> float:
        """Paths checked per second in the current (or last) run.

        Returns 0.0 until at least one path has been checked.
        """
        elapsed = time.monotonic() - self._pass_start
        if elapsed <= 0 or self._pass_checked <= 0:
            return 0.0
        return self._pass_checked / elapsed

    def run(self) -> int:
        """Check every cached path; remove orphans. Returns count removed.

        Blocking — run this in a thread worker.
        This will stat() every file in the cache, so on a large library
        over NFS it can take a while. That's fine — it only runs daily.
        """
        self._running = True
        self._pass_start = time.monotonic()
        self._pass_checked = 0

        all_paths = self.cache.all_filepaths()
        total = len(all_paths)
        self._pass_total = total

        self._log("Cleanup: checking for orphaned entries…")
        logger.info(
            "Cleanup started: checking {:,} cached paths",
            total,
        )

        missing: list[str] = []

        for path_str in all_paths:
            if self._stop_event.is_set():
                break
            if not Path(path_str).exists():
                missing.append(path_str)
            self._pass_checked += 1

        for path_str in missing:
            self.cache.remove(path_str)
            self._log(f"Removed orphan: {Path(path_str).name}")
            logger.debug("Removed orphan: {}", path_str)

        self.cache.flush()
        self._last_removed = len(missing)
        self._running = False

        summary = (
            f"Cleanup done: {len(missing)} orphans removed, "
            f"{total - len(missing):,} entries remain"
        )
        self._log(summary)
        logger.info(summary)
        return len(missing)

    def stop(self) -> None:
        """Signal the running pass to stop at the next iteration."""
        self._stop_event.set()
        self._running = False

    def reset(self) -> None:
        """Clear the stop signal so cleanup can be relaunched."""
        self._stop_event.clear()
