"""Pipeline — orchestrates the four pipeline stages independently of the TUI.

This module owns the lifecycle of Scanner, Inspector, Worker, and Cleanup.
It can be driven by the TUI (which subscribes to callbacks and renders state)
or run headlessly with no UI at all.

Design
------
A single background thread (the *orchestration thread*) runs a tight loop
that sleeps briefly between iterations.  On each iteration it:

  1. Checks watchdogs — stops hung scanner/worker threads.
  2. Fires the scanner if it is due per cron schedule.
  3. Launches the inspector if there is inspection work and it is not running.
  4. Launches the worker if there is tagging work and it is not running.
  5. Fires cleanup if it is due per cron schedule.

Each stage runs in its own daemon thread, started via ``threading.Thread``.
The orchestration thread never blocks on I/O — it only checks flags and
launches threads.

Stats
-----
``pipeline.stats`` calls ``cache.stats()`` directly and returns a fresh
snapshot every time.  Because the caller is free to do this from any thread
(the cache uses its own lock), there is no polling delay or stale-cache
problem.  The TUI reads this property on its 0.5 s panel-refresh interval
rather than maintaining a separate background stats-refresh worker.

Stop / pause / resume
---------------------
``stop()``  — permanent shutdown; sets all stage stop events, stops the
              orchestration thread, closes resources.
``pause()`` — temporary halt; sets stage stop events but keeps the
              orchestration thread alive so it can relaunch stages on
              ``resume()``.
``resume()``— clears the paused flag; the orchestration loop will relaunch
              stages on its next tick.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable

from croniter import croniter
from loguru import logger

from musictagger.cache import FileCache
from musictagger.cleanup import Cleanup
from musictagger.config import Config
from musictagger.inspector import Inspector
from musictagger.scanner import Scanner
from musictagger.tags import TAGS
from musictagger.worker import Worker

# ── Constants ─────────────────────────────────────────────────────────────────

# How long the orchestration loop sleeps between ticks (seconds).
_TICK_S: float = 0.25

# Worker heartbeat timeout before the watchdog fires (seconds).
_WORKER_HANG_TIMEOUT_S: int = 300  # 5 minutes

# Scanner heartbeat timeout before the watchdog fires (seconds).
_SCANNER_HANG_TIMEOUT_S: int = 120  # 2 minutes


def _cron_next(expr: str) -> float:
    """Return the next Unix timestamp for a cron expression in local wall time."""
    return croniter(expr, datetime.now()).get_next(datetime).timestamp()


# ── Pipeline ──────────────────────────────────────────────────────────────────


class Pipeline:
    """Orchestrates Scanner, Inspector, Worker, and Cleanup.

    Construct with a ``Config``, optionally set ``on_log``, then call
    ``start()`` to begin processing.  Call ``stop()`` followed by
    ``join()`` to shut down cleanly.  Always call ``close()`` when done
    to release database connections.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = FileCache(config.db_path)

        # Callback invoked with (source, message) for every log line emitted
        # by a stage.  Defaults to a no-op; the TUI replaces this with its
        # RichLog writer.  Called from background threads — implementations
        # must be thread-safe (e.g. use call_from_thread or a queue).
        self.on_log: Callable[[str, str], None] = lambda source, msg: None

        # Build stages, wiring their log callbacks through our on_log.
        self.scanner = Scanner(config, self.cache, self._make_log("scanner"))
        self.inspector = Inspector(config, self.cache, self._make_log("inspector"))
        self.worker = Worker(
            config,
            self.cache,
            self._make_log("worker"),
            self._make_log("worker"),  # markup log — same sink for now
        )
        self.cleanup = Cleanup(config, self.cache, self._make_log("cleanup"))

        # Scheduling state.
        # First scan fires immediately on startup.
        self._next_scan: float = time.time()
        self._last_scan: float | None = None
        self._next_cleanup: float = _cron_next(config.cleanup_cron)
        self._last_cleanup: float | None = None

        # Paused flag — set by pause(), cleared by resume().
        self._paused: bool = False

        # Orchestration thread and its stop event.
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Track which stage threads are currently running so we can join them.
        self._stage_threads: dict[str, threading.Thread] = {}
        self._stage_lock = threading.Lock()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        """True while the orchestration thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def next_scan(self) -> float:
        """Unix timestamp of the next scheduled scan."""
        return self._next_scan

    @property
    def last_scan(self) -> float | None:
        """Unix timestamp of the last completed scan, or None."""
        return self._last_scan

    @property
    def next_cleanup(self) -> float:
        """Unix timestamp of the next scheduled cleanup."""
        return self._next_cleanup

    @property
    def last_cleanup(self) -> float | None:
        """Unix timestamp of the last completed cleanup, or None."""
        return self._last_cleanup

    @property
    def stats(self) -> dict:
        """Fresh stats snapshot from the cache.

        Calls cache.stats() directly — no polling delay, always current.
        The TUI reads this on its panel-refresh interval instead of
        maintaining a separate background stats-refresh worker.
        """
        enabled_tags = [t for t in TAGS if self.config.tag_cfg(t.name).enabled]
        try:
            return self.cache.stats(enabled_tags=enabled_tags)
        except Exception:
            return {
                "total": 0,
                "needs_inspection": 0,
                "needs_work": 0,
                "errors": 0,
                "done": 0,
                "per_tag": {t.name: 0 for t in TAGS},
            }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the orchestration thread.  Safe to call once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._orchestrate_loop,
            name="pipeline-orchestrator",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Pipeline orchestration thread started")

    def stop(self) -> None:
        """Signal all stages and the orchestration thread to stop."""
        self._stop_event.set()
        self.scanner.stop()
        self.inspector.stop()
        self.worker.stop()
        self.cleanup.stop()
        logger.debug("Pipeline stop requested")

    def join(self, timeout: float = 10.0) -> None:
        """Wait for the orchestration thread and all stage threads to finish.

        Blocks for at most *timeout* seconds total.
        """
        deadline = time.monotonic() + timeout
        if self._thread is not None:
            remaining = max(0.0, deadline - time.monotonic())
            self._thread.join(timeout=remaining)

        with self._stage_lock:
            threads = list(self._stage_threads.values())
        for t in threads:
            remaining = max(0.0, deadline - time.monotonic())
            if remaining > 0:
                t.join(timeout=remaining)

    def close(self) -> None:
        """Release database connections and worker resources.

        Always call this when the pipeline is permanently shut down.
        """
        self.worker.close()
        self.cache.close()

    # ── Pause / resume ────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause all stages.  The orchestration loop keeps running."""
        self._paused = True
        self.scanner.stop()
        self.inspector.stop()
        self.worker.stop()
        self.cleanup.stop()
        logger.info("Pipeline paused")

    def resume(self) -> None:
        """Resume after a pause.  Stages will be relaunched on the next tick."""
        self._paused = False
        logger.info("Pipeline resumed")

    # ── Actions (callable from TUI key bindings or headless signals) ──────────

    def force_scan(self) -> None:
        """Trigger a scan on the next orchestrator tick regardless of schedule."""
        self._next_scan = time.time()

    def force_inspect(self) -> None:
        """Launch an inspection pass immediately if not already running."""
        if self.inspector.running:
            return
        self.inspector.reset()
        self._launch_stage("inspector", self.inspector.run)

    def force_cleanup(self) -> None:
        """Trigger a cleanup run on the next orchestrator tick."""
        self._next_cleanup = time.time()

    def requeue_errors(self) -> int:
        """Reset all error-status rows back to queued. Returns count requeued."""
        count = self.cache.requeue_errors()
        if count:
            self.cache.flush()
            logger.info("Requeued {} error row(s)", count)
        return count

    # ── Watchdog (also exposed for testing) ───────────────────────────────────

    def _check_watchdogs(self) -> None:
        """Stop hung scanner/worker threads if their heartbeat is stale."""
        if self.scanner.running:
            idle_s = time.monotonic() - self.scanner.last_activity
            if idle_s > _SCANNER_HANG_TIMEOUT_S:
                logger.warning(
                    "Pipeline watchdog: scanner heartbeat stale for {}s — stopping",
                    int(idle_s),
                )
                self.scanner.stop()

        if self.worker.running:
            idle_s = time.monotonic() - self.worker.last_activity
            if idle_s > _WORKER_HANG_TIMEOUT_S:
                logger.warning(
                    "Pipeline watchdog: worker heartbeat stale for {}s — stopping",
                    int(idle_s),
                )
                self.worker.stop()
                # Requeue any row the hung pass left in 'working' status.
                try:
                    recovered = self.cache.requeue_working()
                    if recovered:
                        self.cache.flush()
                        logger.warning(
                            "Pipeline watchdog: requeued {} stuck 'working' row(s)",
                            recovered,
                        )
                except Exception as exc:
                    logger.warning("Pipeline watchdog recovery failed: {}", exc)

    # ── Internal stage launcher ───────────────────────────────────────────────

    def _make_log(self, source: str) -> Callable[[str], None]:
        """Return a log callback that forwards messages to self.on_log."""

        def _log(msg: str) -> None:
            try:
                self.on_log(source, msg)
            except Exception:
                pass  # never let a broken log callback kill a stage thread

        return _log

    def _launch_stage(self, name: str, target: Callable[[], None]) -> None:
        """Start *target* in a daemon thread, tracking it by *name*."""

        def _wrapper() -> None:
            try:
                target()
            except Exception as exc:
                logger.warning(
                    "Stage '{}' raised an unhandled exception: {}", name, exc
                )
            finally:
                with self._stage_lock:
                    self._stage_threads.pop(name, None)

        t = threading.Thread(target=_wrapper, name=f"pipeline-{name}", daemon=True)
        with self._stage_lock:
            self._stage_threads[name] = t
        t.start()

    # ── Orchestration loop ────────────────────────────────────────────────────

    def _orchestrate_loop(self) -> None:
        """Main loop: runs on the orchestration thread until stop() is called."""
        logger.debug("Pipeline orchestration loop running")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                # Non-fatal: log and keep going rather than crashing the thread.
                logger.warning("Pipeline orchestration tick error: {}", exc)
            self._stop_event.wait(timeout=_TICK_S)
        logger.debug("Pipeline orchestration loop exited")

    def _tick(self) -> None:
        """Single orchestration tick — called every _TICK_S seconds."""
        if self._paused:
            return

        now = time.time()

        # ── Watchdogs ──────────────────────────────────────────────────────────
        self._check_watchdogs()

        # ── Scanner (cron-driven) ──────────────────────────────────────────────
        # Reset cron after a force-scan (next_scan was set to inf temporarily).
        if self._next_scan == float("inf") and not self.scanner.running:
            self._next_scan = _cron_next(self.config.scan_cron)

        if now >= self._next_scan and not self.scanner.running:
            self._last_scan = now
            self._next_scan = _cron_next(self.config.scan_cron)
            self.scanner.reset()
            self._launch_stage("scanner", self.scanner.run_pass)

        # ── Inspector (continuous while queue non-empty) ───────────────────────
        if not self.inspector.running:
            enabled = [t for t in TAGS if self.config.tag_cfg(t.name).enabled]
            try:
                needs = self.cache.needs_inspection(limit=1, enabled_tags=enabled)
            except Exception:
                needs = []
            if needs:
                self.inspector.reset()
                self._launch_stage("inspector", self.inspector.run)

        # ── Worker (continuous while queue non-empty) ──────────────────────────
        if not self.worker.running:
            # Recover rows stuck in 'working' from a crashed/killed previous pass.
            try:
                recovered = self.cache.requeue_working()
                if recovered:
                    self.cache.flush()
                    logger.warning(
                        "Pipeline: requeued {} stuck 'working' row(s)", recovered
                    )
            except Exception:
                pass

            enabled = [t for t in TAGS if self.config.tag_cfg(t.name).enabled]
            try:
                needs = self.cache.needs_work(limit=1, enabled_tags=enabled)
            except Exception:
                needs = []
            if needs:
                self.worker.reset()
                self._launch_stage(
                    "worker",
                    lambda: self.worker.run(self.config.worker_batch_size),
                )

        # ── Cleanup (cron-driven) ──────────────────────────────────────────────
        if self._next_cleanup == float("inf") and not self.cleanup.running:
            self._next_cleanup = _cron_next(self.config.cleanup_cron)

        if now >= self._next_cleanup and not self.cleanup.running:
            self._last_cleanup = now
            self._next_cleanup = _cron_next(self.config.cleanup_cron)
            self.cleanup.reset()
            self._launch_stage("cleanup", self.cleanup.run)
