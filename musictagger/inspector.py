"""Inspector — the bridge between the scanner and the worker.

The scanner knows *something changed* but never opens the file.
The inspector opens each changed file with mutagen, runs all TagDef
check functions, and records which tags are present or absent.

Files with absent tags (has_* = 0) are then queued for the worker.
Files that are fully tagged get marked done without touching the worker.

Overwrite behaviour
-------------------
Tags configured with ``overwrite = true`` are treated as absent by the
inspector regardless of whether the audio file actually carries them.
This causes ``mark_inspected()`` to write ``has_* = 0`` and set
``processing_status = 'queued'``, so ``needs_work()`` returns the file
and the worker rewrites the tag.

Critically, overwrite only fires when a file is (re-)inspected, which
only happens when the scanner detects an mtime/size change or the cache
is cleared.  Files whose cache row is already ``'done'`` are not
re-inspected and are therefore never rewritten unnecessarily.

The inspector also reads the ``acoustid_fingerprint`` tag (written by the
acoustag tool) and stores a SHA-256 hash of it in the main cache.  The worker
uses this hash as the key into ``embeddings.db`` so it can retrieve cached
EffNet embeddings without recomputing them.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import mutagen
from loguru import logger

from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.embeddings import fingerprint_hash as _fp_hash
from musictagger.tags import TAGS


def _read_acoustid_fingerprint(f: mutagen.FileType) -> str | None:
    """Extract the raw Acoustid fingerprint string from a mutagen file object.

    Mirrors the key conventions used by Picard / acoustag so the values agree:
      - Vorbis/FLAC/Opus : ``acoustid_fingerprint`` (case-insensitive)
      - ID3 (MP3/AIFF)   : ``TXXX:Acoustid Fingerprint``
      - MP4/M4A          : ``----:com.apple.iTunes:Acoustid Fingerprint``
      - ASF/WMA          : ``Acoustid/Fingerprint``

    Returns the fingerprint string, or None if absent or unreadable.
    """
    try:
        from mutagen.aiff import AIFF
        from mutagen.asf import ASF
        from mutagen.id3 import ID3FileType
        from mutagen.mp4 import MP4

        if isinstance(f, MP4):
            raw = (f.tags or {}).get("----:com.apple.iTunes:Acoustid Fingerprint")
            if raw:
                v = raw[0]
                return (
                    v.decode("utf-8", errors="ignore")
                    if isinstance(v, bytes)
                    else str(v)
                )

        if isinstance(f, ASF):
            raw = (f.tags or {}).get("Acoustid/Fingerprint")
            if raw:
                return str(raw[0])

        if isinstance(f, (ID3FileType, AIFF)):
            id3 = f.tags
            if id3:
                for frame in id3.getall("TXXX"):
                    if frame.desc == "Acoustid Fingerprint":
                        return str(frame.text[0]) if frame.text else None

        # Vorbis comments (FLAC, Ogg Vorbis, Ogg Opus, …)
        tags = f.tags or {}
        for key in ("acoustid_fingerprint", "ACOUSTID_FINGERPRINT"):
            val = tags.get(key)
            if val:
                return str(val[0]) if isinstance(val, list) else str(val)

    except Exception as exc:
        # Tag read errors must never interrupt the inspection loop.
        logger.debug("Could not read acoustid_fingerprint: {}", exc)

    return None


class Inspector:
    """Reads tags from files flagged as needing inspection.

    Runs in batches — call run_pass() repeatedly to drain the queue.
    Throttled via config.inspector_throttle_ms.
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
        # Stop signal — set by stop(), cleared by reset().  The Pipeline calls
        # reset() before each new run so the inspector can be relaunched after
        # a pause or between cron cycles cleanly.
        self._stop_event = threading.Event()
        self._inspected = 0
        self._queued = 0
        self._errors = 0
        self._current_file = ""
        # Rate tracking: monotonic start of the current pass and the session
        # inspected count at that moment.  Reset on each run_pass() call.
        self._pass_start: float = 0.0
        self._pass_inspected_at_start: int = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def inspected(self) -> int:
        return self._inspected

    @property
    def queued(self) -> int:
        """Files sent to the work queue this session."""
        return self._queued

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def current_file(self) -> str:
        return self._current_file

    @property
    def pass_rate(self) -> float:
        """Files inspected per second in the current (or last) pass.

        Returns 0.0 if no files have been inspected in this pass yet.
        """
        elapsed = time.monotonic() - self._pass_start
        done_this_pass = self._inspected - self._pass_inspected_at_start
        if elapsed <= 0 or done_this_pass <= 0:
            return 0.0
        return done_this_pass / elapsed

    def run_pass(self) -> int:
        """Inspect one batch of files. Returns count inspected.

        Blocking — run this in a thread worker.
        """
        self._running = True
        self._pass_start = time.monotonic()
        self._pass_inspected_at_start = self._inspected
        sleep_s = self.config.inspector_throttle_ms / 1000.0
        inspected = 0

        enabled_tags = [t for t in TAGS if self.config.tag_cfg(t.name).enabled]
        filepaths = self.cache.needs_inspection(
            limit=self.config.inspector_batch_size,
            enabled_tags=enabled_tags,
        )

        if not filepaths:
            self._running = False
            return 0

        for filepath_str in filepaths:
            if self._stop_event.is_set():
                break

            filepath = Path(filepath_str)
            self._current_file = filepath.name

            if not filepath.exists():
                # Gone since the scanner saw it — cleanup job will remove it.
                # Mark enabled tags False so the file leaves the inspection queue.
                # Disabled tags stay NULL so they're re-checked if later enabled.
                self.cache.mark_inspected(
                    filepath,
                    {
                        t.name: False
                        for t in TAGS
                        if self.config.tag_cfg(t.name).enabled
                    },
                )
                logger.debug("File gone since scan: {}", filepath.name)
                continue

            try:
                f = mutagen.File(filepath_str, easy=False)

                if f is None:
                    # Mutagen doesn't recognise the format — treat enabled tags
                    # as absent; disabled tags stay NULL.
                    self.cache.mark_inspected(
                        filepath,
                        {
                            t.name: False
                            for t in TAGS
                            if self.config.tag_cfg(t.name).enabled
                        },
                    )
                    self._log(f"Unrecognised format: {filepath.name}")
                    logger.debug("Unrecognised format: {}", filepath.name)
                    continue

                # Only inspect tags that are enabled in config.  Disabled tags
                # are deliberately omitted from results so their has_* columns
                # remain NULL — the inspector will revisit them if the tag is
                # later re-enabled without any file change being necessary.
                #
                # Tags with overwrite=True are recorded as absent (False) even
                # when the audio file already carries the tag.  This causes
                # mark_inspected() to set has_*=0 and status='queued', so
                # needs_work() returns the file and the worker rewrites it.
                # Overwrite only fires when a file is (re-)inspected — i.e.
                # when the cache is cleared or the file is physically modified —
                # so unmodified files whose cache row is already 'done' are
                # never touched unnecessarily.
                results: dict[str, bool] = {}
                for tag in TAGS:
                    cfg = self.config.tag_cfg(tag.name)
                    if not cfg.enabled:
                        continue
                    try:
                        present = bool(tag.check_fn(f))
                    except Exception as exc:
                        logger.debug(
                            "Tag check failed {} / {}: {}", filepath.name, tag.name, exc
                        )
                        present = False
                    # Force absent when overwrite is set so the worker rewrites
                    # the tag even if it is already present in the audio file.
                    results[tag.name] = present and not cfg.overwrite

                self.cache.mark_inspected(filepath, results)

                # Read and cache the Acoustid fingerprint hash for the embeddings
                # cache.  This is best-effort — a missing tag is not an error.
                raw_fp = _read_acoustid_fingerprint(f)
                if raw_fp:
                    self.cache.set_fingerprint_hash(filepath, _fp_hash(raw_fp))
                    logger.debug("Stored fingerprint hash for {}", filepath.name)

                missing = [
                    t.description
                    for t in TAGS
                    if self.config.tag_cfg(t.name).enabled and not results.get(t.name)
                ]
                if missing:
                    self._log(f"Queued ({', '.join(missing)}): {filepath.name}")
                    logger.debug("Queued ({}): {}", ", ".join(missing), filepath.name)
                    self._queued += 1
                else:
                    self._log(f"All tags present: {filepath.name}")
                    logger.debug("All tags present: {}", filepath.name)

                inspected += 1
                self._inspected += 1

            except Exception as exc:
                self._errors += 1
                logger.warning("Inspector error for {}: {}", filepath_str, exc)
                self._log(f"Error reading {filepath.name}: {exc}")
                # Mark as error so the file leaves the inspection queue.
                # Without this, needs_inspection() returns it on every pass
                # and one corrupt file burns CPU indefinitely.
                self.cache.mark_error(filepath, str(exc))

            if sleep_s > 0:
                time.sleep(sleep_s)

        self.cache.flush()
        logger.info(
            "Inspector pass complete: {} inspected, {} queued for work, {} errors",
            inspected,
            self._queued,
            self._errors,
        )
        self._running = False
        self._current_file = ""
        return inspected

    def run(self) -> None:
        """Drain the inspection queue by calling run_pass() until empty.

        Loops immediately between passes so a full batch triggers the next
        pass without waiting for an external scheduler tick.  Exits when
        run_pass() returns 0 (empty queue) or stop() is called.
        """
        while not self._stop_event.is_set():
            processed = self.run_pass()
            if processed == 0:
                break

    def stop(self) -> None:
        """Signal the inspector to stop at the next iteration boundary."""
        self._stop_event.set()
        self._running = False

    def reset(self) -> None:
        """Clear the stop signal so the inspector can be relaunched."""
        self._stop_event.clear()
