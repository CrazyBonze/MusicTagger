"""TUI application — ties the pipeline together with a live Textual interface."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Callable

from croniter import croniter

from loguru import logger
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from rich.highlighter import ReprHighlighter
from rich.markup import escape as markup_escape
from rich.text import Text
from textual.widgets import Checkbox, Footer, Header, Label, RichLog, Static

from musictagger.cache import FileCache
from musictagger.cleanup import Cleanup
from musictagger.config import Config
from musictagger.embeddings import EmbeddingCache
from musictagger.inspector import Inspector
from musictagger.scanner import Scanner
from musictagger.tags import TAGS
from musictagger.worker import Worker

# Maximum seconds a worker pass can run without a heartbeat before the
# watchdog considers it hung and force-resets the running flag so the
# orchestrator can launch a fresh pass on the next tick.
_WORKER_HANG_TIMEOUT_S = 300  # 5 minutes

# Maximum seconds the scanner can go without a heartbeat.  The scanner updates
# last_activity on every file processed and every directory successfully listed,
# so this timeout only fires when os.scandir() itself is blocking (NFS stall).
# The per-directory readdir timeout in scanner.py caps individual directories,
# but this watchdog catches cases where even that thread is stuck.
_SCANNER_HANG_TIMEOUT_S = 120  # 2 minutes


# ── Scheduling helpers ────────────────────────────────────────────────────────


def _cron_next(expr: str) -> float:
    """Return the next Unix timestamp for a cron expression in local time.

    croniter(expr, float_timestamp).get_next(float) evaluates in UTC, which
    gives wrong results when the system timezone is not UTC.  Passing a naive
    local datetime and calling get_next(datetime) keeps everything in local
    wall-clock time, matching what the user configured.
    """
    return croniter(expr, datetime.now()).get_next(datetime).timestamp()


# ── Internal messages ──────────────────────────────────────────────────────────


class LogEvent(Message):
    """A log line emitted by a background job.

    If *markup* is True, *text* is treated as pre-trusted Rich markup and
    rendered directly via ``_log_markup``; otherwise it is escaped first.
    """

    def __init__(self, source: str, text: str, markup: bool = False) -> None:
        super().__init__()
        self.source = source
        self.text = text
        self.markup = markup


class StatsUpdate(Message):
    """A fresh stats snapshot from the background stats-refresh worker.

    Posted via ``post_message`` (non-blocking, thread-safe) rather than
    ``call_from_thread`` so the background thread never waits for the main
    thread to process the result.
    """

    def __init__(self, stats: dict) -> None:
        super().__init__()
        self.stats = stats


class StoragePanelUpdate(Message):
    """A fresh storage/embeddings snapshot from the background slow-tick worker.

    Carries all the data needed to update StoragePanel so that none of the
    underlying DB queries or filesystem stat() calls touch the main thread.
    """

    def __init__(
        self,
        library_bytes: int,
        cache_db_bytes: int,
        embeddings_db_bytes: int,
        emb_stats: dict,
        fingerprinted: int,
        total: int,
        error_summary: list,
    ) -> None:
        super().__init__()
        self.library_bytes = library_bytes
        self.cache_db_bytes = cache_db_bytes
        self.embeddings_db_bytes = embeddings_db_bytes
        self.emb_stats = emb_stats
        self.fingerprinted = fingerprinted
        self.total = total
        self.error_summary = error_summary


# ── Widgets ────────────────────────────────────────────────────────────────────


class ActivityLog(RichLog):
    """RichLog that pauses auto-scroll while the user is scrolled up.

    When the view is at the bottom, new lines scroll into view as normal.
    The moment the user scrolls up (mouse wheel or keyboard), auto-scroll is
    suspended immediately so the view stays where they left it.  Scrolling
    back to the bottom — by any means — re-enables it.
    """

    def _on_mouse_scroll_up(self, event: object) -> None:
        # Disable auto-scroll the instant the user wheels up, before the
        # parent handler moves the viewport.
        self.auto_scroll = False
        super()._on_mouse_scroll_up(event)  # type: ignore[misc]

    def _on_mouse_scroll_down(self, event: object) -> None:
        super()._on_mouse_scroll_down(event)  # type: ignore[misc]
        # Re-enable once the viewport reaches the bottom.
        if self.is_vertical_scroll_end:
            self.auto_scroll = True

    def action_scroll_up(self) -> None:
        self.auto_scroll = False
        super().action_scroll_up()

    def action_scroll_home(self) -> None:
        self.auto_scroll = False
        super().action_scroll_home()

    def action_scroll_end(self) -> None:
        self.auto_scroll = True
        super().action_scroll_end()


class StatusPanel(Static):
    """Displays key/value rows for one pipeline component."""

    DEFAULT_CSS = """
    StatusPanel {
        border: round $primary-darken-2;
        padding: 1 2;
        width: 1fr;
        height: 100%;
        margin: 0 1;
    }
    """

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._title = title
        self._rows: dict[str, str] = {}

    def set_row(self, key: str, value: str) -> None:
        self._rows[key] = value
        self._redraw()

    def set_rows(self, rows: dict[str, str]) -> None:
        if rows == self._rows:
            return  # Nothing changed — skip the render to keep the event loop free.
        self._rows = rows
        self._redraw()

    def _redraw(self) -> None:
        lines = [f"[bold cyan]{self._title}[/bold cyan]\n"]
        for k, v in self._rows.items():
            lines.append(f"[dim]{k}[/dim]: {v}")
        self.update("\n".join(lines))


class LibraryOverview(VerticalScroll):
    """Library health summary: stacked bar + per-segment table."""

    DEFAULT_CSS = """
    LibraryOverview {
        border: round $primary-darken-2;
        padding: 1 2;
    }
    LibraryOverview > Static {
        height: auto;
    }
    """

    # Segment order, label, and Rich colour used for both the bar and the table.
    _SEGMENTS: list[tuple[str, str, str]] = [
        ("done", "Done", "green"),
        ("needs_work", "Needs work", "red"),
        ("needs_inspection", "Uninspected", "yellow"),
        ("errors", "Errors", "magenta"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("", id="library-overview-inner")

    def update_stats(self, stats: dict) -> None:
        """Re-render using the latest cache stats dict."""
        total: int = stats.get("total", 0)

        # Compute the count for each named segment.
        counts: dict[str, int] = {}
        for key, _label, _color in self._SEGMENTS:
            counts[key] = stats.get(key, 0)

        # Any rows not captured by the named segments (e.g. status=working).
        accounted = sum(counts.values())
        other = max(0, total - accounted)

        try:
            self.query_one("#library-overview-inner", Static).update(
                self._build_markup(total, counts, other)
            )
        except Exception:
            pass

    def _build_markup(self, total: int, counts: dict[str, int], other: int) -> str:
        lines: list[str] = []

        # ── Title ───────────────────────────────────────────────────────────
        lines.append("[bold cyan]▸ Library Overview[/bold cyan]\n")
        lines.append(f"[dim]Total files[/dim]: [bold]{total:,}[/bold]\n")

        if total == 0:
            lines.append("[dim]No files in library yet.[/dim]")
            return "\n".join(lines)

        # ── Stacked bar ─────────────────────────────────────────────────────
        # Reserve space for margins; use a fixed width that looks good.
        bar_width = 60
        segments_with_other = list(self._SEGMENTS)
        counts_full = dict(counts)
        if other > 0:
            segments_with_other = segments_with_other + [
                ("_other", "In progress", "blue")
            ]
            counts_full["_other"] = other

        bar_parts: list[str] = []
        chars_used = 0
        for idx, (key, _label, color) in enumerate(segments_with_other):
            count = counts_full.get(key, 0)
            # Last segment gets any rounding remainder so the bar is always full.
            if idx == len(segments_with_other) - 1:
                chars = bar_width - chars_used
            else:
                chars = round(count / total * bar_width)
            if chars > 0:
                bar_parts.append(f"[on {color}]{' ' * chars}[/on {color}]")
            chars_used += chars

        lines.append("".join(bar_parts))
        lines.append("")

        # ── Legend / table ───────────────────────────────────────────────────
        # Column widths: colour swatch (1), label (14), count (8), bar (20), pct (5)
        header = f"  [dim]{'':14}  {'Count':>8}  {'':20}  {'%':>5}[/dim]"
        lines.append(header)
        lines.append(f"  [dim]{'─' * 52}[/dim]")

        for key, label, color in segments_with_other:
            count = counts_full.get(key, 0)
            pct = count / total * 100 if total else 0.0
            mini_width = 20
            filled = round(pct / 100 * mini_width)
            mini_bar = (
                f"[{color}]{'█' * filled}[/{color}]"
                f"[dim]{'░' * (mini_width - filled)}[/dim]"
            )
            swatch = f"[on {color}]  [/on {color}]"
            lines.append(
                f"  {swatch} [bold {color}]{label:<14}[/bold {color}]"
                f"  {count:>8,}  {mini_bar}  {pct:>4.1f}%"
            )

        return "\n".join(lines)


class TagCoveragePanel(VerticalScroll):
    """Per-tag coverage table: how many library files carry each enabled tag.

    Each row shows tag description, file count, a mini progress bar, and the
    coverage percentage.  Colour coding gives an instant health read:
      green  — ≥ 90 % covered
      yellow — ≥ 50 % covered
      red    — < 50 % covered

    Only enabled tags are shown; disabled tags are omitted entirely.
    Scrollable so all tags are reachable regardless of terminal height.
    """

    DEFAULT_CSS = """
    TagCoveragePanel {
        border: round $primary-darken-2;
        padding: 1 2;
    }
    TagCoveragePanel > Static {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("", id="tag-coverage-inner")

    def update_stats(
        self,
        stats: dict,
        enabled_tags: list,
        total: int,
    ) -> None:
        """Re-render using the latest per-tag counts."""
        try:
            self.query_one("#tag-coverage-inner", Static).update(
                self._build_markup(stats, enabled_tags, total)
            )
        except Exception:
            pass

    def _build_markup(self, stats: dict, enabled_tags: list, total: int) -> str:
        lines: list[str] = ["[bold cyan]▸ Tag Coverage[/bold cyan]\n"]

        if total == 0 or not enabled_tags:
            lines.append("[dim]No files in library yet.[/dim]")
            return "\n".join(lines)

        bar_width = 18
        header = f"[dim]{'Tag':<22}  {'Count':>7}  {'':>{bar_width}}  {'%':>5}[/dim]"
        lines.append(header)
        lines.append(f"[dim]{'─' * (22 + 7 + bar_width + 8)}[/dim]")

        per_tag: dict[str, int] = stats.get("per_tag", {})
        for tag in enabled_tags:
            count = per_tag.get(tag.name, 0)
            pct = count / total * 100 if total else 0.0

            if pct >= 90:  # noqa: PLR2004
                color = "green"
            elif pct >= 50:  # noqa: PLR2004
                color = "yellow"
            else:
                color = "red"

            filled = round(pct / 100 * bar_width)
            bar = (
                f"[{color}]{'█' * filled}[/{color}]"
                f"[dim]{'░' * (bar_width - filled)}[/dim]"
            )
            # Truncate long descriptions so the table stays aligned.
            label = tag.description[:21]
            lines.append(
                f"[{color}]{label:<22}[/{color}]  {count:>7,}  {bar}  {pct:>4.1f}%"
            )

        return "\n".join(lines)


class StoragePanel(Static):
    """Storage and database health summary.

    Shows library disk usage, cache database sizes, embeddings cache depth,
    fingerprint coverage, and the most common error messages.  Intended to
    surface at a glance whether the embeddings cache is warm, the databases
    are growing unexpectedly, or a recurring error needs attention.

    Updated every 5 s (orchestrator tick) rather than every 0.5 s because
    the filesystem stat calls and SQL aggregates are heavier than the
    in-memory panel refreshes.
    """

    DEFAULT_CSS = """
    StoragePanel {
        border: round $primary-darken-2;
        padding: 1 2;
        height: auto;
    }
    """

    def update_stats(
        self,
        library_bytes: int,
        cache_db_bytes: int,
        embeddings_db_bytes: int,
        emb_stats: dict[str, int],
        fingerprinted: int,
        total: int,
        error_summary: list[tuple[str, int]],
    ) -> None:
        """Re-render with the latest storage snapshot."""
        self.update(
            self._build_markup(
                library_bytes,
                cache_db_bytes,
                embeddings_db_bytes,
                emb_stats,
                fingerprinted,
                total,
                error_summary,
            )
        )

    def _build_markup(
        self,
        library_bytes: int,
        cache_db_bytes: int,
        embeddings_db_bytes: int,
        emb_stats: dict[str, int],
        fingerprinted: int,
        total: int,
        error_summary: list[tuple[str, int]],
    ) -> str:
        lines: list[str] = ["[bold cyan]▸ Storage & Cache[/bold cyan]\n"]

        # ── Library & DB sizes ───────────────────────────────────────────────
        lines.append(
            f"[dim]Library size[/dim]:        [blue]{_fmt_bytes(library_bytes)}[/blue]"
            f"   [dim]cache.db[/dim]: [blue]{_fmt_bytes(cache_db_bytes)}[/blue]"
            f"   [dim]embeddings.db[/dim]: [blue]{_fmt_bytes(embeddings_db_bytes)}[/blue]"
        )

        # ── Embeddings cache ─────────────────────────────────────────────────
        total_emb = emb_stats.get("total_embeddings", 0)
        unique_fp = emb_stats.get("unique_fingerprints", 0)
        fp_pct = fingerprinted / total * 100 if total else 0.0
        fp_color = "green" if fp_pct >= 80 else "yellow" if fp_pct >= 40 else "red"  # noqa: PLR2004
        lines.append(
            f"[dim]Embeddings stored[/dim]:   [blue]{total_emb:,}[/blue]"
            f"   [dim]Unique fingerprints[/dim]: [blue]{unique_fp:,}[/blue]"
            f"   [dim]Fingerprint coverage[/dim]: "
            f"[{fp_color}]{fingerprinted:,} / {total:,}  ({fp_pct:.1f}%)[/{fp_color}]"
        )

        # ── Error summary ────────────────────────────────────────────────────
        if error_summary:
            lines.append("")
            lines.append("[dim]Top errors[/dim]:")
            for msg, count in error_summary:
                # Truncate long messages so they fit on one line.
                short = msg[:72] + "…" if len(msg) > 72 else msg  # noqa: PLR2004
                lines.append(f"  [red]×{count}[/red]  [dim]{short}[/dim]")

        return "\n".join(lines)


# ── Application ────────────────────────────────────────────────────────────────


class MusicTaggerApp(App):
    """Music library tag analysis pipeline."""

    TITLE = "Music Tagger"
    SUB_TITLE = "library tag pipeline"

    CSS = """
    Screen {
        background: $background;
    }

    #stats-bar {
        height: 3;
        background: $surface;
        padding: 0 2;
        border-bottom: solid $primary-darken-3;
        content-align: left middle;
    }

    #panels {
        height: 18;
        margin: 1 0 0 0;
        padding: 0 1;
    }

    #bottom-view {
        height: 1fr;
    }

    #log-filters {
        height: 3;
        margin: 0 2;
        padding: 0 1;
        background: $surface;
        border: round $primary-darken-2;
        align: left middle;
    }

    #log-filters Checkbox {
        margin: 0 2 0 0;
        background: transparent;
        border: none;
        padding: 0 1;
    }

    #log-filters.hidden {
        display: none;
    }

    #activity-log {
        border: round $primary-darken-2;
        margin: 0 2 1 2;
        height: 1fr;
    }

    #overview-pane {
        display: none;
        height: 1fr;
    }

    #overview-pane.visible {
        display: block;
    }

    #overview-top {
        height: 1fr;
        min-height: 16;
    }

    #library-overview {
        width: 1fr;
        height: 1fr;
        margin: 1 1 1 2;
    }

    #tag-coverage {
        width: 1fr;
        height: 1fr;
        margin: 1 2 1 1;
    }

    #storage-panel {
        margin: 0 2 1 2;
    }

    #quit-overlay {
        background: $surface 90%;
        width: 100%;
        height: 100%;
        content-align: center middle;
        text-align: center;
        text-style: bold;
        color: $warning;
        layer: overlay;
        display: none;
    }

    #quit-overlay.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "force_scan", "Force Scan"),
        Binding("i", "force_inspect", "Force Inspect"),
        Binding("c", "run_cleanup", "Cleanup"),
        Binding("r", "requeue_errors", "Requeue Errors"),
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("tab", "cycle_view", "Log/Overview", priority=True, key_display="Tab"),
    ]

    paused: reactive[bool] = reactive(False)
    # "log" shows the activity log; "overview" shows the library chart.
    _view: reactive[str] = reactive("log")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.cache = FileCache(config.db_path)

        # Each job gets a thread-safe log callback that posts a Textual message
        self.scanner = Scanner(config, self.cache, self._make_log("scanner"))
        self.inspector = Inspector(config, self.cache, self._make_log("inspector"))
        self.worker = Worker(
            config,
            self.cache,
            self._make_log("worker"),
            self._make_log_markup("worker"),
        )
        self.cleanup = Cleanup(config, self.cache, self._make_log("cleanup"))

        self._last_scan: float | None = None
        self._last_cleanup: float | None = None
        # Log filter state: which sources are currently visible.
        # All sources are on by default.  "app" is always shown (system msgs).
        self._active_sources: set[str] = {
            "scanner",
            "inspector",
            "worker",
            "cleanup",
            "app",
        }
        # Rolling buffer of (source, formatted_markup) tuples — mirrors the
        # RichLog so we can redraw it when the user toggles a filter.  Capped
        # at the same max_lines as the RichLog widget to bound memory use.
        self._log_buffer: deque[tuple[str, str]] = deque(maxlen=1000)
        # Trigger an immediate scan on first run; subsequent runs follow the cron schedule.
        self._next_scan: float = time.time()
        # First cleanup fires on the next scheduled cron tick after startup.
        self._next_cleanup: float = _cron_next(config.cleanup_cron)
        # Spinner frame counter — incremented on every panel refresh
        self._spin_frame: int = 0
        # Tags active under the current config — computed once and reused so
        # stats(), needs_inspection(), and needs_work() all use the same filter.
        self._enabled_tags = [t for t in TAGS if config.tag_cfg(t.name).enabled]
        # Read-only embeddings cache connection for the overview panel stats.
        # Opened separately from the worker's instance so panel refreshes never
        # block on the worker's lock.
        self._emb_cache = EmbeddingCache(config.embeddings_db_path)
        # Counter used to throttle expensive overview stats (file sizes, SQL
        # aggregates) to once every ~5 s instead of every 0.5 s panel refresh.
        self._overview_tick: int = 0
        # Set to True once the user requests quit so action_quit is not re-entered.
        self._quitting: bool = False
        # Monotonic timestamp of when quit was requested; used by _poll_quit to
        # enforce a hard exit deadline so a long-running inference pass does not
        # prevent the app from closing.
        self._quit_time: float = 0.0
        # Last rendered stats-bar text — used to skip redundant Static.update()
        # calls when nothing has changed between 0.5 s ticks.
        self._stats_bar_text: str = ""
        # Monotonic timestamp of the last successful StatsUpdate.  Used to
        # surface a staleness warning in the stats bar when the background
        # refresh stops posting updates (e.g. due to a repeated exception in
        # _refresh_panels_inner silently swallowed by the guard, or a blocked
        # stats query thread).
        self._stats_last_updated: float = 0.0
        # Guard flag: True while a background cache.stats() call is in flight.
        # _schedule_stats_refresh checks this and skips launching a new fetch
        # rather than cancelling the in-progress one.  Cancellation via
        # exclusive=True would kill a stats query mid-execution if the WAL
        # checkpoint stalls the lock for longer than the 1 s timer interval,
        # causing the StatsUpdate to never be posted and the display to freeze.
        self._stats_refresh_running: bool = False
        # Monotonic timestamp of the last _refresh_panels_inner exception, used
        # to rate-limit TUI activity-log noise to at most once per minute.
        self._panels_error_last_logged: float = 0.0
        # Read model for the main thread: holds the last result of cache.stats().
        # Refreshed every ~1 s from a background thread via _schedule_stats_refresh.
        # All main-thread code (panel refresh, orchestrator) reads from this dict
        # and never calls cache.stats() directly — keeping the event loop free.
        self._stats_cache: dict = {
            "total": 0,
            "needs_inspection": 0,
            "needs_work": 0,
            "in_progress": 0,
            "errors": 0,
            "done": 0,
            "per_tag": {t.name: 0 for t in TAGS},
        }

    # ── Layout ─────────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="stats-bar")
        with Horizontal(id="panels"):
            yield StatusPanel("▸ Scanner", id="panel-scanner")
            yield StatusPanel("▸ Inspector", id="panel-inspector")
            yield StatusPanel("▸ Worker", id="panel-worker")
            yield StatusPanel("▸ Cleanup", id="panel-cleanup")
        with Container(id="bottom-view"):
            with Horizontal(id="log-filters"):
                yield Checkbox("Scanner", value=True, id="filter-scanner")
                yield Checkbox("Inspector", value=True, id="filter-inspector")
                yield Checkbox("Worker", value=True, id="filter-worker")
                yield Checkbox("Cleanup", value=True, id="filter-cleanup")
            yield ActivityLog(
                id="activity-log",
                highlight=False,  # highlighting is applied manually in _log_markup via ReprHighlighter on msg only
                markup=True,
                wrap=True,
                max_lines=1000,
            )
            with Container(id="overview-pane"):
                with Horizontal(id="overview-top"):
                    yield LibraryOverview(id="library-overview")
                    yield TagCoveragePanel(id="tag-coverage")
                yield StoragePanel(id="storage-panel")
        yield Footer()
        yield Label("Finishing up…  please wait", id="quit-overlay")

    def on_mount(self) -> None:
        self._applog(f"Started  library={self.config.music_path}")
        self._applog(f"Database={self.config.db_path}")
        self._applog(f"Tags monitored: {', '.join(t.description for t in TAGS)}")
        # Stats are fetched on a background thread and stored in _stats_cache.
        # Fire immediately so panels have real data on first render, then repeat
        # every 1 s.  Panel refresh (0.5 s) and orchestration (5 s) read from
        # _stats_cache only — they never call cache.stats() on the main thread.
        self._schedule_stats_refresh()
        self.set_interval(1.0, self._schedule_stats_refresh)
        self.set_interval(0.5, self._refresh_panels)
        self.set_interval(5.0, self._orchestrate)
        self._refresh_panels()

    # ── Stats cache ────────────────────────────────────────────────────────────
    # The main thread never calls cache.stats() directly.  Instead, a 1 s
    # background worker fetches the result and posts it back as a StatsUpdate
    # message via post_message (non-blocking, thread-safe).  All panel-refresh
    # and orchestration code reads from _stats_cache (a plain dict, instant,
    # no I/O).
    #
    # Rule for future changes: if you need stats data on the main thread,
    # read self._stats_cache — never call self.cache.stats() inline.
    #
    # Use post_message rather than call_from_thread for all background→UI
    # communication.  post_message is thread-safe via call_soon_threadsafe and
    # is fire-and-forget — the background thread never blocks waiting for the
    # main thread to process the result.  call_from_thread blocks the caller.

    def _schedule_stats_refresh(self) -> None:
        """Launch a background fetch of cache.stats() if one isn't already running.

        Uses a plain boolean flag rather than Textual's exclusive=True so that
        an in-progress fetch is never cancelled mid-query.  exclusive=True
        would kill the previous asyncio task on every 1 s tick; because thread
        workers run via run_in_executor the cancellation only discards the
        result — the OS thread keeps running and holds the SQLite lock until
        the query finishes.  If the lock is stalled by a WAL checkpoint (which
        can take 2–3 s), the stats query exceeds the 1 s interval and every
        tick perpetually cancels the prior result, so StatsUpdate is never
        posted and the display freezes permanently.

        The flag approach lets the slow query finish and deliver its result
        even when it spans multiple timer ticks.
        """
        if self._quitting:
            return
        # Skip — a fetch is already in flight; let it finish undisturbed.
        if self._stats_refresh_running:
            return
        self._stats_refresh_running = True

        def _fetch() -> None:
            try:
                result = self.cache.stats(enabled_tags=self._enabled_tags)
                # post_message is thread-safe and non-blocking; the background
                # thread does not wait for the main thread to process the update.
                self.post_message(StatsUpdate(result))
            except Exception as exc:
                # Non-fatal: the stale cache value stays in place until the
                # next successful fetch.
                logger.warning("Stats refresh error (non-fatal): {}", exc)
            finally:
                # Always clear the flag so the next timer tick can launch a
                # fresh fetch, even if stats() raised an exception.
                self._stats_refresh_running = False

        self.run_worker(
            _fetch,
            thread=True,
            group="stats_refresh",
            exclusive=False,
            exit_on_error=False,
        )

    def on_stats_update(self, event: StatsUpdate) -> None:
        """Apply a freshly fetched stats snapshot. Always called on the main thread."""
        self._stats_cache = event.stats
        self._stats_last_updated = time.monotonic()

    def _schedule_storage_refresh(self, total: int) -> None:
        """Launch a background fetch of storage/embeddings stats."""
        if self._quitting:
            return

        def _fetch() -> None:
            try:
                library_bytes = self.cache.library_size_bytes()
                cache_db_bytes = _file_size(self.config.db_path)
                embeddings_db_bytes = _file_size(self.config.embeddings_db_path)
                emb_stats = self._emb_cache.stats()
                fingerprinted = self.cache.fingerprinted_count()
                error_summary = self.cache.error_summary()
                self.post_message(
                    StoragePanelUpdate(
                        library_bytes,
                        cache_db_bytes,
                        embeddings_db_bytes,
                        emb_stats,
                        fingerprinted,
                        total,
                        error_summary,
                    )
                )
            except Exception as exc:
                logger.warning("Storage panel refresh error (non-fatal): {}", exc)

        self.run_worker(
            _fetch,
            thread=True,
            group="storage_refresh",
            exclusive=True,
            exit_on_error=False,
        )

    def on_storage_panel_update(self, event: StoragePanelUpdate) -> None:
        """Apply a freshly fetched storage snapshot. Always called on the main thread."""
        try:
            self.query_one("#storage-panel", StoragePanel).update_stats(
                event.library_bytes,
                event.cache_db_bytes,
                event.embeddings_db_bytes,
                event.emb_stats,
                event.fingerprinted,
                event.total,
                event.error_summary,
            )
        except Exception:
            pass  # Widget not mounted yet — will retry on next tick

    # ── Panel refresh ──────────────────────────────────────────────────────────

    def _refresh_panels(self) -> None:
        # Guard: any unhandled exception here propagates to the Textual event
        # loop and triggers app.panic() → process exit.  Swallow and log instead.
        try:
            self._refresh_panels_inner()
        except Exception as exc:
            logger.warning("Panel refresh error (non-fatal): {}", exc)
            # Also surface the error in the TUI activity log, but rate-limit to
            # once per minute so a persistent exception doesn't flood the log.
            now = time.monotonic()
            if now - self._panels_error_last_logged > 60.0:
                self._panels_error_last_logged = now
                self._applog_markup(
                    f"[red]Panel refresh error (display may be stale): {exc}[/red]"
                )

    def _refresh_panels_inner(self) -> None:
        # Read from the cached stats dict — never call cache.stats() here.
        # The background stats-refresh worker (_schedule_stats_refresh) keeps
        # this up to date at ~1 s intervals without touching the main thread.
        stats = self._stats_cache
        self._spin_frame += 1

        # Stats bar — only re-render when the content actually changes.
        paused_tag = "  [yellow]⏸ PAUSED[/yellow]" if self.paused else ""
        # Warn if the background stats refresh hasn't delivered an update in
        # over 15 seconds.  This makes a frozen display self-diagnosing: the
        # user can see the warning rather than silently watching stale numbers.
        # _stats_last_updated is 0.0 on first mount so we skip the warning
        # until at least one successful fetch has been delivered.
        stats_age = time.monotonic() - self._stats_last_updated
        stale_tag = (
            "  [red bold]⚠ STATS STALE[/red bold]"
            if self._stats_last_updated > 0.0 and stats_age > 15.0
            else ""
        )
        stats_bar_text = (
            f"[dim]Library[/dim] [cyan]{self.config.music_path}[/cyan]"
            f"   [dim]Total[/dim] [green]{stats['total']:,}[/green]"
            f"   [dim]Needs inspection[/dim] [yellow]{stats['needs_inspection']:,}[/yellow]"
            f"   [dim]Needs work[/dim] [red]{stats['needs_work']:,}[/red]"
            f"   [dim]Done[/dim] {stats['done']:,}"
            f"   [dim]Errors[/dim] [red]{stats['errors']:,}[/red]"
            + paused_tag
            + stale_tag
        )
        if stats_bar_text != self._stats_bar_text:
            self._stats_bar_text = stats_bar_text
            self.query_one("#stats-bar", Static).update(stats_bar_text)

        scanner_rate = self.scanner.pass_rate
        self.query_one("#panel-scanner", StatusPanel).set_rows(
            {
                "Status": _status_glyph(self.scanner.running, self._spin_frame),
                "Last run": _fmt_ago(self._last_scan),
                "Next run": _fmt_next_run(self._next_scan),
                "Schedule": self.config.scan_cron,
                "File throttle": f"{self.config.file_throttle_ms} ms",
                "Dir throttle": f"{self.config.dir_throttle_ms} ms",
                "Scanned": f"{self.scanner.files_scanned:,}",
                "Changed": f"{self.scanner.files_changed:,}",
                "Rate": _fmt_rate(scanner_rate),
            }
        )

        inspector_rate = self.inspector.pass_rate
        self.query_one("#panel-inspector", StatusPanel).set_rows(
            {
                "Status": _status_glyph(self.inspector.running, self._spin_frame),
                "Queue": f"{stats['needs_inspection']:,}",
                "Inspected": f"{self.inspector.inspected:,}",
                "Queued work": f"{self.inspector.queued:,}",
                "Errors": f"{self.inspector.errors}",
                "Throttle": f"{self.config.inspector_throttle_ms} ms",
                "Batch size": str(self.config.inspector_batch_size),
                "Rate": _fmt_rate(inspector_rate),
                "ETA": _fmt_eta(stats["needs_inspection"], inspector_rate),
            }
        )

        worker_rate = self.worker.pass_rate
        self.query_one("#panel-worker", StatusPanel).set_rows(
            {
                "Status": _status_glyph(self.worker.running, self._spin_frame),
                "Queue": f"{stats['needs_work']:,}",
                "In progress": f"{stats.get('in_progress', 0):,}",
                "Batch size": str(self.config.worker_batch_size),
                "Batch progress": _progress_bar(
                    self.worker.batch_done, self.worker.batch_total
                ),
                "Tagged this session": f"{self.worker.processed:,}",
                "Errors (session)": f"{self.worker.errors:,}",
                "Errors (total)": f"{stats['errors']:,}",
                "Rate": _fmt_rate(worker_rate),
                "ETA": _fmt_eta(stats["needs_work"], worker_rate),
            }
        )

        cleanup_rate = self.cleanup.pass_rate
        cleanup_remaining = self.cleanup.pass_total - self.cleanup.pass_checked
        cleanup_next_str = _fmt_next_run(self._next_cleanup)
        self.query_one("#panel-cleanup", StatusPanel).set_rows(
            {
                "Status": _status_glyph(self.cleanup.running, self._spin_frame),
                "Last run": _fmt_ago(self._last_cleanup),
                "Next run": cleanup_next_str,
                "Schedule": self.config.cleanup_cron,
                "Last removed": str(self.cleanup.last_removed),
                "Rate": _fmt_rate(cleanup_rate),
                "ETA": _fmt_eta(cleanup_remaining, cleanup_rate),
            }
        )

        # total is needed both for the overview widgets and the storage refresh
        # below, so extract it unconditionally before the display guard.
        total = stats["total"]

        # Overview widgets — only update when the pane is actually visible to
        # avoid triggering layout reflows (height: auto) on every 0.5 s tick
        # while the user is on the log view.  A single update runs when the
        # user switches to the overview view (watch__view calls _refresh_panels).
        try:
            overview_pane = self.query_one("#overview-pane")
            if overview_pane.display:
                self.query_one("#library-overview", LibraryOverview).update_stats(stats)
                self.query_one("#tag-coverage", TagCoveragePanel).update_stats(
                    stats, self._enabled_tags, total
                )
        except Exception:
            pass  # Widget not mounted yet — will catch up on the next tick

        # Storage & cache stats are heavier (file stat + SQL aggregates).
        # Kick off a background fetch every ~5 s (10 × 0.5 s ticks); the
        # result arrives as a StoragePanelUpdate message handled below.
        self._overview_tick += 1
        if self._overview_tick % 10 == 1:
            self._schedule_storage_refresh(total)

    # ── Orchestration ──────────────────────────────────────────────────────────

    def _orchestrate(self) -> None:
        """Called every 5 s — starts jobs according to schedule."""
        # Guard: any unhandled exception here propagates to the Textual event
        # loop and triggers app.panic() → process exit.  Swallow and log instead.
        try:
            self._orchestrate_inner()
        except Exception as exc:
            logger.warning("Orchestrate error (non-fatal): {}", exc)

    def _orchestrate_inner(self) -> None:
        if self.paused:
            return

        now = time.time()

        # Scanner watchdog — mirrors the worker watchdog.  The scanner updates
        # last_activity on every file and every directory listing.  If no
        # heartbeat is seen for _SCANNER_HANG_TIMEOUT_S the scan thread is
        # stuck inside os.scandir() (NFS stall that even the per-directory
        # timeout thread couldn't unblock in time).  Force-clear the running
        # flag so the orchestrator relaunches on the next tick.
        if self.scanner.running:
            idle_s = time.monotonic() - self.scanner.last_activity
            if idle_s > _SCANNER_HANG_TIMEOUT_S:
                self._applog_markup(
                    f"[red]Scanner watchdog: no heartbeat for {int(idle_s)}s — "
                    f"resetting running flag and relaunching[/red]"
                )
                logger.warning(
                    "Scanner watchdog: no heartbeat for {}s — force-resetting running flag",
                    int(idle_s),
                )
                self.scanner.stop()

        # Scanner — run on schedule.
        # _next_scan may be inf after a manual run (action_force_scan sets it
        # to inf to prevent an immediate re-trigger).  Once the job finishes
        # we compute the real next cron tick here.
        if self._next_scan == float("inf") and not self.scanner.running:
            self._next_scan = _cron_next(self.config.scan_cron)

        if now >= self._next_scan and not self.scanner.running:
            self._last_scan = now
            self._next_scan = _cron_next(self.config.scan_cron)
            self.run_worker(
                self.scanner.run_pass,
                thread=True,
                group="scanner",
                exclusive=True,
                exit_on_error=False,
            )

        # Inspector — run continuously while there's a queue
        if not self.inspector.running:
            if self._stats_cache["needs_inspection"] > 0:
                self.run_worker(
                    self.inspector.run_pass,
                    thread=True,
                    group="inspector",
                    exclusive=True,
                    exit_on_error=False,
                )

        # Worker — run continuously while there's a queue.
        # Watchdog: if the worker claims to be running but its last heartbeat is
        # older than the hang timeout, the thread is stuck (NFS stall, hung
        # TensorFlow graph, etc.).  Force-clear the running flag so the
        # orchestrator can launch a fresh pass on the next tick.  The stuck
        # thread will eventually be killed by Textual when the exclusive group
        # launches a new worker.
        if self.worker.running:
            idle_s = time.monotonic() - self.worker.last_activity
            if idle_s > _WORKER_HANG_TIMEOUT_S:
                self._applog_markup(
                    f"[red]Worker watchdog: no heartbeat for {int(idle_s)}s — "
                    f"resetting running flag and relaunching[/red]"
                )
                logger.warning(
                    "Worker watchdog: no heartbeat for {}s — force-resetting running flag",
                    int(idle_s),
                )
                self.worker.stop()

                # Recover any 'working' row the hung pass left behind — done
                # in a background thread so the DB write + commit don't block
                # the orchestrator's event-loop callback.
                def _recover() -> None:
                    try:
                        recovered = self.cache.requeue_working()
                        if recovered:
                            self.cache.flush()
                            logger.warning(
                                "Worker watchdog: requeued {} stuck 'working' row(s)",
                                recovered,
                            )
                    except Exception as exc:
                        logger.warning("Worker watchdog recovery failed: {}", exc)

                self.run_worker(
                    _recover, thread=True, group="watchdog_recover", exit_on_error=False
                )

        if not self.worker.running:
            # Recover any rows left in 'working' status from a previous pass
            # that exited without committing (e.g. worker thread exception, OS
            # signal mid-batch, or a watchdog reset).  requeue_working() only
            # touches 'working' rows, so it is safe to call even when no rows
            # are stuck — the UPDATE is a no-op.  Run in a background thread to
            # keep the event-loop callback free of blocking DB I/O.
            # Only do this when the worker is truly idle (not running) so we
            # don't race with a batch that's still in flight.
            if self._stats_cache.get("in_progress", 0) > 0 and not self._quitting:

                def _requeue_stuck() -> None:
                    try:
                        recovered = self.cache.requeue_working()
                        if recovered:
                            self.cache.flush()
                            logger.warning(
                                "Orchestrator: requeued {} stuck 'working' row(s) "
                                "left by a previous pass",
                                recovered,
                            )
                    except Exception as exc:
                        # Non-fatal: rows remain in 'working' status and will
                        # be recovered again on the next orchestrator tick or
                        # at next startup.
                        logger.warning("Stuck-working recovery failed: {}", exc)

                self.run_worker(
                    _requeue_stuck,
                    thread=True,
                    group="requeue_stuck",
                    exclusive=True,
                    exit_on_error=False,
                )

            if self._stats_cache["needs_work"] > 0 and not self._quitting:
                # Clear any previous stop_requested so the loop can run.
                # Only do this when not quitting — during shutdown the flag
                # must stay set so run_pass() refuses to re-enter.
                self.worker._stop_requested = False

                def _worker_loop() -> None:
                    # Run consecutive passes until the queue is empty or the
                    # worker is stopped externally (e.g. pause, shutdown).
                    # Looping here avoids the up-to-5 s orchestrator gap
                    # between batches while the ML models are already warm in
                    # memory — the models are the expensive part to load, not
                    # individual batch dispatch.
                    #
                    # Exit conditions:
                    #   - run_pass() returns 0: queue is empty, nothing left.
                    #   - _stop_requested is set: stop() was called between
                    #     passes (pause, quit, watchdog).  Checked on the
                    #     while guard rather than the break so that a stop()
                    #     arriving while run_pass() is executing is caught at
                    #     the next iteration boundary without needing to
                    #     inspect _running (which run_pass() always sets False
                    #     before returning, making it useless as a loop-exit
                    #     signal for the normal completion case).
                    while not self.worker._stop_requested:
                        processed = self.worker.run_pass(self.config.worker_batch_size)
                        if processed == 0:
                            break

                self.run_worker(
                    _worker_loop,
                    thread=True,
                    group="worker",
                    exclusive=True,
                    exit_on_error=False,
                )

        # Cleanup — run on schedule.
        # _next_cleanup may be inf after a manual run (action_run_cleanup sets
        # it to inf to prevent an immediate re-trigger).  Once the job finishes
        # we compute the real next cron tick here.
        if self._next_cleanup == float("inf") and not self.cleanup.running:
            self._next_cleanup = _cron_next(self.config.cleanup_cron)

        # Cleanup only stat()s files and removes orphan DB rows; it is safe to
        # run alongside the scanner and inspector, which use the same
        # lock-protected SQLite connection.  No idle-pipeline gate is needed.
        if now >= self._next_cleanup and not self.cleanup.running:
            self._last_cleanup = now
            self._next_cleanup = _cron_next(self.config.cleanup_cron)
            self.run_worker(
                self.cleanup.run,
                thread=True,
                group="cleanup",
                exclusive=True,
                exit_on_error=False,
            )

    # ── Actions ────────────────────────────────────────────────────────────────

    def action_force_scan(self) -> None:
        if self.scanner.running:
            self._applog("Scanner already running")
            return
        self._applog("Forcing scan…")
        self._last_scan = time.time()
        # Push _next_scan far into the future so the orchestrator does not
        # immediately re-trigger once the job finishes.  The orchestrator will
        # compute the real next cron tick once this run completes.
        self._next_scan = float("inf")
        self.run_worker(
            self.scanner.run_pass,
            thread=True,
            group="scanner",
            exclusive=True,
            exit_on_error=False,
        )

    def action_force_inspect(self) -> None:
        if self.inspector.running:
            self._applog("Inspector already running")
            return
        self._applog("Forcing inspection pass…")
        self.run_worker(
            self.inspector.run_pass,
            thread=True,
            group="inspector",
            exclusive=True,
            exit_on_error=False,
        )

    def action_run_cleanup(self) -> None:
        if self.cleanup.running:
            self._applog("Cleanup already running")
            return
        self._applog("Running cleanup…")
        self._last_cleanup = time.time()
        # Push _next_cleanup far into the future so the orchestrator does not
        # immediately re-trigger once the job finishes.  The orchestrator will
        # compute the real next cron tick once this run completes.
        self._next_cleanup = float("inf")
        self.run_worker(
            self.cleanup.run,
            thread=True,
            group="cleanup",
            exclusive=True,
            exit_on_error=False,
        )

    def action_requeue_errors(self) -> None:
        def _requeue() -> None:
            try:
                count = self.cache.requeue_errors()
                self.cache.flush()
                if count:
                    self.post_message(
                        LogEvent(
                            "app",
                            f"Requeued {count} error(s) — worker will retry on next pass",
                        )
                    )
                else:
                    self.post_message(
                        LogEvent("app", "No error rows with missing tags to requeue")
                    )
            except Exception as exc:
                logger.warning("requeue_errors failed: {}", exc)

        self.run_worker(
            _requeue, thread=True, group="requeue_errors", exit_on_error=False
        )

    def action_cycle_view(self) -> None:
        """Toggle between the activity log and the library overview."""
        self._view = "overview" if self._view == "log" else "log"

    def watch__view(self, view: str) -> None:
        """Show the active bottom pane and hide the other."""
        try:
            log = self.query_one("#activity-log", ActivityLog)
            overview = self.query_one("#overview-pane", Container)
        except Exception:
            return  # Widgets not mounted yet
        try:
            filters = self.query_one("#log-filters", Horizontal)
        except Exception:
            filters = None
        if view == "overview":
            log.display = False
            overview.add_class("visible")
            if filters is not None:
                filters.add_class("hidden")
            # Populate overview widgets immediately rather than waiting up to
            # 0.5 s for the next panel-refresh tick.
            self._refresh_panels()
        else:
            log.display = True
            overview.remove_class("visible")
            if filters is not None:
                filters.remove_class("hidden")
            # Re-render buffered lines now that the widget has its true width.
            # Lines written while the widget was hidden (or before its first
            # resize) were rendered at min_width=78; _redraw_log() replays them
            # at the current widget width so wrapping is correct.
            self._redraw_log()

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._applog_markup(
                "[yellow]⏸  Paused — jobs will finish their current pass then stop[/yellow]"
            )
            self.scanner.stop()
            self.inspector.stop()
            self.worker.stop()
            self.cleanup.stop()
        else:
            self._applog_markup("[green]▶  Resumed[/green]")

    # ── Graceful quit ──────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        """Stop all background jobs, show a waiting overlay, then exit.

        Overrides Textual's default action_quit so the user sees a "Finishing
        up…" message while background threads observe their _running flag and
        wind down.  Without this the TUI disappears instantly but the process
        hangs silently until inference or I/O completes.
        """
        if self._quitting:
            return
        self._quitting = True
        self._quit_time = time.monotonic()

        # Signal every pipeline stage to stop at the next iteration boundary.
        self.scanner.stop()
        self.inspector.stop()
        self.worker.stop()
        self.cleanup.stop()

        any_running = (
            self.scanner.running
            or self.inspector.running
            or self.worker.running
            or self.cleanup.running
        )

        if not any_running:
            # Nothing was active — exit immediately with no overlay.
            self.exit()
            return

        # Show the overlay and poll until all threads are done.
        try:
            self.query_one("#quit-overlay", Label).add_class("visible")
        except Exception:
            pass  # Overlay widget not ready — proceed without it

        self.set_interval(0.1, self._poll_quit)

    # Maximum seconds to wait for background threads to stop after quit is
    # requested before forcing an immediate exit anyway.  The in-flight file's
    # DB row is left as 'working'; startup recovery requeues it automatically.
    _QUIT_TIMEOUT_S: int = 10

    def _poll_quit(self) -> None:
        """Periodic check — exit once all background threads have stopped.

        If threads have not stopped within _QUIT_TIMEOUT_S seconds of the quit
        request, force an immediate exit rather than waiting indefinitely for a
        long-running inference pass (BPM, Essentia) to finish on its own.
        """
        all_stopped = (
            not self.scanner.running
            and not self.inspector.running
            and not self.worker.running
            and not self.cleanup.running
        )
        timed_out = (time.monotonic() - self._quit_time) > self._QUIT_TIMEOUT_S

        if not all_stopped and not timed_out:
            return

        if timed_out and not all_stopped:
            logger.warning(
                "Quit timeout ({}s) reached — forcing exit; "
                "any in-flight work will be requeued on next startup",
                self._QUIT_TIMEOUT_S,
            )

        self.worker.close()
        self.exit()

    # ── Worker error handling ──────────────────────────────────────────────────

    def on_worker_state_changed(self, event: object) -> None:
        """Log any background job errors to loguru.

        With exit_on_error=False, Textual swallows exceptions from thread
        workers instead of crashing the app.  This handler catches those
        ERROR-state transitions and routes them to the loguru log so they
        are not lost entirely.

        This handler must never raise — it runs on the main event-loop thread
        and an unhandled exception here would reach app.panic().
        """
        try:
            from textual.worker import WorkerState  # deferred: only needed here

            # event is a Worker.StateChanged message; access via attributes.
            worker = getattr(event, "worker", None)
            if worker is None:
                return
            if getattr(worker, "state", None) is not WorkerState.ERROR:
                return
            error = getattr(worker, "_error", None)
            logger.error(
                "Background job crashed ({}): {}",
                getattr(worker, "name", "unknown"),
                error,
            )
            self._applog_markup(
                f"[red]Background job error ({markup_escape(str(getattr(worker, 'name', '?')))}): "
                f"{markup_escape(str(error))}[/red]"
            )
        except Exception as exc:
            # Last-resort guard — never let this handler kill the app.
            logger.warning("on_worker_state_changed handler error: {}", exc)

    # ── Log helpers ────────────────────────────────────────────────────────────

    SOURCE_COLORS = {
        "scanner": "green",
        "inspector": "cyan",
        "worker": "yellow",
        "cleanup": "magenta",
        "app": "white",
    }

    def _applog(self, msg: str) -> None:
        """Write a plain-text app message to the activity log.

        The message is displayed literally — any Rich markup characters in
        *msg* are escaped before rendering so filenames like ``[silence].mp3``
        are shown as-is rather than interpreted as markup tags.

        Use ``_applog_markup`` when the message intentionally contains Rich
        markup (e.g. ``[red]…[/red]`` colour tags).
        """
        self._log("app", msg)

    def _applog_markup(self, msg: str) -> None:
        """Write a Rich-markup app message to the activity log.

        *msg* is rendered as Rich markup directly — caller is responsible for
        escaping any user-supplied content embedded in the string.  Only use
        this for messages constructed entirely from trusted, controlled strings
        (e.g. the watchdog alert, pause/resume glyphs).
        """
        self._log_markup("app", msg)

    def _log(self, source: str, msg: str) -> None:
        """Write a message to the activity log, escaping all user content.

        All pipeline log messages (filenames, paths, exception text) go through
        this method.  ``markup_escape`` neutralises any Rich markup characters
        in *msg* so that filenames like ``[silence].mp3`` or
        ``[Remastered].flac`` never reach the Rich markup parser and cannot
        raise ``MarkupError``.
        """
        self._log_markup(source, markup_escape(msg))

    # ReprHighlighter instance shared across all log calls — stateless and
    # thread-safe (read-only after construction).
    _highlighter: ReprHighlighter = ReprHighlighter()

    def _log_markup(self, source: str, msg: str) -> None:
        """Write a pre-trusted Rich markup string to the activity log.

        Builds a ``rich.text.Text`` object directly rather than a markup string
        so that:
          - The structural parts (timestamp, source label) are set as explicit
            spans — immune to markup parsing errors from user content.
          - ``msg`` is parsed as Rich markup (it is either pre-escaped plain
            text from ``_log``, or intentional markup from the worker's mood/
            BPM result lines).
          - ``ReprHighlighter`` is applied to the plain-text portion of ``msg``
            so numbers, repr strings, and paths are coloured as they were when
            ``highlight=True`` was set on the RichLog widget.

        Every line is stored in ``_log_buffer`` (as a markup string) so filter
        toggles can redraw the log without losing history.

        ``query_one`` raises ``NoMatches`` if the ``RichLog`` widget hasn't
        been mounted yet (early startup) or has already been unmounted
        (shutdown).  We swallow that so callers on the main thread can't panic
        the app.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        color = self.SOURCE_COLORS.get(source, "white")

        # Keep a markup string in the buffer for _redraw_log replay.
        # Brackets around the source name are escaped so they are literal text.
        source_label = markup_escape(f"[{source}]")
        line = f"[dim]{ts}[/dim] [{color}]{source_label}[/{color}] {msg}"
        self._log_buffer.append((source, line))

        if source not in self._active_sources:
            return

        # Build a Text object with explicit spans — no full-line markup parsing.
        # This is immune to MarkupError from special characters in file paths
        # or tag values, and lets us apply ReprHighlighter only to the msg body.
        try:
            renderable = Text()
            renderable.append(ts, style="dim")
            renderable.append(" ")
            renderable.append(f"[{source}]", style=color)
            renderable.append(" ")
            try:
                msg_text = Text.from_markup(msg)
            except Exception:
                # msg contained malformed markup — treat it as plain text.
                msg_text = Text(markup_escape(msg))
            self._highlighter(msg_text)
            renderable.append_text(msg_text)
            self.query_one("#activity-log", ActivityLog).write(renderable)
        except Exception:
            pass  # Widget not ready; message is lost but the app stays alive

    def on_log_event(self, event: LogEvent) -> None:
        try:
            if event.markup:
                self._log_markup(event.source, event.text)
            else:
                self._log(event.source, event.text)
        except Exception as exc:
            # Never let a log display error crash the app.
            logger.warning("on_log_event display error: {}", exc)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Toggle a log source filter and redraw the log."""
        # Map checkbox id → source name.
        source_map = {
            "filter-scanner": "scanner",
            "filter-inspector": "inspector",
            "filter-worker": "worker",
            "filter-cleanup": "cleanup",
        }
        checkbox_id = event.checkbox.id or ""
        source = source_map.get(checkbox_id)
        if source is None:
            return
        if event.value:
            self._active_sources.add(source)
        else:
            self._active_sources.discard(source)
        self._redraw_log()

    def _redraw_log(self) -> None:
        """Clear the RichLog and replay buffered lines through the active filter."""
        try:
            log = self.query_one("#activity-log", ActivityLog)
        except Exception:
            return
        log.clear()
        for source, line in self._log_buffer:
            if source in self._active_sources:
                log.write(line)

    def _make_log(self, source: str) -> Callable[[str], None]:
        """Returns a log function safe to call from a background thread.

        Uses ``post_message`` directly rather than ``call_from_thread`` because
        ``post_message`` is already thread-safe (it uses ``call_soon_threadsafe``
        internally) and is non-blocking — the background thread does not wait for
        the main thread to process the message.  ``call_from_thread`` wraps the
        call in a coroutine and blocks the caller until the event loop runs it,
        adding unnecessary latency on both sides.
        """

        def log_fn(msg: str) -> None:
            try:
                self.post_message(LogEvent(source, msg))
            except RuntimeError:
                pass  # App has already stopped; discard the log message

        return log_fn

    def _make_log_markup(self, source: str) -> Callable[[str], None]:
        """Returns a markup log function safe to call from a background thread.

        The message is treated as pre-trusted Rich markup and rendered without
        escaping.  Use only for strings built from safe, controlled content.

        See ``_make_log`` for why ``post_message`` is used instead of
        ``call_from_thread``.
        """

        def log_fn(msg: str) -> None:
            try:
                self.post_message(LogEvent(source, msg, markup=True))
            except RuntimeError:
                pass  # App has already stopped; discard the log message

        return log_fn

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def on_unmount(self) -> None:
        self.scanner.stop()
        self.inspector.stop()
        self.worker.stop()
        self.cleanup.stop()
        # Cancel any in-flight Textual background workers that periodically
        # query the cache (stats and storage refresh).  Thread workers cannot
        # be pre-empted, but cancelling them marks them as done so Textual
        # stops scheduling new ones.  The _closed guard on FileCache ensures
        # that any already-running thread that reaches a DB call after this
        # point silently returns instead of raising.
        try:
            self.workers.cancel_group(self, "stats_refresh")
            self.workers.cancel_group(self, "storage_refresh")
        except Exception:
            pass  # App already torn down — ignore
        self.cache.close()
        self._emb_cache.close()


# ── Formatting helpers ─────────────────────────────────────────────────────────

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _status_glyph(running: bool, frame: int) -> str:
    """Animated spinner while running, static hourglass when idle.

    *frame* should be incremented on every refresh so the spinner advances
    once per tick regardless of wall-clock speed.
    """
    if running:
        return f"[green]{_SPINNER[frame % len(_SPINNER)]}[/green]"
    return "[dim]⧗[/dim]"


def _fmt_ago(t: float | None) -> str:
    if t is None:
        return "never"
    d = time.time() - t
    if d < 60:
        return f"{int(d)}s ago"
    if d < 3600:
        return f"{int(d / 60)}m ago"
    return f"{int(d / 3600)}h {int((d % 3600) / 60)}m ago"


def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s / 60)}m {int(s % 60)}s"
    return f"{int(s / 3600)}h {int((s % 3600) / 60)}m"


def _fmt_next_run(ts: float) -> str:
    """Format a future timestamp as a human-readable countdown.

    Examples: "now", "in 42s", "in 9h 33m"
    """
    now = time.time()
    delta = ts - now
    if delta <= 0:
        return "now"
    if delta < 60:
        return f"in {int(delta)}s"

    return f"in {_fmt_duration(delta)}"


def _fmt_rate(rate: float) -> str:
    """Format an items-per-second rate for display.

    Returns an em-dash when the rate is zero (not yet measured).
    """
    if rate <= 0:
        return "—"
    if rate >= 1.0:
        return f"{rate:.1f}/s"
    # sub-1/s: show seconds-per-item instead (more readable for slow workers)
    return f"1 per {1.0 / rate:.0f}s"


def _fmt_eta(remaining: int, rate: float) -> str:
    """Format an ETA string given a remaining count and a rate (items/sec).

    Returns an em-dash when the rate is zero or remaining is zero.
    """
    if rate <= 0 or remaining <= 0:
        return "—"
    return _fmt_duration(remaining / rate)


def _file_size(path: object) -> int:
    """Return the size of *path* in bytes, or 0 if the file does not exist."""
    try:
        from pathlib import Path as _Path  # noqa: PLC0415

        return _Path(str(path)).stat().st_size
    except OSError:
        return 0


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (GiB / MiB / KiB / B)."""
    if n >= 1 << 30:  # noqa: PLR2004
        return f"{n / (1 << 30):.1f} GiB"
    if n >= 1 << 20:  # noqa: PLR2004
        return f"{n / (1 << 20):.1f} MiB"
    if n >= 1 << 10:  # noqa: PLR2004
        return f"{n / (1 << 10):.1f} KiB"
    return f"{n} B"


def _progress_bar(done: int, total: int, width: int = 18) -> str:
    """Return a Rich-markup progress bar string.

    Example (50 % with width=18): ``[████████░░░░░░░░░░] 5/10``
    """
    if total <= 0:
        return "—"
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(frac * 100)
    return f"[{bar}] {done}/{total} ({pct}%)"
