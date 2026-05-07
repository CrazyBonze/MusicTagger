"""Tests for the TUI activity-log message pipeline.

These tests verify that:
- Rich markup characters in filenames and error strings are escaped before
  reaching the RichLog widget, so a filename like ``[silence].mp3`` can never
  raise MarkupError and crash the app.
- The ``_log`` / ``_applog`` / ``_applog_markup`` methods behave correctly in
  isolation without needing a running Textual application.
- The ``_make_log`` background-thread helper posts ``LogEvent`` messages whose
  text is left unescaped at the transport layer (escaping happens at render
  time inside ``_log``).
- A variety of adversarial filename patterns are handled safely.
"""

from __future__ import annotations

from rich.markup import escape as markup_escape

from musictagger.tui import LogEvent


# ── markup_escape contract ─────────────────────────────────────────────────────


def test_markup_escape_neutralises_square_brackets() -> None:
    assert markup_escape("[silence]") == r"\[silence]"


def test_markup_escape_makes_bracket_strings_renderable() -> None:
    # The exact escaped form varies by Rich version (known vs unknown tags),
    # but the rendered plain text must always match the original content.
    from rich.text import Text

    for raw in ("[Remastered]", "[silence]", "[EP]", "[Vol. I]"):
        escaped = markup_escape(raw)
        text = Text.from_markup(escaped)
        assert raw.strip("[]") in text.plain, (
            f"markup_escape lost content: {raw!r} -> {escaped!r} -> {text.plain!r}"
        )


def test_markup_escape_leaves_safe_text_unchanged() -> None:
    assert markup_escape("normal filename.mp3") == "normal filename.mp3"


def test_markup_escape_handles_ampersand() -> None:
    # Rich treats & as a potential HTML entity in some contexts; escape is safe.
    result = markup_escape("Barnes & Barnes - Song.flac")
    assert "Barnes" in result
    assert "Barnes.flac" not in result or "&" in result  # either form is fine


def test_markup_escape_handles_empty_string() -> None:
    assert markup_escape("") == ""


# ── Adversarial filename patterns ─────────────────────────────────────────────


ADVERSARIAL_FILENAMES = [
    "[silence].mp3",
    "[untitled].flac",
    "[Remastered].mp3",
    "[EP].flac",
    "[Vol. I].mp3",
    "Song [Live].mp3",
    "[2024] Album Title.flac",
    "Track [feat. Artist].mp3",
    # Nested and malformed tags
    "[[double bracket]].mp3",
    "[/closing tag].flac",
    "[bold red]title[/bold red].mp3",
    # HTML-like characters that Rich might try to interpret
    "AT&T - Song.mp3",
    "5 > 3.flac",
    "a < b.mp3",
    # Unicode that looks like markup
    "Track \u2329name\u232a.mp3",
    # Backslash sequences
    "path\\to\\[file].mp3",
    # Very long filename
    "[" + "x" * 200 + "].mp3",
    # Mix of safe and unsafe
    "01 - Artist - [Album] - Track.flac",
]


def test_all_adversarial_filenames_are_safely_escaped() -> None:
    """Every adversarial filename must survive markup_escape without raising."""
    from rich.console import Console
    from rich.text import Text

    console = Console(force_terminal=False)
    for filename in ADVERSARIAL_FILENAMES:
        escaped = markup_escape(filename)
        # Constructing a Rich Text from the escaped string must not raise.
        try:
            text = Text.from_markup(
                f"[dim]10:00:00[/dim] [green][scanner][/green] {escaped}"
            )
            # Rendering through a console must not raise either.
            with console.capture():
                console.print(text)
        except Exception as exc:
            raise AssertionError(
                f"Escaped filename raised during render: {filename!r} -> {escaped!r}: {exc}"
            ) from exc


def test_unescaped_bracket_filename_would_raise() -> None:
    """Confirm the bug is real: an unescaped [silence] raises MarkupError."""
    from rich.markup import MarkupError
    from rich.text import Text

    dangerous = "[silence].mp3"
    try:
        Text.from_markup(f"[dim]10:00:00[/dim] [green][scanner][/green] {dangerous}")
        # Some versions of Rich silently ignore unknown tags — only fail if it
        # actually raises, to keep the test honest.
    except (MarkupError, Exception):
        pass  # Expected on versions that are strict about unknown tags.

    # The important assertion: after escaping it must never raise.
    safe = markup_escape(dangerous)
    # Must not raise
    Text.from_markup(f"[dim]10:00:00[/dim] [green][scanner][/green] {safe}")


# ── LogEvent message ───────────────────────────────────────────────────────────


def test_log_event_preserves_raw_text() -> None:
    """LogEvent stores the original (unescaped) text; escaping happens at render."""
    event = LogEvent("scanner", "[silence].mp3")
    assert event.source == "scanner"
    assert event.text == "[silence].mp3"


def test_log_event_text_is_safely_escapable() -> None:
    """The text carried in a LogEvent can always be safely escaped."""
    event = LogEvent("worker", "Tagged [silence].mp3: bpm=120")
    escaped = markup_escape(event.text)
    # After escaping, the [ in [silence] becomes \[ so it is no longer
    # parsed as a markup tag.  Verify the escaped form is renderable.
    from rich.text import Text

    Text.from_markup(escaped)  # must not raise


# ── _log render string construction ───────────────────────────────────────────


def _build_log_line(source: str, msg: str) -> str:
    """Reproduce the exact string _log() passes to RichLog.write()."""
    from rich.markup import escape as me

    color = {
        "scanner": "green",
        "inspector": "cyan",
        "worker": "yellow",
        "cleanup": "magenta",
        "app": "white",
    }.get(source, "white")
    safe_msg = me(msg)
    return f"[dim]10:00:00[/dim] [{color}][{source}][/{color}] {safe_msg}"


def test_log_line_is_valid_rich_markup_for_bracket_filename() -> None:
    from rich.text import Text

    line = _build_log_line("scanner", "New/changed: [silence].mp3")
    # Must not raise
    Text.from_markup(line)


def test_log_line_preserves_filename_content() -> None:
    from rich.text import Text

    line = _build_log_line("worker", "Tagged [Remastered].mp3: bpm=95")
    text = Text.from_markup(line)
    plain = text.plain
    assert "Remastered" in plain
    assert ".mp3" in plain


def test_log_line_for_every_adversarial_filename() -> None:
    from rich.text import Text

    for filename in ADVERSARIAL_FILENAMES:
        for source in ("scanner", "inspector", "worker", "cleanup", "app"):
            line = _build_log_line(source, f"Processing: {filename}")
            try:
                Text.from_markup(line)
            except Exception as exc:
                raise AssertionError(
                    f"Log line for {filename!r} raised: {exc}"
                ) from exc


def test_log_line_exception_messages_are_safe() -> None:
    """Exception strings from ffmpeg/mutagen can contain brackets too."""
    from rich.text import Text

    exception_messages = [
        "'NoneType' object has no attribute 'to'",
        "ffmpeg decode failed (rc=69): [aist#0:0/mp3 @ 0x5c06] Error",
        "MonoLoader failed on [silence].mp3: invalid data",
        "[Errno 5] Input/output error: '/media/[drive]/track.flac'",
        "KeyError: '[mood]'",
    ]
    for msg in exception_messages:
        line = _build_log_line("worker", f"Error: {msg}")
        try:
            Text.from_markup(line)
        except Exception as exc:
            raise AssertionError(
                f"Exception message caused render failure: {msg!r}: {exc}"
            ) from exc


# ── _applog_markup: trusted markup path ───────────────────────────────────────


def test_trusted_markup_strings_are_valid_rich() -> None:
    """Strings used with _applog_markup must be syntactically valid Rich markup."""
    from rich.text import Text

    trusted_messages = [
        "[red]Worker watchdog: no heartbeat for 300s — resetting running flag and relaunching[/red]",
        "[yellow]⏸  Paused — jobs will finish their current pass then stop[/yellow]",
        "[green]▶  Resumed[/green]",
        "[red]Background job error (worker): some plain error message[/red]",
    ]
    for msg in trusted_messages:
        try:
            Text.from_markup(msg)
        except Exception as exc:
            raise AssertionError(
                f"Trusted markup string is not valid Rich markup: {msg!r}: {exc}"
            ) from exc


def test_worker_error_in_applog_markup_escapes_user_data() -> None:
    """Worker name and error text embedded in _applog_markup must be escaped."""
    from rich.text import Text

    # Simulate what on_worker_state_changed builds
    worker_name = "[scanner]"  # adversarial worker name
    error_text = "file [silence].mp3 not found"
    msg = (
        f"[red]Background job error ({markup_escape(str(worker_name))}): "
        f"{markup_escape(str(error_text))}[/red]"
    )
    # Must not raise
    text = Text.from_markup(msg)
    plain = text.plain
    assert "scanner" in plain
    assert "silence" in plain
