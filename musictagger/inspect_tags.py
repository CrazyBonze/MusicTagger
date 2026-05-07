"""Report musictagger-managed tag values from audio files as CSV.

Usage:
    uv run musictagger-inspect-tags /path/to/directory > report.csv
    uv run musictagger-inspect-tags /path/to/file.mp3

Scans recursively and writes one CSV row per audio file.  Columns are the
file path followed by one column per managed tag.  Missing tags are left
blank.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import mutagen
from mutagen.aiff import AIFF
from mutagen.apev2 import APEv2File
from mutagen.id3 import ID3FileType
from mutagen.mp4 import MP4, MP4FreeForm


_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".dsf",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".tta",
    ".wav",
    ".wv",
}

# Ordered list of (column_header, reader_function) pairs.
# Readers return a string value or "" if the tag is absent.
_COLUMNS: list[tuple[str, Any]] = []


def _register(header: str):
    """Decorator that registers a tag reader under *header*."""

    def _dec(fn):
        _COLUMNS.append((header, fn))
        return fn

    return _dec


# ── Value readers ──────────────────────────────────────────────────────────────


def _first_str(values: Any) -> str:
    """Return the first element of *values* as a stripped string, or ''."""
    if not values:
        return ""
    v = values[0]
    if isinstance(v, MP4FreeForm):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def _read_id3_txxx(f: Any, desc: str) -> str:
    """Read a TXXX frame by description from an ID3 file."""
    key = f"TXXX:{desc}"
    frame = f.tags.get(key)
    if frame is None:
        return ""
    return _first_str(frame.text)


def _read_vorbis(f: Any, *keys: str) -> str:
    """Read the first matching key from a Vorbis/FLAC file."""
    for key in keys:
        val = f.get(key)
        if val:
            return _first_str(val)
    return ""


def _read_mp4(f: Any, atom: str) -> str:
    val = f.get(atom)
    if not val:
        return ""
    return _first_str(val)


def _read_apev2(f: Any, *keys: str) -> str:
    if f.tags is None:
        return ""
    for key in keys:
        val = f.tags.get(key)
        if val:
            return str(val).strip()
    return ""


def _read_score_tag(f: Any, id3_desc: str, vorbis_key: str, mp4_atom: str) -> str:
    """Read a 0–100 integer score tag across all supported formats."""
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        return _read_id3_txxx(f, id3_desc)
    if isinstance(f, MP4):
        return _read_mp4(f, mp4_atom)
    if isinstance(f, APEv2File):
        return _read_apev2(f, id3_desc, vorbis_key.upper())
    # Vorbis / FLAC
    return _read_vorbis(f, vorbis_key.upper(), vorbis_key)


# ── Column definitions ─────────────────────────────────────────────────────────


@_register("bpm")
def _read_bpm(f: Any) -> str:
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        frame = f.tags.get("TBPM")
        return _first_str(frame.text) if frame else ""
    if isinstance(f, MP4):
        val = f.get("tmpo")
        return str(val[0]) if val else ""
    if isinstance(f, APEv2File):
        return _read_apev2(f, "BPM", "TBPM")
    return _read_vorbis(f, "BPM", "bpm", "TBPM")


@_register("mood_happy")
def _read_mood_happy(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_HAPPY", "mood_happy", "----:com.apple.iTunes:MOOD_HAPPY"
    )


@_register("mood_sad")
def _read_mood_sad(f: Any) -> str:
    return _read_score_tag(f, "MOOD_SAD", "mood_sad", "----:com.apple.iTunes:MOOD_SAD")


@_register("mood_relaxed")
def _read_mood_relaxed(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_RELAXED", "mood_relaxed", "----:com.apple.iTunes:MOOD_RELAXED"
    )


@_register("mood_aggressive")
def _read_mood_aggressive(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_AGGRESSIVE", "mood_aggressive", "----:com.apple.iTunes:MOOD_AGGRESSIVE"
    )


@_register("mood_party")
def _read_mood_party(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_PARTY", "mood_party", "----:com.apple.iTunes:MOOD_PARTY"
    )


@_register("mood_dance")
def _read_mood_dance(f: Any) -> str:
    # Written as MOOD_DANCEABILITY by convention
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        for desc in ("MOOD_DANCEABILITY", "MOOD_DANCE"):
            val = _read_id3_txxx(f, desc)
            if val:
                return val
        return ""
    if isinstance(f, MP4):
        for atom in (
            "----:com.apple.iTunes:MOOD_DANCEABILITY",
            "----:com.apple.iTunes:MOOD_DANCE",
        ):
            val = _read_mp4(f, atom)
            if val:
                return val
        return ""
    if isinstance(f, APEv2File):
        return _read_apev2(f, "MOOD_DANCEABILITY", "MOOD_DANCE")
    return _read_vorbis(
        f, "MOOD_DANCEABILITY", "mood_danceability", "MOOD_DANCE", "mood_dance"
    )


@_register("mood")
def _read_mood(f: Any) -> str:
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        frame = f.tags.get("TMOO")
        return _first_str(frame.text) if frame else ""
    if isinstance(f, MP4):
        return _read_mp4(f, "----:com.apple.iTunes:MOOD")
    if isinstance(f, APEv2File):
        return _read_apev2(f, "MOOD")
    return _read_vorbis(f, "MOOD", "mood")


@_register("theme")
def _read_theme(f: Any) -> str:
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        return _read_id3_txxx(f, "THEME")
    if isinstance(f, MP4):
        return _read_mp4(f, "----:com.apple.iTunes:THEME")
    if isinstance(f, APEv2File):
        return _read_apev2(f, "THEME")
    return _read_vorbis(f, "THEME", "theme")


@_register("electronic")
def _read_electronic(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_ELECTRONIC", "mood_electronic", "----:com.apple.iTunes:MOOD_ELECTRONIC"
    )


@_register("acoustic")
def _read_acoustic(f: Any) -> str:
    return _read_score_tag(
        f, "MOOD_ACOUSTIC", "mood_acoustic", "----:com.apple.iTunes:MOOD_ACOUSTIC"
    )


@_register("instrumental")
def _read_instrumental(f: Any) -> str:
    return _read_score_tag(
        f,
        "MOOD_INSTRUMENTAL",
        "mood_instrumental",
        "----:com.apple.iTunes:MOOD_INSTRUMENTAL",
    )


@_register("timbre_brightness")
def _read_timbre_brightness(f: Any) -> str:
    return _read_score_tag(
        f,
        "TIMBRE_BRIGHTNESS",
        "timbre_brightness",
        "----:com.apple.iTunes:TIMBRE_BRIGHTNESS",
    )


@_register("tonality")
def _read_tonality(f: Any) -> str:
    return _read_score_tag(f, "TONALITY", "tonality", "----:com.apple.iTunes:TONALITY")


@_register("key")
def _read_key(f: Any) -> str:
    """Musical key — TKEY (ID3/AIFF), INITIALKEY (Vorbis/FLAC), iTunes initialkey (MP4)."""
    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            return ""
        frame = f.tags.get("TKEY")
        return _first_str(frame.text) if frame else ""
    if isinstance(f, MP4):
        return _read_mp4(f, "----:com.apple.iTunes:initialkey")
    if isinstance(f, APEv2File):
        return _read_apev2(f, "INITIALKEY", "KEY")
    # Vorbis / FLAC
    return _read_vorbis(f, "INITIALKEY", "initialkey", "KEY", "key")


# ── File iteration ─────────────────────────────────────────────────────────────


def _iter_targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")
    return sorted(
        child
        for child in path.rglob("*")
        if child.is_file() and child.suffix.lower() in _AUDIO_EXTENSIONS
    )


def _read_row(path: Path) -> dict[str, str]:
    """Return a dict of column → value for *path*."""
    try:
        f = mutagen.File(str(path), easy=False)
    except Exception:
        f = None

    row: dict[str, str] = {"path": str(path)}

    if f is None:
        for header, _ in _COLUMNS:
            row[header] = ""
        return row

    for header, reader in _COLUMNS:
        try:
            row[header] = reader(f)
        except Exception:
            row[header] = ""

    return row


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the `musictagger-inspect-tags` console script."""
    parser = argparse.ArgumentParser(
        prog="musictagger-inspect-tags",
        description=(
            "Report musictagger-managed tag values from audio files as CSV. "
            "Scans directories recursively."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Audio file or directory to inspect",
    )
    args = parser.parse_args()

    files = _iter_targets(args.path.expanduser())

    if not files:
        print(f"No supported audio files found in: {args.path}", file=sys.stderr)
        return

    headers = ["path"] + [h for h, _ in _COLUMNS]
    writer = csv.DictWriter(sys.stdout, fieldnames=headers, lineterminator="\n")
    writer.writeheader()

    for path in files:
        writer.writerow(_read_row(path))


if __name__ == "__main__":
    main()
