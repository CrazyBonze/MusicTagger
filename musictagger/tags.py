"""Tag definitions registry.

To add a new tag to the pipeline:
  1. Write a check_fn(mutagen.FileType) -> bool
  2. Add a TagDef entry to TAGS

That's it. The cache schema migrates automatically on next startup,
and the inspector loop is driven entirely by this list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TagDef:
    """Describes one tag the pipeline knows how to check and (eventually) fill."""

    # Column suffix: has_{name} in the DB, displayed in the TUI
    name: str

    # Human-readable label shown in the TUI and logs
    description: str

    # Called by the inspector with a mutagen.FileType object.
    # Return True if the tag is already present and non-empty.
    check_fn: Callable[[Any], bool]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _check_keys(f: Any, *keys: str) -> bool:
    """Return True if any of *keys* is present and non-empty in *f*."""
    for key in keys:
        try:
            val = f[key]
            if val:
                return True
        except (KeyError, TypeError):
            pass
    return False


# ── Check functions ────────────────────────────────────────────────────────────


def _check_bpm(f: Any) -> bool:
    """BPM lives in different fields depending on container format:
    - ID3  (mp3, aiff …) : TBPM
    - Vorbis/FLAC/Ogg    : bpm  (case-insensitive via mutagen)
    - MP4/AAC            : tmpo
    """
    return _check_keys(f, "TBPM", "bpm", "BPM", "tmpo")


def _check_mood_happy(f: Any) -> bool:
    # MP4 freeform key uses iTunes namespace
    return _check_keys(
        f,
        "TXXX:MOOD_HAPPY",
        "mood_happy",
        "MOOD_HAPPY",
        "----:com.apple.iTunes:MOOD_HAPPY",
    )


def _check_mood_sad(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:MOOD_SAD",
        "mood_sad",
        "MOOD_SAD",
        "----:com.apple.iTunes:MOOD_SAD",
    )


def _check_mood_relaxed(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:MOOD_RELAXED",
        "mood_relaxed",
        "MOOD_RELAXED",
        "----:com.apple.iTunes:MOOD_RELAXED",
    )


def _check_mood_aggressive(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:MOOD_AGGRESSIVE",
        "mood_aggressive",
        "MOOD_AGGRESSIVE",
        "----:com.apple.iTunes:MOOD_AGGRESSIVE",
    )


def _check_mood_party(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:MOOD_PARTY",
        "mood_party",
        "MOOD_PARTY",
        "----:com.apple.iTunes:MOOD_PARTY",
    )


def _check_mood_dance(f: Any) -> bool:
    """Danceability score (0–100).

    Jaikoz/Picard commonly expose this as "MOOD_DANCEABILITY" for MP4.
    """
    return _check_keys(
        f,
        "TXXX:MOOD_DANCEABILITY",
        "MOOD_DANCEABILITY",
        "mood_danceability",
        "----:com.apple.iTunes:MOOD_DANCEABILITY",
        "TXXX:MOOD_DANCE",
        "mood_dance",
        "MOOD_DANCE",
        "TXXX:MOOD_DANCEABILITY",
        "----:com.apple.iTunes:MOOD_DANCE",
    )


def _check_mood(f: Any) -> bool:
    """Canonical mood tag used by Navidrome and common taggers."""
    return _check_keys(
        f,
        "TMOO",
        "mood",
        "MOOD",
        "----:com.apple.iTunes:MOOD",
    )


def _check_electronic(f: Any) -> bool:
    # Some apps store as MOOD_ELECTRONIC
    return _check_keys(
        f,
        "TXXX:ELECTRONIC",
        "electronic",
        "ELECTRONIC",
        "TXXX:MOOD_ELECTRONIC",
        "MOOD_ELECTRONIC",
        "mood_electronic",
        "----:com.apple.iTunes:MOOD_ELECTRONIC",
        "----:com.apple.iTunes:ELECTRONIC",
    )


def _check_acoustic(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:ACOUSTIC",
        "acoustic",
        "ACOUSTIC",
        "TXXX:MOOD_ACOUSTIC",
        "MOOD_ACOUSTIC",
        "mood_acoustic",
        "----:com.apple.iTunes:MOOD_ACOUSTIC",
        "----:com.apple.iTunes:ACOUSTIC",
    )


def _check_instrumental(f: Any) -> bool:
    return _check_keys(
        f,
        "TXXX:INSTRUMENTAL",
        "instrumental",
        "INSTRUMENTAL",
        "TXXX:MOOD_INSTRUMENTAL",
        "MOOD_INSTRUMENTAL",
        "mood_instrumental",
        "----:com.apple.iTunes:MOOD_INSTRUMENTAL",
        "----:com.apple.iTunes:INSTRUMENTAL",
    )


def _check_timbre_brightness(f: Any) -> bool:
    """Timbre brightness 0–100 (0 = dark, 100 = bright).

    Prefer explicit TIMBRE_BRIGHTNESS but accept legacy TIMBRE/timbre keys.
    """
    return _check_keys(
        f,
        # Preferred explicit key
        "TXXX:TIMBRE_BRIGHTNESS",
        "timbre_brightness",
        "TIMBRE_BRIGHTNESS",
        "----:com.apple.iTunes:TIMBRE_BRIGHTNESS",
        # Legacy variants kept for compatibility
        "TXXX:TIMBRE",
        "timbre",
        "TIMBRE",
        "----:com.apple.iTunes:TIMBRE",
    )


def _check_tonality(f: Any) -> bool:
    """Tonal/atonal score encoded as 0–100 (0 = atonal, 100 = tonal)."""
    return _check_keys(
        f,
        "TXXX:TONALITY",
        "tonality",
        "TONALITY",
        "----:com.apple.iTunes:TONALITY",
    )


def _check_key(f: Any) -> bool:
    """Musical key, e.g. "C major" or "A minor".

    Field names used by common taggers and players:
    - ID3 (mp3, aiff …) : TKEY
    - Vorbis/FLAC/Ogg   : KEY  (Jaikoz canonical; INITIALKEY accepted as alias)
    - MP4/AAC            : ----:com.apple.iTunes:initialkey
    - WMA                : WM/InitialKey
    """
    return _check_keys(
        f,
        "TKEY",
        "key",
        "KEY",
        "initialkey",
        "INITIALKEY",
        "----:com.apple.iTunes:initialkey",
        "WM/InitialKey",
    )


# ── Registry ───────────────────────────────────────────────────────────────────
#
# Add TagDef entries here to extend the system.
# Order doesn't matter — the inspector checks all of them.

TAGS: list[TagDef] = [
    TagDef(
        name="bpm",
        description="Tempo (BPM)",
        check_fn=_check_bpm,
    ),
    TagDef(
        name="mood_happy",
        description="Mood: Happy",
        check_fn=_check_mood_happy,
    ),
    TagDef(
        name="mood_sad",
        description="Mood: Sad",
        check_fn=_check_mood_sad,
    ),
    TagDef(
        name="mood_relaxed",
        description="Mood: Relaxed",
        check_fn=_check_mood_relaxed,
    ),
    TagDef(
        name="mood_aggressive",
        description="Mood: Aggressive",
        check_fn=_check_mood_aggressive,
    ),
    TagDef(
        name="mood_party",
        description="Mood: Party",
        check_fn=_check_mood_party,
    ),
    TagDef(
        name="mood_dance",
        description="Mood: Dance",
        check_fn=_check_mood_dance,
    ),
    TagDef(
        name="mood",
        description="Mood",
        check_fn=_check_mood,
    ),
    TagDef(
        name="electronic",
        description="Electronic",
        check_fn=_check_electronic,
    ),
    TagDef(
        name="acoustic",
        description="Acoustic",
        check_fn=_check_acoustic,
    ),
    TagDef(
        name="instrumental",
        description="Instrumental",
        check_fn=_check_instrumental,
    ),
    TagDef(
        name="timbre_brightness",
        description="Timbre Brightness",
        check_fn=_check_timbre_brightness,
    ),
    TagDef(
        name="tonality",
        description="Tonality",
        check_fn=_check_tonality,
    ),
    TagDef(
        name="key",
        description="Musical Key",
        check_fn=_check_key,
    ),
]
