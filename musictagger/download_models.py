"""Download Essentia model files used by musictagger.

Running this script fetches all model files required for Essentia inference
and saves them to the configured models directory.

Usage:
    uv run musictagger-download-models [--models-dir PATH]
    python -m musictagger.download_models [--models-dir PATH]

The default destination is ~/.local/share/musictagger/models/.

All models are downloaded from https://essentia.upf.edu/models/ and are
licensed under CC BY-NC-SA 4.0 by the Music Technology Group (MTG), UPF.

Two-stage pipeline:
  Every Essentia classifier (mood, timbre, etc.) works in two steps:
    1. Audio → embeddings  (the shared feature extractor)
    2. Embeddings → class probabilities  (the lightweight classifier head)

  We use the Discogs-EffNet backbone for all classifier heads.  The mood text
  tag uses the Genre Discogs400 classifier (400 music styles, 3.3M tracks).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# ── Model catalogue ────────────────────────────────────────────────────────────
#
# Each entry is (filename, URL).  The extractor must be listed first so that
# the download order is logical when displayed to the user.

_BASE = "https://essentia.upf.edu/models"

MODELS: list[tuple[str, str]] = [
    # ── Shared feature extractor ───────────────────────────────────────────────
    # All mood/timbre/tonality classifiers take Discogs-EffNet embeddings as input.
    (
        "discogs-effnet-bs64-1.pb",
        f"{_BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    ),
    # ── Tempo (TempoCNN) ──────────────────────────────────────────────────────
    # Used as a secondary BPM backend for octave resolution when DeepRhythm
    # returns a low-confidence result.  Requires 11025 Hz audio (handled
    # internally by EssentiaEngine.resolve_bpm_octave).
    (
        "deepsquare-k16-3.pb",
        f"{_BASE}/tempo/tempocnn/deepsquare-k16-3.pb",
    ),
    # ── Classifier heads ──────────────────────────────────────────────────────
    (
        "mood_happy-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    ),
    (
        "mood_sad-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    ),
    (
        "mood_relaxed-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    ),
    (
        "mood_aggressive-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    ),
    (
        "mood_party-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb",
    ),
    (
        "danceability-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
    ),
    (
        "mood_electronic-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.pb",
    ),
    (
        "mood_acoustic-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb",
    ),
    (
        "voice_instrumental-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb",
    ),
    (
        "timbre-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/timbre/timbre-discogs-effnet-1.pb",
    ),
    (
        "tonal_atonal-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.pb",
    ),
    # Genre Discogs400 — used to derive the mood text tag.
    # 400-class multi-label classifier trained on 3.3M Discogs tracks.
    # Top-N subgenre labels by Sigmoid score are written as the mood value.
    (
        "genre_discogs400-discogs-effnet-1.pb",
        f"{_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    ),
    (
        "genre_discogs400-discogs-effnet-1.json",
        f"{_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    ),
]


def missing_models(dest: Path) -> list[str]:
    """Return the model filenames that are not present in *dest*."""
    return [filename for filename, _ in MODELS if not (dest / filename).exists()]


def download_models(dest: Path, force: bool = False) -> None:
    """Download all model files to *dest*, skipping files that already exist.

    Set *force* to re-download even if a file is present.
    """
    dest.mkdir(parents=True, exist_ok=True)

    total = len(MODELS)
    for i, (filename, url) in enumerate(MODELS, 1):
        target = dest / filename
        prefix = f"[{i}/{total}]"

        if target.exists() and not force:
            print(f"{prefix} Already present: {filename}")
            continue

        print(f"{prefix} Downloading {filename} … ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, target)
            size_mb = target.stat().st_size / 1_048_576
            print(f"done ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            # Remove a partial file if the download failed mid-way.
            if target.exists():
                target.unlink()

    print(f"\nModels directory: {dest}")


def main() -> None:
    """Entry point for the `musictagger-download-models` console script."""
    default_dir = Path.home() / ".local/share/musictagger/models"

    parser = argparse.ArgumentParser(
        description="Download Essentia model weights for musictagger"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=default_dir,
        metavar="PATH",
        help=f"Destination directory (default: {default_dir})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist",
    )
    args = parser.parse_args()

    download_models(args.models_dir.expanduser(), force=args.force)


if __name__ == "__main__":
    main()
