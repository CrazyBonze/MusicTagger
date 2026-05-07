"""Compute worker — DeepRhythm BPM inference + Essentia tag inference.

Processes files that the inspector flagged as needing tag work.  Three
inference backends are used:

  DeepRhythm (PyTorch CNN)
    Primary BPM backend.  Loaded lazily on first BPM request.  Returns a
    softmax confidence score alongside the BPM estimate.

  TempoCNN (Essentia TensorFlow — ``deepsquare-k16-3.pb``)
    Secondary BPM backend, invoked when DeepRhythm's softmax confidence falls
    below ``config.bpm_confidence_threshold`` (default 0.10).  Two Essentia
    algorithms are used together:

      TempoCNN
        Produces a majority-vote ``globalTempo`` scalar.

      TensorflowPredictTempoCNN
        Produces the raw ``(n_patches, 256)`` softmax distribution over the
        30–286 BPM axis (as documented in the DeepSquare model JSON).

    Both the DeepRhythm BPM and TempoCNN's ``globalTempo`` are scored against
    the averaged probability distribution; the candidate with the higher
    probability mass wins.  This corrects systematic half/double-tempo errors
    that DeepRhythm exhibits on genres underrepresented in its training set
    (e.g. metal, ambient).  ``deepsquare-k16-3.pb`` must be present in
    ``config.models_dir`` (downloaded by ``musictagger-download-models``).

  Essentia TensorFlow pipelines
    Two model families are used:

      Discogs-EffNet classifier heads
        Handles mood, danceability, acoustic/electronic/instrumental
        character, timbre brightness, and tonality.  The pipeline has two
        stages:
          1. Audio → Discogs-EffNet embeddings  (shared, loaded once)
          2. Embeddings → class probabilities   (one lightweight model per tag)
        All classifier models take the same EffNet embeddings, so the
        extractor runs only once per file regardless of how many of those tags
        are missing.

        The ``mood`` text tag is derived from the Genre Discogs400 classifier
        (``genre_discogs400-discogs-effnet-1.pb``), which predicts 400 music
        style labels trained on 3.3 M Discogs tracks (ROC-AUC 0.95).  The
        top-``_GENRE_MOOD_TOP_N`` subgenre labels by Sigmoid score are written
        as plain label strings (confidence scores are used for selection only
        and are not stored in the tag), giving a reliable genre-as-mood signal
        without the over-firing artefacts of the previous MTG-Jamendo moodtheme
        model.

Decoder note:
  deeprhythm uses librosa.load() internally when given a file path.  For
  .m4a / AAC files, libsndfile often fails and librosa falls back to audioread
  (deprecated), producing noisy warnings.  We sidestep this entirely by
  decoding to raw PCM ourselves via ffmpeg and calling
  DeepRhythmPredictor.predict_from_audio(), which accepts a numpy array
  directly.

  Essentia's MonoLoader handles its own decoding internally and works well
  across formats, so we don't need the ffmpeg workaround for Essentia.

CUDA / CPU selection:
  The DeepRhythm predictor is initialised lazily on first use.  If torch
  reports CUDA is available we try it first; if that fails we fall back to CPU.
  Either way the same predictor instance is reused for the lifetime of the
  process.

  Essentia/TensorFlow uses its own device selection.  No explicit CUDA
  configuration is needed — TF picks up available GPUs automatically.

Model files:
  Essentia model files must be present in config.models_dir before the worker
  can process Essentia tags. Run ``musictagger-download-models`` to fetch them.
  Some classifiers also require metadata JSON files for label names. Files that
  are absent are skipped gracefully with a warning; the tags they cover remain
  in ``needs_work`` and will be retried on the next pass once the files are
  present.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import warnings
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
import concurrent.futures.thread as _cft
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mutagen
import numpy as np
from loguru import logger
from rich.markup import escape as markup_escape

from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.embeddings import EmbeddingCache
from musictagger.mood_mappings import label_for_class
from musictagger.tags import TAGS

# Essentia pulls in TensorFlow on first import. Silence TensorFlow's C++ info and
# warning logs before that happens so they do not corrupt the Textual UI.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress a harmless PyTorch internal warning emitted by the DeepRhythm model
# on every forward pass when run on CPU with certain kernel configurations.
warnings.filterwarnings(
    "ignore",
    message=r"Using padding='same' with even kernel lengths and odd dilation.*",
    category=UserWarning,
)

# ── Essentia model catalogue ───────────────────────────────────────────────────
#
# Maps TagDef.name → (classifier_filename, positive_class_index).
#
# All classifiers use the shared Discogs-EffNet extractor.  Each model is a
# 2-class softmax; the class ordering varies per model — check the official
# Essentia JSON metadata to confirm which index is the positive class.
#
# Per official Essentia JSON metadata (classes list = [index0, index1]):
#
#   mood_happy:        [happy, non_happy]          → index 0
#   mood_sad:          [non_sad, sad]               → index 1
#   mood_relaxed:      [non_relaxed, relaxed]       → index 1
#   mood_aggressive:   [aggressive, not_aggressive] → index 0
#   mood_party:        [non_party, party]           → index 1
#   mood_dance:        [danceable, not_danceable]   → index 0
#   electronic:        [electronic, non_electronic] → index 0
#   acoustic:          [acoustic, non_acoustic]     → index 0
#   instrumental:      [instrumental, voice]        → index 0
#   timbre_brightness: [bright, dark]               → index 0
#   tonality:          [atonal, tonal]              → index 1

_ESSENTIA_TAG_MODELS: dict[str, tuple[str, int]] = {
    # tag_name: (classifier_pb_filename, positive_class_index)
    "mood_happy": ("mood_happy-discogs-effnet-1.pb", 0),
    "mood_sad": ("mood_sad-discogs-effnet-1.pb", 1),
    "mood_relaxed": ("mood_relaxed-discogs-effnet-1.pb", 1),
    "mood_aggressive": ("mood_aggressive-discogs-effnet-1.pb", 0),
    "mood_party": ("mood_party-discogs-effnet-1.pb", 1),
    "mood_dance": ("danceability-discogs-effnet-1.pb", 0),
    "electronic": ("mood_electronic-discogs-effnet-1.pb", 0),
    "acoustic": ("mood_acoustic-discogs-effnet-1.pb", 0),
    "instrumental": ("voice_instrumental-discogs-effnet-1.pb", 0),
    "timbre_brightness": ("timbre-discogs-effnet-1.pb", 0),
    "tonality": ("tonal_atonal-discogs-effnet-1.pb", 1),
}

_EFFNET_EXTRACTOR_FILENAME = "discogs-effnet-bs64-1.pb"
_GENRE_DISCOGS400_MODEL_FILENAME = "genre_discogs400-discogs-effnet-1.pb"
_GENRE_DISCOGS400_METADATA_FILENAME = "genre_discogs400-discogs-effnet-1.json"

# Sample rate expected by Essentia's MonoLoader for all EffNet-based models.
_ESSENTIA_SR = 16000
_ESSENTIA_LOGGING_CONFIGURED = False


# TempoCNN — secondary BPM backend, separate model and sample rate from the
# EffNet pipeline.  Invoked only when DeepRhythm confidence is below threshold.
_TEMPOCNN_MODEL_FILENAME = "deepsquare-k16-3.pb"
_TEMPOCNN_SR = 11025
# BPM axis for the DeepSquare model: 256 classes linearly spaced 30–286 BPM.
_TEMPOCNN_BPM_MIN: float = 30.0
_TEMPOCNN_BPM_MAX: float = 286.0
_TEMPOCNN_N_CLASSES: int = 256

# KeyExtractor — classical HPCP-based musical key detection, built into Essentia.
# Requires audio at the standard CD sample rate; no model file needed.
_KEY_EXTRACTOR_SR = 44100


def _configure_essentia_runtime() -> None:
    """Suppress Essentia/TensorFlow log noise before importing algorithms.

    Essentia's C++ logger emits recurring warnings like "No network created..."
    during model lifecycle events, and TensorFlow prints startup diagnostics to
    stderr. Both pollute the TUI without indicating a user-actionable failure.
    Keep Essentia errors enabled so genuine failures still surface.
    """
    global _ESSENTIA_LOGGING_CONFIGURED

    if _ESSENTIA_LOGGING_CONFIGURED:
        return

    import essentia  # type: ignore[import]

    essentia.log.infoActive = False
    essentia.log.warningActive = False
    _ESSENTIA_LOGGING_CONFIGURED = True


def _top_subgenres_from_genre_scores(
    classes: list[str],
    averaged: np.ndarray,
    threshold: float,
    min_results: int,
    max_results: int,
) -> list[str]:
    """Return display labels from a Genre Discogs400 prediction.

    Each class label has the form ``"Genre---Subgenre"`` (e.g.
    ``"Electronic---Techno"``).  Display names are resolved via
    ``label_for_class``: most classes strip the genre prefix and title-case
    the subgenre; the 32 class keys involved in subgenre collisions (where the
    same subgenre name appears under two parent genres) resolve to explicit
    human-readable names defined in ``mood_mappings.GENRE_LABEL_OVERRIDES``.

    Confidence scores are used for selection only and are not included in the
    returned strings.

    Selection rules (applied in order):
      1. Include every label whose score is at or above *threshold*, up to
         *max_results* entries.
      2. If the threshold pass-list has fewer than *min_results* entries, pad
         it with the next-highest-scoring labels (regardless of threshold) until
         *min_results* is reached.
      3. A display label that has already been selected is skipped (guards
         against duplicate output if two class keys resolve to the same name).

    Example output: ``["Boom Bap", "Techno", "Indie Rock"]``
    """
    ranked = sorted(
        zip(classes, averaged.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    seen: set[str] = set()
    results: list[str] = []
    for label, score in ranked:
        above_threshold = score >= threshold
        under_max = len(results) < max_results
        below_min = len(results) < min_results

        if not above_threshold and not below_min:
            # Threshold not met and minimum already satisfied — stop.
            break
        if not under_max:
            break

        display = label_for_class(label)
        if display in seen:
            continue
        seen.add(display)
        results.append(display)

    return results


# ── Tag writers ───────────────────────────────────────────────────────────────
#
# One function per tag name that knows how to write that value into a mutagen
# file object.  Keyed by TagDef.name so process_file() can dispatch cleanly.
#
# BPM:
#   ID3 (mp3/aiff): TBPM frame
#   Vorbis/FLAC/Ogg: lowercase "bpm" key
#   MP4/AAC: "tmpo" atom (integer-only)
#
# Mood/score tags (0–100 integers):
#   ID3: TXXX frame with description = upper-case tag name
#   Vorbis/FLAC/Ogg: lower-case key
#   MP4: freeform atom ----:com.apple.iTunes:<TAG_NAME>
#   APEv2 (WavPack): upper-case key


def _write_bpm(f: mutagen.FileType, bpm: int) -> None:
    """Write integer BPM into *f* using the appropriate tag format.

    mutagen.File() returns container-specific types (MP3, FLAC, MP4, …),
    not the bare tag-class types.  We detect the right format by checking
    the container type, not the tag type:

    - ID3FileType subclasses (MP3, TrueAudio, DSF, …): TBPM frame
    - AIFF: also uses ID3 tags, stored in f.tags which is an ID3 instance
    - MP4: tmpo atom (integer list)
    - Everything else (FLAC, OGG, Opus, WavPack, …): Vorbis comment "bpm"
    """
    from mutagen.aiff import AIFF
    from mutagen.id3 import ID3FileType, TBPM
    from mutagen.mp4 import MP4

    if isinstance(f, (ID3FileType, AIFF)):
        # Both MP3 (ID3FileType) and AIFF expose their tags as an ID3 object.
        if f.tags is None:
            f.add_tags()
        f.tags["TBPM"] = TBPM(encoding=3, text=[str(bpm)])
    elif isinstance(f, MP4):
        f["tmpo"] = [bpm]
    else:
        # Vorbis comments are case-insensitive, so alias writes collapse to one
        # logical field. Use the exact display key Jaikoz expects.
        f["BPM"] = [str(bpm)]


def _make_score_writer(tag_name: str) -> Callable[[mutagen.FileType, int], None]:
    """Return a writer function for a 0–100 integer score tag.

    The tag name is used as:
      - ID3:  TXXX description (upper-case)
      - Vorbis: key (lower-case)
      - MP4 freeform: ----:com.apple.iTunes:<UPPER_CASE>
      - APEv2: key (upper-case)
    """
    upper = tag_name.upper()
    # Some tags have different conventional names on MP4 to match tooling
    # expectations (Jaikoz/Picard). Map those here.
    if tag_name == "mood_dance":
        mp4_upper = "MOOD_DANCEABILITY"
    elif tag_name in ("electronic", "acoustic", "instrumental"):
        mp4_upper = f"MOOD_{upper}"
    elif tag_name == "timbre_brightness":
        mp4_upper = "TIMBRE_BRIGHTNESS"
    else:
        mp4_upper = upper
    mp4_key = f"----:com.apple.iTunes:{mp4_upper}"

    def _write(f: mutagen.FileType, score: int) -> None:
        from mutagen.aiff import AIFF
        from mutagen.apev2 import APEv2File
        from mutagen.id3 import ID3FileType, TXXX
        from mutagen.mp4 import MP4, MP4FreeForm

        if isinstance(f, (ID3FileType, AIFF)):
            if f.tags is None:
                f.add_tags()
            # Match the exact Jaikoz-visible ID3 descriptions used in the
            # verified baseline files.
            if tag_name == "mood_dance":
                desc = "MOOD_DANCEABILITY"
            elif tag_name.startswith("mood_"):
                mood_name = tag_name.split("_", 1)[1].upper()
                desc = f"MOOD_{mood_name}"
            elif tag_name in ("electronic", "acoustic", "instrumental"):
                desc = f"MOOD_{upper}"
            elif tag_name == "timbre_brightness":
                desc = "TIMBRE_BRIGHTNESS"
            else:
                desc = upper

            txxx_key = f"TXXX:{desc}"
            f.tags[txxx_key] = TXXX(encoding=3, desc=desc, text=[str(score)])
        elif isinstance(f, MP4):
            # MP4 freeform atoms store arbitrary bytes; encode as UTF-8.
            f[mp4_key] = [MP4FreeForm(str(score).encode())]
        elif isinstance(f, APEv2File):
            # Prefer MOOD_* for character tags to match tooling
            if tag_name in ("electronic", "acoustic", "instrumental"):
                f[f"MOOD_{upper}"] = str(score)
            elif tag_name == "mood_dance":
                f["MOOD_DANCEABILITY"] = str(score)
            elif tag_name == "timbre_brightness":
                f["TIMBRE_BRIGHTNESS"] = str(score)
            else:
                f[upper] = str(score)
        else:
            # Vorbis comments are case-insensitive, so writing both lowercase and
            # uppercase aliases does not create distinct fields. Write the exact
            # keys Jaikoz expects for FLAC/Ogg.
            val = [str(score)]
            if tag_name.startswith("mood_"):
                if tag_name == "mood_dance":
                    f["MOOD_DANCEABILITY"] = val
                else:
                    f[upper] = val
            elif tag_name in ("electronic", "acoustic", "instrumental"):
                f[f"MOOD_{upper}"] = val
            elif tag_name == "timbre_brightness":
                f["TIMBRE_BRIGHTNESS"] = val
            elif tag_name == "tonality":
                f["TONALITY"] = val
            else:
                f[upper] = val

    return _write


def _make_text_list_writer(
    tag_name: str,
) -> Callable[[mutagen.FileType, list[str]], None]:
    """Return a writer for multi-value text tags stored as '; '-joined text."""
    upper = tag_name.upper()
    mp4_key = f"----:com.apple.iTunes:{upper}"

    def _write(f: mutagen.FileType, values: list[str]) -> None:
        from mutagen.aiff import AIFF
        from mutagen.apev2 import APEv2File
        from mutagen.id3 import ID3FileType, TMOO, TXXX
        from mutagen.mp4 import MP4, MP4FreeForm

        if not values:
            return

        joined = "; ".join(values)

        if isinstance(f, (ID3FileType, AIFF)):
            if f.tags is None:
                f.add_tags()
            if tag_name == "mood":
                f.tags["TMOO"] = TMOO(encoding=3, text=[joined])
            else:
                desc = upper
                txxx_key = f"TXXX:{desc}"
                f.tags[txxx_key] = TXXX(encoding=3, desc=desc, text=[joined])
        elif isinstance(f, MP4):
            f[mp4_key] = [MP4FreeForm(joined.encode())]
        elif isinstance(f, APEv2File):
            f[upper] = joined
        else:
            f[upper] = [joined]

    return _write


def _write_acoustid_fingerprint(f: mutagen.FileType, value: str) -> None:
    """Write the raw Chromaprint fingerprint string into *f*.

    Uses the same key conventions as Picard / acoustag so the tag is
    interoperable with those tools:
      - ID3 (MP3, AIFF)  : TXXX:Acoustid Fingerprint
      - MP4/M4A           : ----:com.apple.iTunes:Acoustid Fingerprint
      - ASF/WMA           : Acoustid/Fingerprint
      - Everything else   : acoustid_fingerprint Vorbis comment
    """
    # Deferred imports — mutagen format classes are only needed at write time.
    from mutagen.aiff import AIFF
    from mutagen.asf import ASF, ASFUnicodeAttribute
    from mutagen.id3 import ID3FileType, TXXX
    from mutagen.mp4 import MP4, MP4FreeForm

    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            f.add_tags()
        desc = "Acoustid Fingerprint"
        f.tags.delall(f"TXXX:{desc}")
        f.tags.add(TXXX(encoding=3, desc=desc, text=[value]))
    elif isinstance(f, MP4):
        f["----:com.apple.iTunes:Acoustid Fingerprint"] = [MP4FreeForm(value.encode())]
    elif isinstance(f, ASF):
        if f.tags is None:
            f.add_tags()
        f.tags["Acoustid/Fingerprint"] = [ASFUnicodeAttribute(value)]
    else:
        # Vorbis comments (FLAC, Ogg Vorbis, Ogg Opus, WavPack, …)
        f["acoustid_fingerprint"] = [value]


def _write_key(f: mutagen.FileType, value: str) -> None:
    """Write the musical key string into *f* using standard interoperable fields.

    Field mapping:
    - ID3 (mp3, aiff …) : TKEY frame
    - MP4/AAC            : ----:com.apple.iTunes:initialkey  (Navidrome alias)
    - Everything else    : KEY Vorbis comment  (Jaikoz-canonical for FLAC/Ogg)
    """
    from mutagen.aiff import AIFF
    from mutagen.id3 import ID3FileType, TKEY
    from mutagen.mp4 import MP4, MP4FreeForm

    if isinstance(f, (ID3FileType, AIFF)):
        if f.tags is None:
            f.add_tags()
        f.tags["TKEY"] = TKEY(encoding=3, text=[value])
    elif isinstance(f, MP4):
        f["----:com.apple.iTunes:initialkey"] = [MP4FreeForm(value.encode())]
    else:
        # Vorbis comments: Jaikoz uses KEY as the canonical field name.
        # Navidrome and most other players also recognise KEY.
        f["KEY"] = [value]


def _fmt_tag_value(tag_name: str, value: object) -> str:
    """Format a tag value for log output.

    Long opaque strings (e.g. the raw Chromaprint fingerprint) are truncated
    to ``first5…last5`` so they don't flood the TUI or log file.
    """
    s = str(value)
    if tag_name == "acoustid_fingerprint" and len(s) > 13:  # noqa: PLR2004
        return f"{s[:5]}…{s[-5:]}"
    return s


# Build the full writer dispatch table: bpm + Essentia score tags + mood/key text tags.
_TAG_WRITERS: dict[str, Callable[..., None]] = {
    "bpm": _write_bpm,
    **{name: _make_score_writer(name) for name in _ESSENTIA_TAG_MODELS},
    "mood": _make_text_list_writer("mood"),
    "key": _write_key,
    "acoustid_fingerprint": _write_acoustid_fingerprint,
}


# ── ffmpeg decoder ─────────────────────────────────────────────────────────────


# Maximum wall-clock seconds to allow ffmpeg to decode a single file.
# NFS stalls, corrupt containers, or pathological files can otherwise block
# the worker thread indefinitely.  180-second clips at 22 kHz decode in well
# under a minute on any modern machine; 120 s is generous.
_FFMPEG_TIMEOUT_S = 120


def _load_mono_ffmpeg(
    path: str,
    sr: int = 22050,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Decode *path* with ffmpeg and return ``(y, sr)`` — mono float32.

    Why ffmpeg instead of letting deeprhythm call librosa.load() internally:
    - .m4a / AAC files often fail in libsndfile, causing librosa to fall back
      to the deprecated audioread backend and emit warnings on every file.
    - ffmpeg handles virtually every codec/container without complaints.

    Raises ``RuntimeError`` if the decode takes longer than ``_FFMPEG_TIMEOUT_S``
    seconds, which prevents a single NFS-stalled or pathological file from
    blocking the entire worker thread forever.
    """
    cmd = ["ffmpeg", "-v", "error", "-err_detect", "ignore_err", "-nostdin", "-vn"]

    if offset and offset > 0:
        cmd += ["-ss", str(offset)]

    cmd += ["-i", path]

    if duration and duration > 0:
        cmd += ["-t", str(duration)]

    # Mono float32 PCM at the desired sample rate → stdout
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg timed out after {_FFMPEG_TIMEOUT_S}s decoding {path} "
            f"(NFS stall or corrupt container?)"
        )

    # Check returncode first (actual ffmpeg error), then empty stdout separately
    # (successful decode but no usable audio — e.g. a very short/silent track).
    err = proc.stderr.decode("utf-8", "replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed (rc={proc.returncode}): "
            f"{err[:500] if err else 'unknown error'}"
        )
    if not proc.stdout:
        raise RuntimeError(
            "ffmpeg produced no audio data (file may be too short or silent)"
            + (f": {err[:200]}" if err else "")
        )

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y, sr


# ── Audio prefetch ─────────────────────────────────────────────────────────────


@dataclass
class PreloadedAudio:
    """Decoded audio buffers for all three sample rates used per file.

    All three ffmpeg decode passes are kicked off concurrently before
    inference starts so they can overlap with the previous file's model
    work.  Each field is ``None`` when decoding failed or was not needed;
    callers fall back to decoding inline when a field is ``None``.

    bpm_audio    — (samples, sr) at 22 050 Hz, 180-second centre clip,
                   for DeepRhythm.
    essentia_audio — samples at 16 000 Hz (full track) for the EffNet
                   extractor.  ``None`` when the file already has cached
                   embeddings (no decode needed).
    key_audio    — samples at 44 100 Hz (full track) for KeyExtractor.
                   ``None`` when the key tag is not being computed.
    errors       — per-stream decode errors, keyed by stream name.  The
                   worker logs these and falls back to inline decoding.
    """

    bpm_audio: tuple[np.ndarray, int] | None = None
    essentia_audio: np.ndarray | None = None
    key_audio: np.ndarray | None = None
    errors: dict[str, Exception] = field(default_factory=dict)


# A small thread pool used exclusively for concurrent ffmpeg subprocesses
# during audio prefetch.  Three workers matches the three sample rates
# (22 050 / 16 000 / 44 100 Hz) that can be decoded simultaneously.
# The pool is module-level so it is shared across Worker instances and
# survives across batch passes without the cost of repeated pool creation.
#
# Threads are daemon so that an in-flight ffmpeg decode does not prevent the
# process from exiting after the app closes.  The OS will terminate the child
# ffmpeg subprocesses when the process exits.  Worker.close() additionally
# calls _PREFETCH_POOL.shutdown(cancel_futures=True) to drop any queued (not
# yet started) futures immediately on shutdown.
class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are daemon threads.

    Python 3.12's ThreadPoolExecutor has no thread_factory parameter, so we
    override ``_adjust_thread_count`` to pass ``daemon=True`` when creating
    each worker thread.  Daemon threads do not prevent process exit — if the
    app closes while an ffmpeg decode is in flight the OS reclaims the child
    process naturally.
    """

    def _adjust_thread_count(self) -> None:
        if self._idle_semaphore.acquire(timeout=0):
            return

        def _weakref_cb(_: object, q: object = self._work_queue) -> None:
            q.put(None)  # type: ignore[union-attr]

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                target=_cft._worker,
                args=(
                    weakref.ref(self, _weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                name=thread_name,
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _cft._threads_queues[t] = self._work_queue


_PREFETCH_POOL = _DaemonThreadPoolExecutor(
    max_workers=3,
    thread_name_prefix="audio-prefetch",
)


def _preload_audio(
    filepath_str: str,
    track_len: float,
    needs_bpm: bool,
    needs_essentia: bool,
    needs_key: bool,
    has_cached_embeddings: bool,
) -> PreloadedAudio:
    """Decode audio at all required sample rates concurrently.

    Fires up to three ffmpeg subprocesses in parallel threads so that I/O
    and decode work overlaps with the previous file's model inference.

    *needs_bpm*, *needs_essentia*, *needs_key* gate which streams are
    decoded.  *has_cached_embeddings* suppresses the 16 kHz decode when
    the embeddings cache already has a hit (no EffNet forward pass needed).

    All errors are captured per-stream in ``PreloadedAudio.errors``; the
    caller is responsible for deciding whether to fall back or fail.
    """
    result = PreloadedAudio()
    futures: dict[str, Future[object]] = {}

    # ── BPM (22 050 Hz, centre-clipped 180 s) ─────────────────────────────
    if needs_bpm:
        bpm_sr = 22050
        clip_len = 180.0
        t_start = (track_len - clip_len) / 2.0 if track_len > clip_len else 0.0
        duration = clip_len

        futures["bpm"] = _PREFETCH_POOL.submit(
            _load_mono_ffmpeg, filepath_str, bpm_sr, t_start, duration
        )

    # ── Essentia EffNet (16 000 Hz, full track) ────────────────────────────
    # Skip when cached embeddings are available — no audio decode needed.
    if needs_essentia and not has_cached_embeddings:
        futures["essentia"] = _PREFETCH_POOL.submit(
            _load_mono_ffmpeg, filepath_str, _ESSENTIA_SR
        )

    # ── KeyExtractor (44 100 Hz, full track) ──────────────────────────────
    if needs_key:
        futures["key"] = _PREFETCH_POOL.submit(
            _load_mono_ffmpeg, filepath_str, _KEY_EXTRACTOR_SR
        )

    # Collect results — block until all submitted decodes finish.
    for name, future in futures.items():
        try:
            decoded = future.result()
            if name == "bpm":
                result.bpm_audio = decoded  # type: ignore[assignment]
            elif name == "essentia":
                audio, _ = decoded  # type: ignore[misc]
                result.essentia_audio = audio
            elif name == "key":
                audio, _ = decoded  # type: ignore[misc]
                result.key_audio = audio
        except Exception as exc:
            result.errors[name] = exc

    return result


# ── Essentia inference engine ──────────────────────────────────────────────────


class EssentiaEngine:
    """Manages Essentia TensorFlow models for tag inference.

    Models are loaded lazily on first use and then kept in memory for the
    lifetime of the engine. Missing .pb files produce a warning and are skipped
    — their tags stay in ``needs_work`` and will be retried on the next pass.

    The engine is not thread-safe; it is intended to be used exclusively from
    the Worker's background thread.
    """

    def __init__(self, models_dir: Path) -> None:
        _configure_essentia_runtime()
        self._models_dir = models_dir
        # Lazy-loaded EffNet extractor and classifier heads.
        self._extractor: object | None = None
        self._classifiers: dict[str, object] = {}
        self._metadata: dict[str, dict[str, object]] = {}
        self._extractor_broken: bool = False
        # Set of tag names whose model files were confirmed missing, so we can
        # warn once and then skip without repeated filesystem checks.
        self._missing_models: set[str] = set()
        # Set of tag names whose classifier failed to execute (bad graph or
        # incompatible output node). We suppress further attempts to avoid
        # Essentia C++ warnings spamming the TUI.
        self._broken_models: set[str] = set()
        self._broken_metadata: set[str] = set()
        # TempoCNN wrapper — provides globalTempo scalar via majority voting.
        self._tempocnn: object | None = None
        self._tempocnn_broken: bool = False
        # TensorflowPredictTempoCNN — provides raw (n_patches, 256) probability
        # matrix used to score and compare BPM candidates.
        self._tempocnn_predictor: object | None = None
        self._tempocnn_predictor_broken: bool = False
        # KeyExtractor is a classical algorithm (no model file) but is cached
        # here to avoid repeated C++ object construction on every file.
        self._key_extractor: object | None = None

    def _model_path(self, filename: str) -> Path | None:
        """Return the path to *filename* if it exists, else None."""
        p = self._models_dir / filename
        return p if p.exists() else None

    def _load_extractor(self) -> object | None:
        """Lazy-load the shared Discogs-EffNet extractor."""
        if self._extractor is not None:
            return self._extractor
        if self._extractor_broken:
            return None

        pb = self._model_path(_EFFNET_EXTRACTOR_FILENAME)
        if pb is None:
            logger.warning(
                "Essentia extractor model not found: {} — "
                "run `musictagger-download-models` to fetch it",
                self._models_dir / _EFFNET_EXTRACTOR_FILENAME,
            )
            return None

        from essentia.standard import TensorflowPredictEffnetDiscogs  # type: ignore[import]

        logger.debug("Loading Essentia extractor from {} — start", pb)
        try:
            self._extractor = TensorflowPredictEffnetDiscogs(
                graphFilename=str(pb),
                output="PartitionedCall:1",
            )
        except Exception as exc:
            logger.warning("Failed to load Essentia extractor ({}). Disabling.", exc)
            self._extractor_broken = True
            self._extractor = None
            return None
        logger.debug("Loading Essentia extractor — done")
        return self._extractor

    def _load_metadata(
        self,
        metadata_key: str,
        filename: str,
    ) -> dict[str, object] | None:
        """Lazy-load JSON metadata for classifiers that expose named labels."""
        if metadata_key in self._metadata:
            return self._metadata[metadata_key]
        if metadata_key in self._broken_metadata:
            return None

        path = self._model_path(filename)
        if path is None:
            if metadata_key not in self._missing_models:
                logger.warning(
                    "Essentia metadata not found for {!r}: {} — skipping",
                    metadata_key,
                    self._models_dir / filename,
                )
                self._missing_models.add(metadata_key)
            return None

        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            logger.warning(
                "Failed to read Essentia metadata for {!r} from {}: {}",
                metadata_key,
                path,
                exc,
            )
            self._broken_metadata.add(metadata_key)
            return None

        self._metadata[metadata_key] = data
        return data

    def _load_classifier(
        self,
        classifier_key: str,
        filename: str,
        *,
        output: str = "model/Softmax",
        input: str | None = None,
    ) -> object | None:
        """Lazy-load a TensorflowPredict2D classifier head."""
        if classifier_key in self._classifiers:
            return self._classifiers[classifier_key]
        if classifier_key in self._broken_models:
            return None

        pb = self._model_path(filename)
        if pb is None:
            if classifier_key not in self._missing_models:
                logger.warning(
                    "Essentia classifier model not found for {!r}: {} — skipping",
                    classifier_key,
                    self._models_dir / filename,
                )
                self._missing_models.add(classifier_key)
            return None

        from essentia.standard import TensorflowPredict2D  # type: ignore[import]

        logger.debug(
            "Loading Essentia classifier for {!r} from {} — start", classifier_key, pb
        )
        kwargs: dict[str, str] = {"graphFilename": str(pb), "output": output}
        if input is not None:
            kwargs["input"] = input
        try:
            self._classifiers[classifier_key] = TensorflowPredict2D(**kwargs)
        except Exception as exc:
            logger.warning(
                "Failed to load Essentia classifier for {!r} ({}). Disabling.",
                classifier_key,
                exc,
            )
            self._broken_models.add(classifier_key)
            return None
        logger.debug("Loading Essentia classifier for {!r} — done", classifier_key)
        return self._classifiers[classifier_key]

    def _load_audio(
        self,
        filepath_str: str,
        preloaded: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode *filepath_str* to mono float32 at 16 kHz via ffmpeg.

        Previously used Essentia's MonoLoader for this, but MonoLoader's C++
        decoder can segfault on certain MP3 files (e.g. files with malformed
        headers or unusual encoding), killing the entire process with no
        recoverable exception.  ffmpeg handles those files safely and already
        has timeout protection via subprocess.run(timeout=...), so it is used
        here for all Essentia audio loading.

        *preloaded* — if provided, used directly without calling ffmpeg.
        """
        if preloaded is not None:
            audio = preloaded
        else:
            audio, _ = _load_mono_ffmpeg(filepath_str, sr=_ESSENTIA_SR)

        if audio is None or len(audio) == 0:
            raise RuntimeError(f"ffmpeg returned empty audio for {filepath_str}")

        return audio

    def _predict_key(
        self,
        filepath_str: str,
        preloaded: np.ndarray | None = None,
    ) -> str | None:
        """Estimate the musical key of *filepath_str* using Essentia KeyExtractor.

        KeyExtractor is a classical HPCP-based algorithm bundled with Essentia —
        no model file is required.  Audio is decoded at 44100 Hz via ffmpeg
        (not MonoLoader — see _load_audio for why MonoLoader is avoided).

        *preloaded* — if provided, used directly without calling ffmpeg.

        Returns a string such as ``"C major"`` or ``"A minor"``, or ``None`` if
        the algorithm fails (e.g. file too short or silent).
        """
        from essentia.standard import KeyExtractor  # type: ignore[import]

        if preloaded is not None:
            audio = preloaded
        else:
            try:
                audio, _ = _load_mono_ffmpeg(filepath_str, sr=_KEY_EXTRACTOR_SR)
            except RuntimeError as exc:
                logger.warning(
                    "KeyExtractor: ffmpeg decode failed on {}: {}", filepath_str, exc
                )
                return None

        if audio is None or len(audio) == 0:
            logger.warning("KeyExtractor: empty audio from {}", filepath_str)
            return None

        if self._key_extractor is None:
            self._key_extractor = KeyExtractor()

        try:
            key, scale, strength = self._key_extractor(audio)
        except Exception as exc:
            logger.warning("KeyExtractor failed on {}: {}", filepath_str, exc)
            return None

        logger.debug(
            "KeyExtractor: {} {} (strength {:.3f}) on {}",
            key,
            scale,
            strength,
            Path(filepath_str).name,
        )
        return f"{key} {scale}"

    def _extract_embeddings(self, filepath_str: str, audio: np.ndarray) -> object:
        """Run the shared Discogs-EffNet extractor for one audio buffer."""
        extractor = self._load_extractor()
        if extractor is None:
            raise RuntimeError("Essentia extractor is unavailable")

        try:
            return extractor(audio)
        except Exception as exc:
            logger.warning(
                "Essentia extractor failed on {}: {} — disabling extractor for the session",
                filepath_str,
                exc,
            )
            self._extractor_broken = True
            self._extractor = None
            raise RuntimeError(
                f"Essentia extractor failed on {filepath_str}: {exc}"
            ) from exc

    def _predict_effnet_scores(
        self,
        filepath_str: str,
        embeddings: object,
        tag_names: list[str],
    ) -> dict[str, int]:
        """Run the existing binary EffNet score classifiers."""
        available = []
        for tag_name in tag_names:
            if tag_name in self._broken_models:
                continue
            filename, idx = _ESSENTIA_TAG_MODELS[tag_name]
            clf = self._load_classifier(tag_name, filename)
            if clf is not None:
                available.append((tag_name, clf, idx))

        if not available:
            return {}

        results: dict[str, int] = {}
        for tag_name, classifier, positive_idx in available:
            try:
                predictions = classifier(embeddings)
                score_raw = float(np.mean(predictions[:, positive_idx]))
                results[tag_name] = int(round(score_raw * 100))
            except Exception as exc:
                logger.warning(
                    "Essentia classifier failed for {!r} on {}: {} — disabling this model for the session",
                    tag_name,
                    filepath_str,
                    exc,
                )
                self._broken_models.add(tag_name)

        return results

    def _predict_mood_from_genre(
        self,
        filepath_str: str,
        embeddings: object,
        threshold: float,
        min_results: int,
        max_results: int,
        log_fn: Callable[[str], None] | None = None,
    ) -> list[str]:
        """Predict mood labels using the Genre Discogs400 classifier.

        Runs the Genre Discogs400 classifier over the shared EffNet embeddings
        and returns plain subgenre label strings selected by *threshold*,
        *min_results*, and *max_results* (see ``_top_subgenres_from_genre_scores``).

        If *log_fn* is provided it is called with a scored summary string of the
        form ``"Mood: Label (0.82), Label (0.61)"`` so callers can surface
        confidence scores in the TUI log without embedding them in tag values.

        Returns an empty list if the model or its metadata is unavailable or
        fails at runtime.
        """
        classifier_key = "genre_discogs400"
        metadata_key = f"{classifier_key}:metadata"

        if classifier_key in self._broken_models:
            return []

        classifier = self._load_classifier(
            classifier_key,
            _GENRE_DISCOGS400_MODEL_FILENAME,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        metadata = self._load_metadata(
            metadata_key, _GENRE_DISCOGS400_METADATA_FILENAME
        )
        if classifier is None or metadata is None:
            return []

        classes = metadata.get("classes")
        if not isinstance(classes, list) or not all(
            isinstance(c, str) for c in classes
        ):
            if metadata_key not in self._broken_metadata:
                logger.warning(
                    "Essentia metadata for {!r} does not contain a valid classes list",
                    classifier_key,
                )
                self._broken_metadata.add(metadata_key)
            return []

        try:
            predictions = np.asarray(classifier(embeddings), dtype=float)
        except Exception as exc:
            logger.warning(
                "Essentia classifier failed for {!r} on {}: {} — disabling this model for the session",
                classifier_key,
                filepath_str,
                exc,
            )
            self._broken_models.add(classifier_key)
            return []

        averaged = predictions.mean(axis=0) if predictions.ndim > 1 else predictions

        if len(classes) != len(averaged):
            logger.warning(
                "Essentia classifier output size mismatch for {!r}: {} labels, {} scores",
                classifier_key,
                len(classes),
                len(averaged),
            )
            self._broken_models.add(classifier_key)
            return []

        labels = _top_subgenres_from_genre_scores(
            classes, averaged, threshold, min_results, max_results
        )
        if labels:
            # Build scored representations for logging only; scores must not
            # appear in the tag values written to files.
            # Use label_for_class so display names match the selected labels,
            # and keep the maximum score when two class keys resolve to the
            # same display name (collision case).
            subgenre_scores: dict[str, float] = {}
            for c, s in zip(classes, averaged.tolist()):
                key = label_for_class(c)
                if key not in subgenre_scores or s > subgenre_scores[key]:
                    subgenre_scores[key] = s
            # Plain version for the loguru file log.
            scored_plain = ", ".join(
                f"{lbl} ({subgenre_scores.get(lbl, 0.0):.2f})" for lbl in labels
            )
            logger.debug(
                "Mood prediction for {}: {}", Path(filepath_str).name, scored_plain
            )
            if log_fn is not None:
                # Rich markup version for the TUI: label bold, score dim.
                # Labels come from model metadata so must be escaped.
                scored_markup = ", ".join(
                    f"[bold]{markup_escape(lbl)}[/bold]"
                    f" [dim]({subgenre_scores.get(lbl, 0.0):.2f})[/dim]"
                    for lbl in labels
                )
                log_fn(f"[dim]Mood:[/dim] {scored_markup}")
        return labels

    # ── TempoCNN fallback ──────────────────────────────────────────────────────

    def _load_tempocnn(self) -> object | None:
        """Lazy-load the TempoCNN model, returning None if unavailable."""
        if self._tempocnn is not None:
            return self._tempocnn
        if self._tempocnn_broken:
            return None

        pb = self._model_path(_TEMPOCNN_MODEL_FILENAME)
        if pb is None:
            logger.warning(
                "TempoCNN model not found: {} —"
                " run `musictagger-download-models` to fetch it",
                self._models_dir / _TEMPOCNN_MODEL_FILENAME,
            )
            self._tempocnn_broken = True
            return None

        from essentia.standard import TempoCNN  # type: ignore[import]

        logger.debug("Loading TempoCNN from {} — start", pb)
        try:
            self._tempocnn = TempoCNN(graphFilename=str(pb))
        except Exception as exc:
            logger.warning("Failed to load TempoCNN ({}). BPM fallback disabled.", exc)
            self._tempocnn_broken = True
            return None
        logger.debug("Loading TempoCNN — done")
        return self._tempocnn

    def _load_tempocnn_predictor(self) -> object | None:
        """Lazy-load TensorflowPredictTempoCNN for raw probability output.

        Returns the (n_patches, 256) softmax distribution per patch, which is
        used to score BPM candidates against the model's own probability mass.
        """
        if self._tempocnn_predictor is not None:
            return self._tempocnn_predictor
        if self._tempocnn_predictor_broken:
            return None

        pb = self._model_path(_TEMPOCNN_MODEL_FILENAME)
        if pb is None:
            # Already warned in _load_tempocnn; stay silent here.
            self._tempocnn_predictor_broken = True
            return None

        from essentia.standard import TensorflowPredictTempoCNN  # type: ignore[import]

        logger.debug("Loading TensorflowPredictTempoCNN from {} — start", pb)
        try:
            self._tempocnn_predictor = TensorflowPredictTempoCNN(graphFilename=str(pb))
        except Exception as exc:
            logger.warning(
                "Failed to load TensorflowPredictTempoCNN ({}). Scoring disabled.", exc
            )
            self._tempocnn_predictor_broken = True
            return None
        logger.debug("Loading TensorflowPredictTempoCNN — done")
        return self._tempocnn_predictor

    def predict_bpm(self, filepath_str: str, deeprhythm_bpm: float) -> float:
        """Compare DeepRhythm and TempoCNN results, returning the most likely BPM.

        Both models are run.  TempoCNN's majority-vote global tempo and the
        DeepRhythm BPM are each scored against TempoCNN's averaged per-patch
        probability distribution (256 classes, 30–286 BPM).  The candidate with
        the higher probability mass wins.

        If TempoCNN is unavailable or fails at any stage, *deeprhythm_bpm* is
        returned unchanged.
        """
        wrapper = self._load_tempocnn()
        predictor = self._load_tempocnn_predictor()
        if wrapper is None or predictor is None:
            return deeprhythm_bpm

        # Use ffmpeg instead of MonoLoader — see _load_audio for why.
        try:
            audio, _ = _load_mono_ffmpeg(filepath_str, sr=_TEMPOCNN_SR)
        except RuntimeError as exc:
            logger.warning(
                "TempoCNN ffmpeg decode failed on {}: {} — keeping DeepRhythm result",
                filepath_str,
                exc,
            )
            return deeprhythm_bpm

        try:
            # global_tempo: majority-vote scalar BPM from the wrapper.
            # raw_preds:    (n_patches, 256) softmax — used for scoring.
            global_tempo, _, _ = wrapper(audio)
            raw_preds = np.asarray(predictor(audio))  # (n_patches, 256)
        except Exception as exc:
            logger.warning(
                "TempoCNN failed on {}: {} — keeping DeepRhythm result",
                filepath_str,
                exc,
            )
            return deeprhythm_bpm

        tempocnn_bpm = float(global_tempo)

        # Average the per-patch distributions → single 256-element probability
        # vector over the BPM axis.  atleast_2d guards against a single patch
        # being returned as a 1-D array.
        probs = np.atleast_2d(raw_preds).mean(axis=0)  # (256,)
        bpm_axis = np.linspace(_TEMPOCNN_BPM_MIN, _TEMPOCNN_BPM_MAX, len(probs))

        def mass_near(target: float, window: float = 3.0) -> float:
            return float(probs[np.abs(bpm_axis - target) <= window].sum())

        dr_mass = mass_near(deeprhythm_bpm)
        tc_mass = mass_near(tempocnn_bpm)

        winner = tempocnn_bpm if tc_mass >= dr_mass else deeprhythm_bpm
        logger.debug(
            "BPM comparison on {}: DeepRhythm={:.1f} (mass={:.3f})"
            " TempoCNN={:.1f} (mass={:.3f}) → {:.1f}",
            Path(filepath_str).name,
            deeprhythm_bpm,
            dr_mass,
            tempocnn_bpm,
            tc_mass,
            winner,
        )
        return winner

    # ── Essentia EffNet + mood inference ──────────────────────────────────────

    def predict(
        self,
        filepath_str: str,
        tag_names: list[str],
        mood_threshold: float = 0.10,
        mood_min_results: int = 1,
        mood_max_results: int = 4,
        log_fn: Callable[[str], None] | None = None,
        cached_embeddings: object | None = None,
        preloaded_essentia: np.ndarray | None = None,
        preloaded_key: np.ndarray | None = None,
    ) -> tuple[dict[str, object], object | None]:
        """Run Essentia inference for all *tag_names* on *filepath_str*.

        Returns a ``({tag_name: value}, embeddings)`` tuple.  Values are either
        0–100 integers, multi-value text lists, or plain strings depending on
        the tag.  The ``embeddings`` return value is the raw EffNet embedding
        array (``np.ndarray`` of shape ``(n_patches, 1280)``) so the caller can
        store it in the embeddings cache; it is ``None`` when the EffNet
        pipeline did not run (e.g. only ``key`` was requested).

        The audio is decoded once via ffmpeg at 16 kHz and the embeddings are
        computed once, then each classifier head runs over the same embeddings.
        The ``key`` tag uses a separate 44100 Hz ffmpeg decode via KeyExtractor.

        *mood_threshold*, *mood_min_results*, and *mood_max_results* control
        the label-selection logic for the ``"mood"`` tag.

        *cached_embeddings* — if provided, the EffNet audio decode and
        extractor forward pass are skipped and this array is used directly.

        *preloaded_essentia* — pre-decoded 16 kHz audio array; skips the
        ffmpeg decode when provided (from ``_preload_audio``).

        *preloaded_key* — pre-decoded 44 100 Hz audio array; skips the
        KeyExtractor ffmpeg decode when provided.
        """
        effnet_tags = [t for t in tag_names if t in _ESSENTIA_TAG_MODELS]
        wants_mood = "mood" in tag_names
        wants_key = "key" in tag_names

        if not effnet_tags and not wants_mood and not wants_key:
            return {}, None

        results: dict[str, object] = {}
        embeddings: object | None = None

        # EffNet pipeline: decode at 16 kHz once for all embedding-based tags.
        if effnet_tags or wants_mood:
            if cached_embeddings is not None:
                embeddings = cached_embeddings
                logger.debug(
                    "Essentia: using cached embeddings for {}",
                    Path(filepath_str).name,
                )
            else:
                logger.debug("Essentia: loading audio for {}", Path(filepath_str).name)
                audio = self._load_audio(filepath_str, preloaded=preloaded_essentia)
                logger.debug(
                    "Essentia: audio loaded, extracting embeddings for {}",
                    Path(filepath_str).name,
                )
                embeddings = self._extract_embeddings(filepath_str, audio)
                logger.debug(
                    "Essentia: embeddings done for {}", Path(filepath_str).name
                )

            if effnet_tags:
                results.update(
                    self._predict_effnet_scores(filepath_str, embeddings, effnet_tags)
                )
            if wants_mood:
                mood_values = self._predict_mood_from_genre(
                    filepath_str,
                    embeddings,
                    threshold=mood_threshold,
                    min_results=mood_min_results,
                    max_results=mood_max_results,
                    log_fn=log_fn,
                )
                if mood_values:
                    results["mood"] = mood_values

        # KeyExtractor: separate 44100 Hz decode, no model file required.
        if wants_key:
            key_value = self._predict_key(filepath_str, preloaded=preloaded_key)
            if key_value is not None:
                results["key"] = key_value

        return results, embeddings


# ── Worker ─────────────────────────────────────────────────────────────────────


class Worker:
    """Processes files that the inspector flagged as needing tag work.

    Uses DeepRhythm to predict BPM and Essentia for mood/timbre/tonality tags.
    Results are written into the audio file's tags with mutagen and the cache
    is updated.
    """

    def __init__(
        self,
        config: Config,
        cache: FileCache,
        log_fn: Callable[[str], None] | None = None,
        markup_log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self._log = log_fn or (lambda msg: None)
        # Separate callback for pre-trusted Rich markup strings (e.g. scored
        # mood lines).  Falls back to the plain log so callers never need to
        # branch on whether markup is available.
        self._log_markup = markup_log_fn or self._log
        self._running = False
        # Stop signal — set by stop(), cleared by reset().  Replaces the old
        # _stop_requested bool: threading.Event is the standard Python idiom
        # for cross-thread stop signalling and makes the semantics explicit.
        self._stop_event = threading.Event()
        self._processed = 0
        self._errors = 0
        self._predictor = None  # lazy — imported and initialised on first use
        self._essentia: EssentiaEngine | None = None  # lazy
        # Content-addressed cache for EffNet embeddings, keyed by
        # SHA-256(acoustid_fingerprint).  Always open — silently unused when
        # a file has no fingerprint tag.
        self._embedding_cache = EmbeddingCache(config.embeddings_db_path)
        # Monotonic timestamp updated at the start of each file and at run_pass
        # completion.  Used by the TUI watchdog to detect a hung worker thread.
        self._last_activity: float = 0.0
        # Rate tracking: start of the current pass and the processed count at
        # that moment, so the TUI can show files/sec and an ETA.
        self._pass_start: float = 0.0
        self._pass_processed_at_start: int = 0
        # Batch progress: total items fetched this pass and how many have been
        # handled (regardless of success/error) so the TUI can draw a bar.
        self._batch_total: int = 0
        self._batch_done: int = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def processed(self) -> int:
        return self._processed

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def last_activity(self) -> float:
        """Monotonic timestamp of the most recent worker heartbeat."""
        return self._last_activity

    @property
    def batch_total(self) -> int:
        """Number of files in the current (or last) batch."""
        return self._batch_total

    @property
    def batch_done(self) -> int:
        """Number of files handled so far in the current (or last) batch."""
        return self._batch_done

    @property
    def pass_rate(self) -> float:
        """Files processed per second in the current (or last) pass.

        Returns 0.0 if the pass has not yet completed any files.
        """
        elapsed = time.monotonic() - self._pass_start
        done_this_pass = self._processed - self._pass_processed_at_start
        if elapsed <= 0 or done_this_pass <= 0:
            return 0.0
        return done_this_pass / elapsed

    # ── Predictor lifecycle ────────────────────────────────────────────────────

    def _init_predictor(self) -> object:
        """Initialise DeepRhythmPredictor, preferring CUDA over CPU."""
        from deeprhythm import DeepRhythmPredictor  # type: ignore[import]

        try:
            import torch  # type: ignore[import]

            if getattr(torch, "cuda", None) and torch.cuda.is_available():
                logger.debug("Initialising DeepRhythmPredictor on CUDA")
                try:
                    self._log("DeepRhythm: using CUDA")
                    return DeepRhythmPredictor(device="cuda")
                except Exception as exc:
                    logger.warning(
                        "DeepRhythm CUDA init failed ({}); falling back to CPU", exc
                    )
        except Exception as exc:
            logger.debug("Torch/CUDA check failed ({}); using CPU", exc)

        logger.debug("Initialising DeepRhythmPredictor on CPU")
        self._log("DeepRhythm: using CPU")
        try:
            return DeepRhythmPredictor(device="cpu")
        except Exception as exc:
            raise RuntimeError(f"DeepRhythm CPU init failed: {exc}") from exc

    def _get_predictor(self) -> object:
        if self._predictor is None:
            self._predictor = self._init_predictor()
        return self._predictor

    def _get_essentia(self) -> EssentiaEngine:
        """Lazy-initialise the shared Essentia engine."""
        if self._essentia is None:
            self._essentia = EssentiaEngine(self.config.models_dir)
        return self._essentia

    # ── BPM prediction ─────────────────────────────────────────────────────────

    def _predict_bpm(
        self,
        filepath_str: str,
        track_len: float = 0.0,
        preloaded: tuple[np.ndarray, int] | None = None,
    ) -> int:
        """Return integer BPM for the file at *filepath_str*.

        Decodes with ffmpeg (avoids librosa/audioread warnings on .m4a etc.),
        then calls DeepRhythmPredictor.predict_from_audio().

        Uses the middle 180 seconds of the track for more reliable results.
        For tracks shorter than 180 seconds the full track is used from the
        start.

        *track_len* is the known duration in seconds; pass it in from the
        caller's already-open mutagen handle to avoid a redundant file parse.
        If zero or unknown, the full file is used from the start.

        *preloaded* — if provided, the (samples, sr) tuple from a prior
        ``_preload_audio`` call is used directly and ffmpeg is not invoked.

        Raises RuntimeError for audio that is too short/silent or when the
        model cannot produce a result.  Raises AttributeError if deeprhythm /
        PyTorch internals fail on unanalysable audio.
        """
        sr = 22050
        clip_len = 180.0

        if preloaded is not None:
            y, sr = preloaded
        else:
            if track_len > clip_len:
                t_start = (track_len - clip_len) / 2.0
            else:
                t_start = 0.0

            duration = clip_len

            y, sr = _load_mono_ffmpeg(
                filepath_str, sr=sr, offset=t_start, duration=duration
            )

        # 2 seconds minimum — shorter clips produce meaningless or None results.
        if y is None or len(y) < sr * 2:
            raise RuntimeError("audio too short after decoding")

        predictor = self._get_predictor()

        # Accommodate slight API differences between deeprhythm versions.
        # Request the softmax confidence alongside the BPM estimate so we can
        # gate the TempoCNN fallback on low-confidence tracks.
        # Both call attempts are wrapped so any non-TypeError exception from the
        # fallback call is caught and re-raised as RuntimeError rather than
        # propagating unchecked.
        try:
            try:
                result, confidence = predictor.predict_from_audio(
                    y, sr, include_confidence=True
                )
            except TypeError:
                result, confidence = predictor.predict_from_audio(
                    y, sr=sr, include_confidence=True
                )
        except AttributeError as exc:
            # DeepRhythm returns None internally on silent/inaudible audio and
            # then tries to call .to() on it, producing an AttributeError.
            # Log the cause and surface a clear message.
            logger.debug(
                "DeepRhythm NoneType error on {} — track may be silent or inaudible ({})",
                Path(filepath_str).name,
                exc,
            )
            raise RuntimeError(
                "track may be silent or inaudible (model returned None)"
            ) from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"DeepRhythm inference failed: {exc}") from exc

        if result is None:
            raise RuntimeError("track may be silent or inaudible (model returned None)")

        bpm = float(result)
        logger.debug(
            "DeepRhythm: {:.1f} BPM (confidence {:.3f}) on {}",
            bpm,
            confidence,
            Path(filepath_str).name,
        )

        if confidence < self.config.bpm_confidence_threshold:
            logger.debug(
                "Low-confidence BPM ({:.3f} < {:.2f}) on {} — falling back to TempoCNN",
                confidence,
                self.config.bpm_confidence_threshold,
                Path(filepath_str).name,
            )
            bpm = self._get_essentia().predict_bpm(filepath_str, bpm)

        return int(round(bpm))

    # ── Tag writing ────────────────────────────────────────────────────────────

    def _write_tags(
        self,
        filepath_str: str,
        results: dict[str, object],
        mf: object | None = None,
    ) -> None:
        """Write computed tag values into the file using mutagen.

        *mf* may be a mutagen file object already opened by the caller
        (``easy=False``); if ``None``, the file is opened here.  Reusing
        the caller's handle avoids a redundant parse when the file was
        already opened to read metadata such as track duration.
        """
        f = mf if mf is not None else mutagen.File(filepath_str, easy=False)
        if f is None:
            raise RuntimeError(f"mutagen could not open {filepath_str}")

        for tag_name, value in results.items():
            writer = _TAG_WRITERS.get(tag_name)
            if writer is None:
                logger.debug("No writer registered for tag {!r} — skipping", tag_name)
                continue
            writer(f, value)

        f.save()

    # ── Core pass ──────────────────────────────────────────────────────────────

    def run_pass(self, batch_size: int = 20) -> int:
        """Process one batch of files needing work.

        Returns the number of file paths fetched from the queue at the start of
        the pass (which may be larger than the number successfully tagged if
        some files were skipped or errored).  Returns 0 only when the queue was
        empty — callers use this as the "nothing left to do" signal.
        """
        # Refuse to start a new pass after stop() has been called.  This closes
        # the race between Worker.run() (which loops until 0 is returned) and
        # the shutdown sequence (which closes the DB immediately once
        # worker.running flips to False between passes).
        if self._stop_event.is_set():
            return 0
        self._running = True
        self._last_activity = time.monotonic()
        self._pass_start = time.monotonic()
        self._pass_processed_at_start = self._processed

        enabled_tags = [t for t in TAGS if self.config.tag_cfg(t.name).enabled]
        filepaths = self.cache.needs_work(limit=batch_size, enabled_tags=enabled_tags)
        if not filepaths:
            self._running = False
            self._last_activity = time.monotonic()
            return 0

        self._batch_total = len(filepaths)
        self._batch_done = 0

        # Audio prefetch — overlap ffmpeg decodes with model inference.
        #
        # Pattern: before the loop, submit a speculative decode for filepaths[0].
        # At the top of each iteration, `current_prefetch` holds the future for
        # the current file; we collect it after the cheap DB/mutagen checks, then
        # immediately submit a new future for filepaths[i+1] so both can overlap
        # with the current file's inference.
        #
        # The first prefetch uses track_len=0.0 (unknown until mutagen opens the
        # file) which causes the BPM decode to start from t=0; for most files
        # under 180 s this is identical to the centred clip.  Subsequent prefetches
        # also use 0.0 for the next file's track_len for the same reason.
        current_prefetch: Future[PreloadedAudio] | None = _PREFETCH_POOL.submit(
            _preload_audio,
            filepaths[0],
            0.0,  # track_len unknown before mutagen open; safe fallback
            True,  # needs_bpm — speculative
            True,  # needs_essentia — speculative
            True,  # needs_key — speculative
            False,  # has_cached_embeddings — conservative: always prefetch
        )

        for i, filepath_str in enumerate(filepaths):
            if self._stop_event.is_set():
                # Best-effort cancel; the thread may already be running.
                if current_prefetch is not None:
                    current_prefetch.cancel()
                break

            # Heartbeat — lets the TUI watchdog know the thread is still alive.
            self._last_activity = time.monotonic()

            filepath = Path(filepath_str)
            if not filepath.exists():
                # Stale cache entry — file was moved or deleted since the
                # inspector ran.  Mark error so it leaves the work queue;
                # the cleanup job will remove the row entirely on its next run.
                # No flush here — the end-of-pass flush (or the periodic
                # every-5-files flush) will commit this write.
                logger.debug("Stale entry, skipping: {}", filepath_str)
                self.cache.mark_error(filepath, "file not found")
                # Drain the stale future and pre-fetch the next file.
                if current_prefetch is not None:
                    try:
                        current_prefetch.result()
                    except Exception:
                        pass  # result discarded; errors were captured inside
                if i + 1 < len(filepaths):
                    current_prefetch = _PREFETCH_POOL.submit(
                        _preload_audio,
                        filepaths[i + 1],
                        0.0,
                        True,
                        True,
                        True,
                        False,
                    )
                else:
                    current_prefetch = None
                continue

            missing = self._missing_tags(filepath_str)
            if not missing:
                # Inspector already satisfied this file; mark done and move on.
                self.cache.mark_done(filepath)
                if current_prefetch is not None:
                    try:
                        current_prefetch.result()
                    except Exception:
                        pass
                if i + 1 < len(filepaths):
                    current_prefetch = _PREFETCH_POOL.submit(
                        _preload_audio,
                        filepaths[i + 1],
                        0.0,
                        True,
                        True,
                        True,
                        False,
                    )
                else:
                    current_prefetch = None
                continue

            self.cache.mark_working(filepath)

            # Open the mutagen handle once per file.  track_len is passed to
            # _predict_bpm so it can centre the BPM clip without a second open,
            # and mf is reused by _write_tags to avoid parsing the file again.
            try:
                mf = mutagen.File(filepath_str, easy=False)
            except Exception:
                mf = None
            track_len = 0.0
            if mf is not None:
                try:
                    track_len = float(mf.info.length)
                except Exception:
                    pass

            # Collect the preloaded audio for the current file.  In practice
            # the decode has been running concurrently with the DB checks and
            # mutagen open above, so the wait here is usually brief or zero.
            preloaded: PreloadedAudio | None = None
            if current_prefetch is not None:
                try:
                    preloaded = current_prefetch.result()
                except Exception as exc:
                    # Unexpected error in the prefetch machinery itself
                    # (not a decode failure — those are captured inside
                    # PreloadedAudio.errors).  Fall back to inline decoding.
                    logger.debug(
                        "Audio prefetch machinery error for {}: {} — decoding inline",
                        filepath.name,
                        exc,
                    )
                    preloaded = None

            # Submit prefetch for the next file now — this will run concurrently
            # with the inference below.
            if i + 1 < len(filepaths):
                current_prefetch = _PREFETCH_POOL.submit(
                    _preload_audio,
                    filepaths[i + 1],
                    0.0,  # next file's track_len unknown until its mutagen open
                    True,
                    True,
                    True,
                    False,
                )
            else:
                current_prefetch = None

            try:
                tag_results = self.process_file(
                    filepath_str, missing, track_len=track_len, preloaded=preloaded
                )
                if not tag_results:
                    # All backends failed or returned nothing — leave the file
                    # in the queue so it will be retried rather than silently
                    # marking it done with no tags written.
                    self._log(f"No tags produced for {filepath.name} — leaving queued")
                    logger.warning(
                        "Worker produced no tag results for {} — skipping mark_done",
                        filepath.name,
                    )
                    self.cache.mark_error(filepath, "no tag results produced")
                    self._errors += 1
                else:
                    self._write_tags(filepath_str, tag_results, mf=mf)
                    # Update has_* columns for every tag that was written so the
                    # stats and inspection queue reflect the true state.  Using
                    # mark_done_with_tags() avoids leaving NULL has_* columns on
                    # done rows, which would otherwise cause the inspector to
                    # re-visit the file on every pass.
                    self.cache.mark_done_with_tags(filepath, list(tag_results.keys()))
                    # Sync the cached mtime/size to the post-write stat.
                    # Writing tags changes the file's mtime on disk; without
                    # this the scanner re-detects every written file as
                    # 'changed' on its next pass and re-queues it for
                    # inspection, creating an infinite loop when overwrite=True.
                    self.cache.refresh_stat(filepath)
                    self._processed += 1
                    written = ", ".join(
                        f"{k}={_fmt_tag_value(k, v)}" for k, v in tag_results.items()
                    )
                    self._log(f"Tagged {filepath.name}: {written}")
                    logger.info("Tagged {}: {}", filepath.name, written)
            except RuntimeError as exc:
                # Expected skips: audio too short, silent or inaudible tracks
                # (NoneType .to() from DeepRhythm, caught and re-raised in
                # _predict_bpm), or other recoverable inference failures.
                self._log(f"Skipped {filepath.name}: {exc}")
                logger.debug("Skipped {} ({})", filepath.name, exc)
                self.cache.mark_error(filepath, str(exc))
                self._errors += 1
            except Exception as exc:
                self._log(f"Error processing {filepath.name}: {exc}")
                logger.warning("Worker error for {}: {}", filepath.name, exc)
                self.cache.mark_error(filepath, str(exc))
                self._errors += 1

            self._batch_done += 1

            # Commit every few files rather than after every single file.
            # flush() acquires the cache lock to run COMMIT, so flushing
            # per-file creates unnecessary lock contention with the stats
            # refresh and other concurrent threads.  Every 5 files is
            # frequent enough for the TUI panels to feel responsive while
            # keeping lock acquisitions proportional to batch size.
            if self._batch_done % 5 == 0:
                self.cache.flush()

        # Always flush at the end of the pass to commit any remaining writes.
        self.cache.flush()

        logger.info(
            "Worker pass complete: {} processed, {} errors, {} total tagged",
            len(filepaths),
            self._errors,
            self._processed,
        )
        self._running = False
        self._last_activity = time.monotonic()
        return len(filepaths)

    def _missing_tags(self, filepath_str: str) -> list[str]:
        """Return tag names that the worker should compute for this file.

        A tag is included when ALL of the following are true:
          - it is enabled in config  (disabled tags are never computed)
          - it is absent in the cache (has_* = 0 or NULL) OR overwrite is set

        Queries the cache directly rather than assuming all tags are missing.
        """
        row = self.cache.get_tag_states(filepath_str)

        result = []
        for t, val in zip(TAGS, row if row is not None else [None] * len(TAGS)):
            cfg = self.config.tag_cfg(t.name)
            if not cfg.enabled:
                continue
            if not val or cfg.overwrite:
                result.append(t.name)
        return result

    def _compute_fingerprint(self, filepath_str: str) -> tuple[str, str] | None:
        """Compute a Chromaprint fingerprint for *filepath_str* via fpcalc.

        Returns ``(raw_fingerprint, fp_hash)`` on success, or ``None`` if
        ``fpcalc`` is unavailable or the file cannot be fingerprinted.
        Failures are logged at DEBUG level and never propagate — fingerprinting
        is always best-effort.
        """
        try:
            # Deferred import: pyacoustid is an optional runtime dependency.
            # Importing at the top of the module would cause an ImportError on
            # systems where the package is absent; deferring keeps startup fast.
            import acoustid  # noqa: PLC0415

            _duration, raw_fp = acoustid.fingerprint_file(
                filepath_str, force_fpcalc=True
            )
            if not raw_fp:
                return None
            raw_fp_str = raw_fp.decode() if isinstance(raw_fp, bytes) else str(raw_fp)
            from musictagger.embeddings import fingerprint_hash as _fp_hash

            return raw_fp_str, _fp_hash(raw_fp_str)
        except Exception as exc:
            logger.debug(
                "Fingerprint generation failed for {}: {}", Path(filepath_str).name, exc
            )
            return None

    def process_file(
        self,
        filepath_str: str,
        missing: list[str],
        track_len: float = 0.0,
        preloaded: PreloadedAudio | None = None,
    ) -> dict[str, object]:
        """Compute tag values for all *missing* tags on *filepath_str*.

        Returns a ``{tag_name: value}`` dict.  Dispatches to:
          - DeepRhythm for ``"bpm"``
          - Essentia for score tags, the ``"mood"`` text tag, and ``"key"``
          - fpcalc (via pyacoustid) for ``"acoustid_fingerprint"`` when the
            file has no fingerprint tag; the result is also stored in the
            embeddings cache so the EffNet forward pass can be skipped on the
            next worker run.

        *track_len* is the known duration in seconds from the caller's
        already-open mutagen handle; it is forwarded to ``_predict_bpm`` to
        avoid a redundant file parse when computing BPM clip offsets.

        *preloaded* — a ``PreloadedAudio`` from a prior ``_preload_audio``
        call.  When provided the three ffmpeg decode passes are skipped and
        the pre-decoded buffers are used directly.  Any stream that failed
        during prefetch (recorded in ``preloaded.errors``) falls back to
        inline decoding transparently.

        The Essentia engine is called once per file with all Essentia-backed
        missing tags batched together, so the shared EffNet extractor runs only
        once.  The ``key`` tag uses a separate audio decode inside the engine.
        """
        results: dict[str, object] = {}

        # Log any prefetch errors so they're visible in the log without
        # failing the file — each path below falls back to inline decoding.
        if preloaded and preloaded.errors:
            for stream, exc in preloaded.errors.items():
                logger.debug(
                    "Audio prefetch error for {} ({}): {} — will decode inline",
                    Path(filepath_str).name,
                    stream,
                    exc,
                )

        # ── BPM via DeepRhythm ─────────────────────────────────────────────
        if "bpm" in missing:
            bpm_preloaded = (
                preloaded.bpm_audio
                if preloaded and "bpm" not in preloaded.errors
                else None
            )
            results["bpm"] = self._predict_bpm(
                filepath_str, track_len=track_len, preloaded=bpm_preloaded
            )

        # ── Mood/timbre/tonality/key via Essentia ─────────────────────────
        essentia_missing = [
            t for t in missing if t in _ESSENTIA_TAG_MODELS or t in ("mood", "key")
        ]
        if essentia_missing:
            # Resolve the fingerprint hash for the embeddings cache.
            # If the inspector already stored one, use it directly.
            # Otherwise compute it now and write the fingerprint to the file.
            fp_hash = self.cache.get_fingerprint_hash(filepath_str)
            raw_fp: str | None = None

            if not fp_hash:
                fp_result = self._compute_fingerprint(filepath_str)
                if fp_result is not None:
                    raw_fp, fp_hash = fp_result
                    # Store the hash in cache.db so future passes skip fpcalc.
                    self.cache.set_fingerprint_hash(Path(filepath_str), fp_hash)
                    # Queue the fingerprint string for writing to the audio file.
                    results["acoustid_fingerprint"] = raw_fp
                    logger.debug("Computed fingerprint for {}", Path(filepath_str).name)

            # Use cached EffNet embeddings when available.
            cached_embeddings: object | None = None
            if fp_hash:
                cached_embeddings = self._embedding_cache.get(
                    fp_hash, _EFFNET_EXTRACTOR_FILENAME
                )

            # Pass preloaded buffers through to the Essentia engine.
            # Fall back to None (inline decode) if the prefetch errored.
            preloaded_essentia = (
                preloaded.essentia_audio
                if preloaded and "essentia" not in preloaded.errors
                else None
            )
            preloaded_key = (
                preloaded.key_audio
                if preloaded and "key" not in preloaded.errors
                else None
            )

            essentia_results, fresh_embeddings = self._get_essentia().predict(
                filepath_str,
                essentia_missing,
                mood_threshold=self.config.mood_threshold,
                mood_min_results=self.config.mood_min_results,
                mood_max_results=self.config.mood_max_results,
                log_fn=self._log_markup,
                cached_embeddings=cached_embeddings,
                preloaded_essentia=preloaded_essentia,
                preloaded_key=preloaded_key,
            )
            results.update(essentia_results)

            # Store freshly computed embeddings for future runs.
            if fp_hash and fresh_embeddings is not None and cached_embeddings is None:
                self._embedding_cache.put(
                    fp_hash, _EFFNET_EXTRACTOR_FILENAME, fresh_embeddings
                )
                self._embedding_cache.flush()

        # Log any tags that had no handler (future-proofing).
        handled = (
            set(results)
            | {"bpm", "mood", "key", "acoustid_fingerprint"}
            | set(_ESSENTIA_TAG_MODELS)
        )
        for tag_name in missing:
            if tag_name not in handled:
                logger.debug("No inference handler for tag {!r} — skipping", tag_name)

        return results

    def run(self, batch_size: int = 20) -> None:
        """Drain the work queue by calling run_pass() until empty.

        Recovers any rows left in 'working' status from a previous crashed or
        killed pass before starting — these rows would otherwise be silently
        excluded from needs_work() forever.  Recovery runs exactly once per
        worker session at startup, not on every tick.

        Loops immediately between passes so a full batch triggers the next
        pass without waiting for an external scheduler tick — keeping the
        ML models warm in memory across batches.  Exits when run_pass()
        returns 0 (empty queue) or stop() is called.
        """
        # One-shot recovery: reset rows stuck in 'working' from a prior crash.
        try:
            recovered = self.cache.requeue_working()
            if recovered:
                self.cache.flush()
                logger.warning(
                    "Worker startup: requeued {} stuck 'working' row(s) "
                    "from a previous interrupted pass",
                    recovered,
                )
        except Exception as exc:
            logger.warning("Worker startup requeue_working failed: {}", exc)

        while not self._stop_event.is_set():
            processed = self.run_pass(batch_size)
            if processed == 0:
                break

    def stop(self) -> None:
        """Signal the worker to stop at the next iteration boundary."""
        self._stop_event.set()
        self._running = False

    def reset(self) -> None:
        """Clear the stop signal so the worker can be relaunched."""
        self._stop_event.clear()

    def close(self) -> None:
        """Flush and close the embedding cache.

        Call this once when the worker is permanently shut down.  Safe to call
        multiple times.

        Note: _PREFETCH_POOL is module-level and shared across Worker instances.
        It is intentionally not shut down here — the daemon threads it manages
        will be abandoned naturally when the process exits.  Shutting the pool
        down here would permanently poison the module for any subsequent Worker
        instances in the same process (e.g. in tests).  If an explicit flush of
        queued prefetch work is needed at process exit, call
        ``_PREFETCH_POOL.shutdown(wait=False, cancel_futures=True)`` directly.
        """
        self._embedding_cache.close()
