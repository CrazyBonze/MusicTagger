# musictagger

`musictagger` is a terminal UI app for scanning a music library, detecting missing analysis tags, and filling them in with a background pipeline.

It is built to be gentle on large libraries: the scanner only uses `stat()`, the inspector only opens files to read tags, and the worker only processes files that are actually missing tags.

## Using musictagger

### Install

Python 3.12 is required. Install from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yourname/musictagger
cd musictagger
uv sync
```

### Quick start

Point musictagger at your music library:

```bash
musictagger /path/to/music
```

On first run a default config file is written to:

```text
~/.config/musictagger/config.toml
```

Open that file to customize paths, throttles, and which tags are enabled.

### Commands

| Command | Description |
|---|---|
| `musictagger /path/to/music` | Start the TUI and begin processing |
| `musictagger /path/to/music --config /path/to/config.toml` | Use a specific config file |
| `musictagger --info` | Print resolved config and storage paths, then exit |
| `musictagger --download-models /path/to/music` | Download Essentia models then start |
| `musictagger --no-download-models /path/to/music` | Skip model download check at startup |
| `musictagger --requeue-errors /path/to/music` | Reset error rows to queued for retry, then exit |
| `musictagger-download-models` | Download Essentia models without starting the TUI |
| `musictagger-download-models --models-dir /path` | Download models to a specific directory |
| `musictagger-download-models --force` | Re-download models even if already present |
| `musictagger-clear-tags /path/to/file-or-dir` | Remove musictagger-written tags from files |
| `musictagger-clear-tags /path/to/dir --recursive` | Same, recursively |
| `musictagger-inspect-tags /path/to/dir` | Print a CSV of tag presence across files |

### TUI controls

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Force a library scan now |
| `i` | Force an inspection pass now |
| `c` | Run orphan cleanup now |
| `r` | Requeue files that errored, for retry |
| `p` | Pause or resume the pipeline |
| `Tab` | Toggle between activity log and library overview |

The UI shows:
- overall library totals
- inspection and work queues
- scanner, inspector, worker, and cleanup state
- per-tag completion counts
- a live activity log with per-source filter toggles (Scanner / Inspector / Worker / Cleanup)

**Activity log:** scroll up to inspect past entries — auto-scroll pauses while you are scrolled up and resumes automatically when you return to the bottom.

### Config file

Default location:

```text
~/.config/musictagger/config.toml
```

Pass `--config /path/to/config.toml` to use a different file.

Settings are applied in this priority order: CLI arguments > config file > built-in defaults.

Full annotated example:

```toml
# Path to your music library.
# music_path = "/mnt/music"

[database]
# Where to store the SQLite cache.
path = "~/.local/share/musictagger/cache.db"
# Where to store the content-addressed embeddings cache.
# Safe to delete — the worker will repopulate it on the next pass.
embeddings_path = "~/.local/share/musictagger/embeddings.db"

[logging]
# Where to write the log file.
path      = "~/.local/share/musictagger/musictagger.log"
# Minimum level written: DEBUG, INFO, WARNING, ERROR
level     = "DEBUG"
rotation  = "10 MB"
retention = "30 days"

[scanner]
# When to run a full library walk (cron expression). Default: every hour.
cron             = "0 * * * *"
# Sleep between individual files (ms). Raise this for NFS mounts.
file_throttle_ms = 10
# Extra sleep between directories (ms).
dir_throttle_ms  = 200

[inspector]
# Sleep between file opens (ms).
throttle_ms = 50
# Files to inspect per pass.
batch_size  = 100

[worker]
# Files to process per pass.
batch_size = 20
# Directory containing Essentia .pb model files.
models_dir = "~/.local/share/musictagger/models"
# What to do at startup when models are missing: ask, always, never.
download_models = "ask"
# DeepRhythm confidence threshold. Below this value TempoCNN is used as a
# fallback. Tune by watching confidence scores in the log.
bpm_confidence_threshold = 0.10
# Genre Discogs400 Sigmoid threshold for mood label selection.
# Labels at or above this score are included (up to mood_max_results).
# If fewer than mood_min_results pass, the top mood_min_results are used
# regardless of score.
mood_threshold   = 0.15
mood_min_results = 1
mood_max_results = 4

[cleanup]
# When to remove cache rows for deleted files (cron). Default: midnight daily.
cron = "0 0 * * *"

# ── Per-tag settings ──────────────────────────────────────────────────────────
# enabled   = true   Include this tag in the pipeline (default).
#             false  Skip entirely — never inspected or written.
# overwrite = false  Only fill the tag when it is absent (default).
#             true   Rewrite even if already present; set back to false once done.

[tags.bpm]
enabled   = true
overwrite = false

[tags.mood_happy]
enabled   = true
overwrite = false
```

### Essentia models

The mood, timbre, tonality, and other non-BPM tags require Essentia `.pb` model files. BPM uses DeepRhythm and does not need them.

Download models:

```bash
musictagger-download-models
# or to a custom path:
musictagger-download-models --models-dir /path/to/models
```

Set the download policy in config:

```toml
[worker]
download_models = "ask"    # prompt at startup (default)
download_models = "always" # always download if missing
download_models = "never"  # never download; skip those tags
```

If models are missing the worker skips the affected tags gracefully — they stay queued and are retried once models are available.

### Acoustic fingerprinting

musictagger can read and write [Acoustid](https://acoustid.org/) Chromaprint fingerprints. If a file already has an `acoustid_fingerprint` tag (written by [Picard](https://picard.musicbrainz.org/) or the acoustag tool), the inspector reads it and stores a hash of it in the cache. The worker then uses this hash to look up pre-computed EffNet embeddings in `embeddings.db`, skipping the expensive audio decode and model forward pass.

If a file has no fingerprint tag and `fpcalc` is installed, the worker computes one and writes it to the file on the same pass. `fpcalc` is part of the [Chromaprint](https://acoustid.org/chromaprint) package:

```bash
# Debian / Ubuntu
apt install libchromaprint-tools

# macOS
brew install chromaprint
```

The embeddings database (`embeddings.db`) is safe to delete at any time — the worker will repopulate it on the next pass.

### Supported audio formats

`.mp3` `.flac` `.ogg` `.m4a` `.aac` `.wav` `.aiff` `.aif` `.wv` `.ape` `.opus` `.mpc` `.wma` `.alac`

### Tags written

| Tag | Description |
|---|---|
| `bpm` | Tempo in beats per minute |
| `mood` | Top genre/style labels (e.g. `Synth-Pop; House; Freestyle`) |
| `mood_happy` | Happy mood score (0–100) |
| `mood_sad` | Sad mood score (0–100) |
| `mood_relaxed` | Relaxed mood score (0–100) |
| `mood_aggressive` | Aggressive mood score (0–100) |
| `mood_party` | Party mood score (0–100) |
| `mood_dance` | Danceability score (0–100) |
| `electronic` | Electronic character score (0–100) |
| `acoustic` | Acoustic character score (0–100) |
| `instrumental` | Instrumental score (0–100) |
| `timbre_brightness` | Timbral brightness score (0–100) |
| `tonality` | Tonal/atonal score (0–100) |
| `key` | Musical key and scale (e.g. `C major`) |
| `acoustid_fingerprint` | Chromaprint fingerprint (written when absent and `fpcalc` is available) |

The `mood` tag is derived from the Genre Discogs400 classifier. Labels whose sigmoid confidence meets `mood_threshold` (default `0.15`) are included, up to `mood_max_results` (default `4`). Subgenres that appear under two parent genres in the Discogs taxonomy are disambiguated via an explicit mapping (e.g. `Electronic---Hardcore` → `Hardcore Techno`, `Rock---Hardcore` → `Hardcore`).

Silence detection: files with RMS energy below −80 dBFS are detected before any inference runs and recorded as a clean error rather than causing a crash.

`musictagger/tags.py` is the single source of truth for supported tags.

---

## Development

### Current scope

Implemented:
- SQLite-backed cache with auto-migrating `has_*` columns
- throttled scanner for large libraries
- inspector driven by the tag registry in `musictagger/tags.py`
- worker that writes BPM, Essentia-backed tags, and Acoustid fingerprints into files
- content-addressed embeddings cache keyed by Chromaprint fingerprint hash (`embeddings.db`)
- mood label disambiguation map for the 16 subgenres shared across parent genres
- orphan cleanup for deleted or moved files
- live Textual TUI
- startup handling for missing Essentia model files

### Lint, format, and test

```bash
# lint
uv run ruff check musictagger tests

# format
uv run ruff format musictagger tests

# tests
uv run pytest tests
```

Test suite notes:
- most tests are fast unit/integration tests using temporary files and monkeypatching
- `tests/test_music_playground.py` runs against the local `music/` folder if it exists
- the current tests do not require Essentia model downloads or `fpcalc`

### Project layout

- `musictagger/__main__.py` — startup, CLI parsing, model-download policy
- `musictagger/config.py` — config loading and default config generation
- `musictagger/tags.py` — tag registry and tag presence checks
- `musictagger/cache.py` — SQLite cache and queue/state queries
- `musictagger/scanner.py` — stat-only library scanner
- `musictagger/inspector.py` — mutagen-based tag inspection and fingerprint hash extraction
- `musictagger/worker.py` — BPM, Essentia inference, fingerprinting, and tag writing
- `musictagger/embeddings.py` — content-addressed EffNet embedding cache
- `musictagger/mood_mappings.py` — Genre Discogs400 label disambiguation map
- `musictagger/cleanup.py` — stale cache entry removal
- `musictagger/tui.py` — Textual application
