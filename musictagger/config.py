"""Configuration for the music tagger pipeline.

Settings are resolved in this priority order (highest wins):
  1. CLI arguments  (music_path, --config)
  2. Config file    (~/.config/musictagger/config.toml, or --config path)
  3. Built-in defaults

The config file is TOML.  If it does not exist at startup, a default file is
written there so the user has something to edit.

Example config.toml:
  music_path = "/mnt/music"

  [database]
  path = "~/.local/share/musictagger/cache.db"

  [logging]
  path            = "~/.local/share/musictagger/musictagger.log"
  level           = "INFO"
  rotation        = "10 MB"
  retention       = "30 days"

  [scanner]
  cron = "*/60 * * * *"
  file_throttle_ms  = 10
  dir_throttle_ms   = 200

  [inspector]
  throttle_ms = 50
  batch_size  = 100

  [worker]
  batch_size = 20
  # Directory containing Essentia .pb model files.
  # Run `musictagger-download-models` to populate it.
  models_dir = "~/.local/share/musictagger/models"

  [cleanup]
  cron = "0 * * * *"

  # Per-tag control.  enabled = false skips the tag entirely (file opens are
  # skipped for that tag; existing values are untouched).  overwrite = true
  # forces re-computation even when the tag is already present in the file —
  # on startup the cache rows for that tag are reset so the worker rewrites them.
  # Defaults: enabled = true, overwrite = false.
  [tags.bpm]
  enabled   = true
  overwrite = false

  [tags.mood_happy]
  enabled   = false
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# croniter is used to evaluate cron schedule expressions.
from croniter import CroniterBadCronError, croniter

# ── Per-tag config ─────────────────────────────────────────────────────────────


@dataclass
class TagConfig:
    """Runtime settings for a single tag.

    enabled   — when False the tag is ignored by the inspector and worker;
                existing values in audio files are never touched.
    overwrite — when True the inspector treats the tag as absent even when
                the audio file already carries it, forcing the worker to
                rewrite the tag.  Overwrite only fires when a file is
                (re-)inspected — i.e. when the scanner detects an mtime/size
                change or the cache is cleared.  Files whose cache row is
                already 'done' are not re-inspected and are therefore never
                rewritten unnecessarily.  Set back to false once the rewrite
                pass is complete.
    """

    enabled: bool = True
    overwrite: bool = False


# Default location for the config file — follows XDG Base Directory spec.
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "musictagger" / "config.toml"

# Written on first run if no config file exists.
_DEFAULT_TOML = """\
# musictagger configuration
# All values shown are the built-in defaults.

# Path to your music library (required — also settable as a CLI argument).
# music_path = "/mnt/music"

[database]
# Where to store the SQLite cache.
path = "{db_path}"
# Where to store the content-addressed embeddings cache.
# Safe to delete — the worker will repopulate it on the next pass.
embeddings_path = "{embeddings_db_path}"

[logging]
# Where to write the log file.
path = "{log_path}"
# Minimum level written to the log file: DEBUG, INFO, WARNING, ERROR
level = "INFO"
# Rotate the log file when it reaches this size.
rotation = "10 MB"
# Delete old rotated logs after this period.
retention = "30 days"

[scanner]
# When to run the full library walk (cron expression).
# Default: every hour on the hour.
cron = "0 * * * *"
# Sleep between individual files during the walk (ms).
# Raise this if the scanner hammers an NFS mount.
file_throttle_ms = 10
# Extra sleep between directories (ms).
dir_throttle_ms = 200

[inspector]
# Sleep between file opens (ms).  Mutagen is heavier than stat().
throttle_ms = 50
# Files to inspect per pass before yielding back to the scheduler.
batch_size = 100

[worker]
# Files to process per pass.
batch_size = 20
# Directory containing Essentia .pb model files.
# Run `musictagger-download-models` to populate it.
models_dir = "{models_dir}"
# What to do at startup when models are missing: ask, always, never.
download_models = "ask"
# DeepRhythm softmax confidence threshold below which both TempoCNN and
# DeepRhythm results are compared and the most likely one is used.
# Tune by logging confidence scores across your library.
bpm_confidence_threshold = 0.10
# Genre Discogs400 Sigmoid threshold for mood label selection.
# Labels at or above this score are included, up to mood_max_results.
# If fewer than mood_min_results pass the threshold, the top mood_min_results
# are used regardless of score.
mood_threshold = 0.15
mood_min_results = 1
mood_max_results = 4

[cleanup]
# When to remove cache rows for deleted files (cron expression).
# Default: once a day at midnight.
cron = "0 0 * * *"

# ── Per-tag settings ───────────────────────────────────────────────────────────
# Each [tags.<name>] section controls one tag independently.
#
#   enabled   = true   Include this tag in the pipeline (default).
#               false  Skip entirely — never inspected or written.
#                      The has_* cache column stays NULL so re-enabling later
#                      triggers fresh inspection without a file-change scan.
#
#   overwrite = false  Only fill the tag when it is absent (default).
#               true   Rewrite the tag even if already present in the file.
#                      Fires whenever the file is (re-)inspected: on cache
#                      clear or when the scanner detects a file change.
#                      Unmodified files already marked 'done' are skipped.
#                      Set back to false once the rewrite pass is done.
#
{tag_sections}"""


@dataclass
class Config:
    """All runtime settings for the pipeline."""

    # Path to the music library (NFS mount or local)
    music_path: Path

    # Where to store the SQLite cache database
    db_path: Path = field(
        default_factory=lambda: Path.home() / ".local/share/musictagger/cache.db"
    )

    # Where to store the content-addressed embeddings database.
    # Keyed by SHA-256 of the Acoustid fingerprint; safe to delete and rebuild.
    embeddings_db_path: Path = field(
        default_factory=lambda: Path.home() / ".local/share/musictagger/embeddings.db"
    )

    # Where to write the log file
    log_path: Path = field(
        default_factory=lambda: Path.home() / ".local/share/musictagger/musictagger.log"
    )

    # Minimum log level written to file
    log_level: str = "INFO"

    # loguru rotation / retention strings
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    # Cron expression controlling when the scanner does a full library walk
    scan_cron: str = "0 * * * *"

    # Sleep between individual files during the walk (ms)
    file_throttle_ms: int = 10

    # Extra sleep between directories (ms)
    dir_throttle_ms: int = 200

    # Sleep between inspector file opens (ms)
    inspector_throttle_ms: int = 50

    # How many files to inspect per pass before yielding
    inspector_batch_size: int = 100

    # How many files the worker processes per pass
    worker_batch_size: int = 20

    # Directory containing Essentia .pb model files
    models_dir: Path = field(
        default_factory=lambda: Path.home() / ".local/share/musictagger/models"
    )

    # Startup policy when Essentia models are missing: ask, always, never
    model_download_policy: str = "ask"

    # Cron expression controlling when orphan cleanup runs
    cleanup_cron: str = "0 0 * * *"

    # DeepRhythm softmax confidence threshold below which TempoCNN is used
    # instead.  Tune by logging confidence scores across your library.
    bpm_confidence_threshold: float = 0.10

    # Genre Discogs400 Sigmoid score threshold for mood label selection.
    # Only labels at or above this threshold are included.  If fewer than
    # mood_min_results pass, the top mood_min_results are taken regardless.
    # Results are always capped at mood_max_results.
    mood_threshold: float = 0.15
    mood_min_results: int = 1
    mood_max_results: int = 4

    # Per-tag enabled/overwrite settings.  Keys are tag names from tags.TAGS.
    # Tags absent from this dict use TagConfig defaults (enabled=True, overwrite=False).
    tag_configs: dict[str, TagConfig] = field(default_factory=dict)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def tag_cfg(self, tag_name: str) -> TagConfig:
        """Return the TagConfig for *tag_name*, falling back to defaults."""
        return self.tag_configs.get(tag_name, TagConfig())

    # ── Loaders ────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        music_path: str | None = None,
        config_path: Path | None = None,
    ) -> Config:
        """Build a Config from the file + optional CLI overrides.

        Priority: CLI args > config file > built-in defaults.
        """
        cfg_path = config_path or DEFAULT_CONFIG_PATH
        toml = _load_toml(cfg_path)

        # Resolve music_path: CLI arg beats config file entry.
        raw_music = music_path or toml.get("music_path")
        if not raw_music:
            raise SystemExit(
                "No music path set.\n"
                f"Pass it as a CLI argument or set music_path in {cfg_path}"
            )

        db_default = Path.home() / ".local/share/musictagger/cache.db"
        embeddings_default = Path.home() / ".local/share/musictagger/embeddings.db"
        log_default = Path.home() / ".local/share/musictagger/musictagger.log"

        db_section = toml.get("database", {})
        logging = toml.get("logging", {})
        scanner = toml.get("scanner", {})
        inspector = toml.get("inspector", {})
        worker = toml.get("worker", {})
        cleanup = toml.get("cleanup", {})

        # Build per-tag configs from [tags.<name>] sections.
        raw_tags: dict[str, dict] = toml.get("tags", {})
        tag_configs: dict[str, TagConfig] = {
            name: TagConfig(
                enabled=bool(section.get("enabled", True)),
                overwrite=bool(section.get("overwrite", False)),
            )
            for name, section in raw_tags.items()
        }

        models_default = Path.home() / ".local/share/musictagger/models"

        return cls(
            music_path=Path(raw_music).expanduser(),
            db_path=Path(db_section.get("path", db_default)).expanduser(),
            embeddings_db_path=Path(
                db_section.get("embeddings_path", embeddings_default)
            ).expanduser(),
            log_path=Path(logging.get("path", log_default)).expanduser(),
            log_level=str(logging.get("level", "INFO")).upper(),
            log_rotation=str(logging.get("rotation", "10 MB")),
            log_retention=str(logging.get("retention", "30 days")),
            scan_cron=_validated_cron(
                scanner.get("cron", "0 * * * *"), "[scanner] cron"
            ),
            file_throttle_ms=int(scanner.get("file_throttle_ms", 10)),
            dir_throttle_ms=int(scanner.get("dir_throttle_ms", 200)),
            inspector_throttle_ms=int(inspector.get("throttle_ms", 50)),
            inspector_batch_size=int(inspector.get("batch_size", 100)),
            worker_batch_size=int(worker.get("batch_size", 20)),
            models_dir=Path(worker.get("models_dir", models_default)).expanduser(),
            model_download_policy=str(worker.get("download_models", "ask")).lower(),
            cleanup_cron=_validated_cron(
                cleanup.get("cron", "0 0 * * *"), "[cleanup] cron"
            ),
            bpm_confidence_threshold=float(
                worker.get("bpm_confidence_threshold", 0.10)
            ),
            mood_threshold=float(worker.get("mood_threshold", 0.10)),
            mood_min_results=int(worker.get("mood_min_results", 1)),
            mood_max_results=int(worker.get("mood_max_results", 4)),
            tag_configs=tag_configs,
        )


def _validated_cron(expr: str, label: str) -> str:
    """Return *expr* if it is a valid cron expression, otherwise raise SystemExit."""
    try:
        croniter(expr)
    except (CroniterBadCronError, KeyError, ValueError) as exc:
        raise SystemExit(
            f"Invalid cron expression for {label}: {expr!r}\n{exc}"
        ) from exc
    return expr


def _load_toml(path: Path) -> dict:
    """Read the TOML config file, writing a default one if absent."""
    if not path.exists():
        _write_default(path)
        return {}

    with path.open("rb") as fh:
        return tomllib.load(fh)


def _write_default(path: Path) -> None:
    """Write a commented default config so the user has something to edit."""
    from musictagger.tags import TAGS

    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path.home() / ".local/share/musictagger"

    tag_sections = "\n".join(
        f"[tags.{tag.name}]\n# {tag.description}\nenabled   = true\noverwrite = false\n"
        for tag in TAGS
    )

    path.write_text(
        _DEFAULT_TOML.format(
            db_path=data_dir / "cache.db",
            embeddings_db_path=data_dir / "embeddings.db",
            log_path=data_dir / "musictagger.log",
            models_dir=data_dir / "models",
            tag_sections=tag_sections,
        )
    )
