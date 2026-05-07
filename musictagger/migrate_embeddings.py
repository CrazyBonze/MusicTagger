"""Re-encode the embeddings database from raw_f32 to zlib_f16 format.

This one-shot migration script rewrites every legacy ``raw_f32`` row in
``embeddings.db`` to the more compact ``zlib_f16`` encoding (float16 +
zlib compression), which typically reduces the database size by ~6-8x.

New rows written by musictagger after this migration are already stored in
``zlib_f16`` format, so you only need to run this script once to convert the
existing rows that were stored before the compression feature was added.

Usage:
    uv run musictagger-migrate-embeddings [--db PATH]
    python -m musictagger.migrate_embeddings [--db PATH]

The default database path is ~/.local/share/musictagger/embeddings.db (or
wherever musictagger has been configured to store it).

The migration is safe to interrupt and resume — completed rows are committed
in small batches, and the script will skip any rows that have already been
converted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

_DEFAULT_DB = Path.home() / ".local/share/musictagger/embeddings.db"


def _file_size_str(path: Path) -> str:
    """Return a human-readable file size string, e.g. '22.3 GB'."""
    try:
        size = path.stat().st_size
    except OSError:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return str(size)  # unreachable


def _progress(done: int, total: int) -> None:
    pct = done / total * 100 if total else 100
    bar_width = 40
    filled = int(bar_width * done / total) if total else bar_width
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r  [{bar}] {done}/{total} ({pct:.1f}%)", end="", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``musictagger-migrate-embeddings`` console script."""
    parser = argparse.ArgumentParser(
        description="Re-encode embeddings.db rows from raw_f32 to zlib_f16 (~6-8x smaller)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        metavar="PATH",
        help=f"Path to embeddings.db (default: {_DEFAULT_DB})",
    )
    args = parser.parse_args()

    db_path: Path = args.db.expanduser()

    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Deferred import — EmbeddingCache pulls in numpy; keep startup fast for
    # the help message path above.
    from musictagger.embeddings import EmbeddingCache

    size_before = _file_size_str(db_path)
    print(f"Database : {db_path}")
    print(f"Size before: {size_before}")

    cache = EmbeddingCache(db_path)
    try:
        stats_before = cache.stats()
        total_rows = stats_before["total_embeddings"]
        legacy_rows = stats_before["legacy_rows"]

        if legacy_rows == 0:
            print(
                f"Nothing to do — all {total_rows} rows are already in zlib_f16 format."
            )
            return

        print(
            f"Rows to migrate: {legacy_rows} of {total_rows} "
            f"({legacy_rows / total_rows * 100:.1f}%)"
        )
        print("Migrating", end="", flush=True)

        t0 = time.monotonic()
        done, _total = cache.migrate_encoding(progress_cb=_progress)
        elapsed = time.monotonic() - t0

        print()  # newline after progress bar
    finally:
        cache.close()

    size_after = _file_size_str(db_path)
    print(f"Size after : {size_after}")
    print(f"Migrated   : {done} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
