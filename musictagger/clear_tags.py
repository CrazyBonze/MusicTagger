"""Clear all metadata tags from audio files.

Usage:
    uv run musictagger-clear-tags /path/to/file.mp3
    uv run musictagger-clear-tags /path/to/directory
    uv run musictagger-clear-tags --recursive /path/to/directory

Behavior:
  - File path: clears tags from that one file.
  - Directory path: clears tags from audio files directly inside the directory.
    It is intentionally non-recursive unless --recursive is given.
  - --recursive: descends into all subdirectories.  The target must be a
    subdirectory of the current working directory (not cwd itself and not a
    parent of cwd) to prevent accidental wide sweeps.  A confirmation prompt
    listing the affected file count is shown before any changes are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mutagen


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


def _iter_targets(path: Path, *, recursive: bool) -> list[Path]:
    """Return audio files to process under *path*.

    When *recursive* is False only direct children are included.
    """
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise SystemExit(f"Path not found: {path}")

    if recursive:
        return sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.suffix.lower() in _AUDIO_EXTENSIONS
        )

    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in _AUDIO_EXTENSIONS
    )


def _assert_safe_recursive_target(target: Path) -> None:
    """Raise SystemExit if *target* is not a strict subdirectory of cwd.

    This prevents ``--recursive`` from being aimed at cwd itself, a parent
    directory, or an entirely unrelated path — all of which could wipe tags
    from a far larger set of files than intended.
    """
    cwd = Path.cwd().resolve()
    resolved = target.resolve()

    # resolved must be strictly inside cwd (not equal to it)
    try:
        resolved.relative_to(cwd)
    except ValueError:
        raise SystemExit(
            f"Safety check failed: {resolved} is not inside the current working "
            f"directory ({cwd}).\n"
            "Run the command from a parent of the target directory."
        )

    if resolved == cwd:
        raise SystemExit(
            f"Safety check failed: target is the current working directory ({cwd}).\n"
            "Pass a subdirectory, not '.' or an equivalent path."
        )


def _clear_tags(path: Path) -> tuple[bool, str]:
    """Clear tags from *path*.

    Returns ``(changed, message)``.
    """
    try:
        f = mutagen.File(str(path), easy=False)
    except Exception as exc:
        return False, f"error opening file: {exc}"

    if f is None:
        return False, "unsupported or unrecognised file"

    if f.tags is None:
        return False, "no tags present"

    try:
        f.delete()
    except Exception as exc:
        return False, f"error clearing tags: {exc}"

    return True, "tags cleared"


def main() -> None:
    """Entry point for the `musictagger-clear-tags` console script."""
    parser = argparse.ArgumentParser(
        prog="musictagger-clear-tags",
        description="Clear all metadata tags from one file or a directory of files.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Audio file or directory to process",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Descend into subdirectories. The target must be a subdirectory of the "
            "current working directory. A confirmation prompt is shown first."
        ),
    )
    args = parser.parse_args()

    target_path = args.path.expanduser()

    if args.recursive and not target_path.is_file():
        _assert_safe_recursive_target(target_path)

    files = _iter_targets(target_path, recursive=args.recursive)

    if not files:
        print(f"No supported audio files found in: {target_path}")
        return

    if args.recursive and not target_path.is_file():
        # Count how many directories are spanned so the prompt is informative.
        dirs = {f.parent for f in files}
        print(
            f"About to clear tags from {len(files)} file(s) across "
            f"{len(dirs)} director{'ies' if len(dirs) != 1 else 'y'} under:"
        )
        print(f"  {target_path.resolve()}")
        answer = input("Continue? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    changed = 0
    skipped = 0
    total = len(files)

    for index, path in enumerate(files, 1):
        did_change, message = _clear_tags(path)
        if did_change:
            changed += 1
            print(f"[{index}/{total}] Cleared: {path}")
        else:
            skipped += 1
            print(f"[{index}/{total}] Skipped: {path} ({message})")

    print(f"\nDone. Files scanned: {total}, cleared: {changed}, skipped: {skipped}")


if __name__ == "__main__":
    main()
