# AGENTS.md - musictagger

Reference for human contributors and agentic coding tools working in this repository.

## Project Overview
`musictagger` is a Textual terminal UI that scans a music library, inspects audio tags with mutagen, and fills missing analysis tags such as BPM, mood, timbre, tonality, key, and more.

Pipeline stages:
- scanner: `stat()` only, never opens audio files
- inspector: opens files with mutagen and records which tags exist
- worker: computes and writes missing tags
- cleanup: removes stale cache rows for deleted or moved files
- tui: orchestrates those jobs and shows live status

Important modules:
- `musictagger/__main__.py`: CLI entrypoint and startup flow
- `musictagger/config.py`: config dataclass and TOML loading
- `musictagger/tags.py`: tag registry and tag presence checks
- `musictagger/cache.py`: SQLite cache, schema migration, work queues
- `musictagger/scanner.py`: throttled filesystem walk
- `musictagger/inspector.py`: mutagen-based inspection
- `musictagger/worker.py`: BPM and Essentia-backed inference plus tag writing
- `musictagger/cleanup.py`: orphaned cache entry cleanup
- `musictagger/tui.py`: Textual app and orchestration loop
- `musictagger/logging.py`: loguru setup and stdlib logging interception
- `musictagger/clear_tags.py`: `musictagger-clear-tags` console script
- `musictagger/download_models.py`: `musictagger-download-models` console script
- `musictagger/inspect_tags.py`: `musictagger-inspect-tags` console script

## Environment And Tooling
- Python is pinned to `3.12` in `.python-version`
- dependencies are managed with `uv`
- build backend is `hatchling`
- dev dependencies include `pytest` and `ruff`
- entrypoints live in `pyproject.toml`

Bootstrap:

```bash
uv sync
```

## Build, Run, Lint, And Test
```bash
# run the app
uv run musictagger /path/to/music
python -m musictagger /path/to/music

# helper scripts
uv run musictagger-download-models [--models-dir PATH] [--force]
uv run musictagger-clear-tags /path/to/file-or-directory [--recursive]
uv run musictagger-inspect-tags /path/to/directory        # outputs CSV to stdout

# build
uv build

# lint and format
uv run ruff check musictagger tests
uv run ruff check --fix musictagger tests
uv run ruff format musictagger tests
uv run ruff format --check musictagger tests

# tests
uv run pytest tests
uv run pytest tests/test_cache.py
uv run pytest tests/test_cache.py::test_mark_changed_adds_file_to_inspection_queue
uv run pytest -x
uv run pytest -k "cache"
uv run pytest tests/test_main.py -q
```

Testing notes:
- most tests are fast and use `tmp_path` plus monkeypatching
- `tests/test_music_playground.py` auto-skips when a local `music/` directory is absent
- the normal suite does not require downloaded Essentia models
- there is no `conftest.py`; fixtures are defined locally in each test file

## Additional Instruction Files
Checked at the repo root:
- `.cursor/rules/`: not present
- `.cursorrules`: not present
- `.github/copilot-instructions.md`: not present

If any of those files are added later, treat them as part of the repository instructions.

## Architecture Conventions
- `musictagger/tags.py` is the single source of truth for supported tags
- adding a `TagDef` automatically affects cache schema, inspection, work queues, and TUI stats
- the scanner must remain stat-only and should not open audio files
- the inspector is the only stage that checks whether tags already exist
- the worker should skip unavailable model-backed work gracefully instead of crashing the app
- long-running jobs are stoppable via a `_running` flag checked inside loops
- the TUI launches background jobs with `run_worker(..., thread=True, group="<stage>", exclusive=True)`

## Code Style Guidelines

### Imports
- start modules with a module docstring
- put `from __future__ import annotations` immediately after the module docstring
- group imports as standard library, third-party, then local package imports
- separate groups with one blank line and avoid star imports
- use deferred (in-function) imports for heavyweight optional dependencies such as `essentia`, `deeprhythm`, and `torch` to keep startup fast; document this intent with a comment

### Formatting
- Ruff uses default settings; there is no custom `[tool.ruff]` config
- line length therefore defaults to `88`
- keep changes focused and do not reformat unrelated code
- larger modules use Unicode section banner comments; match and preserve nearby style:
  `# ── Title ──────────────────────────────────────────────────────────────────────`

### Types
- annotate function parameters and return types consistently
- annotate `-> None` explicitly on every void function, including `__init__` and lifecycle methods
- use built-in generics: `list[str]`, `dict[str, bool]`, `tuple[int, int]`
- use `X | Y` unions; never use `Optional[X]` or `Union[X, Y]`
- dataclass fields are annotated inline; use `@dataclass(frozen=True)` for immutable value objects
- when a type cannot be named at module level due to deferred imports, annotate as `object | None`

### Naming
- classes use `PascalCase`
- functions, methods, and variables use `snake_case`
- private helpers and attributes use a single leading underscore
- public module constants use `SCREAMING_SNAKE_CASE`; private module constants use `_SCREAMING_SNAKE_CASE`
- tag column names are derived as `has_{tag.name}` via the `_col()` helper in `cache.py`; do not hardcode scattered variants

### Docstrings And Comments
- every module has a descriptive docstring explaining its purpose or design
- public classes and important public methods have short docstrings
- inline comments address file-format quirks, threading rationale, and pipeline design decisions
- avoid comments that only restate obvious code

### Logging
- the project uses `loguru`, imported as `from loguru import logger`
- `musictagger/logging.py` routes stdlib logging into loguru
- in logger calls always use loguru brace formatting: `logger.info("Scan started: {}", path)`
- do not use f-strings in logger calls
- do not print to stdout or stderr from TUI code; it corrupts the terminal UI
- standalone scripts (`clear_tags.py`, `download_models.py`, `inspect_tags.py`) may use `print()` since they run outside the TUI
- in `__main__.py`, `logger` is imported inside functions that run after `setup_logging()` to avoid early sink access; this is an intentional exception to the module-level import convention

### Error Handling
- outer background job loops catch broad `Exception`, log it, and keep the app alive
- filesystem probes catch `OSError` and return a safe fallback
- tag lookups catch `KeyError` and `TypeError` and treat missing data as absent tags
- shutdown-sensitive cross-thread TUI calls catch `RuntimeError` and discard the update silently
- when swallowing an exception intentionally, always leave a comment explaining why

### Data And Persistence
- SQLite is used directly; there is no ORM
- external values must go through parameterized `?` queries; never interpolate values into SQL strings
- schema evolution happens in `FileCache._migrate()` via `ALTER TABLE ... ADD COLUMN`
- SQLite writes are protected by `threading.Lock()` and committed in batches via `flush()`
- the shared connection uses `check_same_thread=False` with WAL journal mode

### Textual Patterns
- layout is defined in `compose()` with `yield`
- app or widget CSS lives in `CSS` or `DEFAULT_CSS` class attributes
- keybindings live in `BINDINGS`
- reactive app state uses `reactive[...]`
- cross-thread UI updates use `call_from_thread` to post a `Message` subclass; never access widgets directly from background threads

### Testing Conventions
- tests use `pytest` with `tmp_path` for temporary files; no shared heavy fixtures
- monkeypatching is common for startup flow, mutagen, subprocess, and external integrations
- tests that need to verify DB state use `cache._conn.execute(...)` directly
- wrap manually created `FileCache` instances in `try/finally: cache.close()` rather than a `with` block
- when fixing a regression, add a narrow unit or integration test near the affected module and name it after the specific scenario (e.g., `test_inspector_does_not_loop_when_a_tag_is_disabled`)

## Practical Guidance For Agents
- make small, local changes unless the architecture clearly requires more
- preserve the split between scanner, inspector, worker, cleanup, and TUI orchestration
- do not add tag-specific logic outside `tags.py` unless the design truly requires it
- avoid introducing file-open or expensive work into scanner hot paths
- be careful with terminal output because the app owns the terminal during TUI runs
- if you touch `worker.py`, be mindful of optional model files, `ffmpeg`, and heavyweight imports
- if you touch tests, keep single-test invocation easy and mention any new prerequisites
