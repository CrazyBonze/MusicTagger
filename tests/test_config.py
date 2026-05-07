"""Unit tests for configuration loading and default file creation."""

from __future__ import annotations

from pathlib import Path

import pytest

import musictagger.config as config_module


def test_load_uses_config_values_when_no_cli_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'music_path = "~/library"',
                "[database]",
                'path = "~/db.sqlite"',
                "[logging]",
                'path = "~/tagger.log"',
                'level = "debug"',
                'rotation = "5 MB"',
                'retention = "10 days"',
                "[scanner]",
                'cron = "*/15 * * * *"',
                "file_throttle_ms = 1",
                "dir_throttle_ms = 2",
                "[inspector]",
                "throttle_ms = 3",
                "batch_size = 4",
                "[worker]",
                "batch_size = 5",
                'models_dir = "~/models"',
                "[cleanup]",
                'cron = "0 */6 * * *"',
            ]
        )
    )

    config = config_module.Config.load(config_path=config_path)

    assert config.music_path == Path("~/library").expanduser()
    assert config.db_path == Path("~/db.sqlite").expanduser()
    assert config.log_path == Path("~/tagger.log").expanduser()
    assert config.log_level == "DEBUG"
    assert config.log_rotation == "5 MB"
    assert config.log_retention == "10 days"
    assert config.scan_cron == "*/15 * * * *"
    assert config.file_throttle_ms == 1
    assert config.dir_throttle_ms == 2
    assert config.inspector_throttle_ms == 3
    assert config.inspector_batch_size == 4
    assert config.worker_batch_size == 5
    assert config.models_dir == Path("~/models").expanduser()
    assert config.model_download_policy == "ask"
    assert config.cleanup_cron == "0 */6 * * *"


def test_load_cli_music_path_overrides_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('music_path = "~/from-config"\n')

    config = config_module.Config.load(
        music_path=str(tmp_path / "from-cli"),
        config_path=config_path,
    )

    assert config.music_path == tmp_path / "from-cli"


def test_load_writes_default_config_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setattr(config_module.Path, "home", lambda: fake_home)
    config_path = tmp_path / "missing-config.toml"

    config = config_module.Config.load(
        music_path=str(tmp_path / "music"),
        config_path=config_path,
    )

    assert config.music_path == tmp_path / "music"
    assert config_path.exists()
    content = config_path.read_text()
    assert str(fake_home / ".local/share/musictagger/cache.db") in content
    assert str(fake_home / ".local/share/musictagger/models") in content


def test_load_raises_when_music_path_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[logging]\nlevel = "INFO"\n')

    with pytest.raises(SystemExit, match="No music path set"):
        config_module.Config.load(config_path=config_path)


def test_default_config_includes_all_tag_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generated config file must contain a [tags.<name>] section for every tag."""
    from musictagger.tags import TAGS

    fake_home = tmp_path / "home"
    monkeypatch.setattr(config_module.Path, "home", lambda: fake_home)
    config_path = tmp_path / "config.toml"

    config_module.Config.load(
        music_path=str(tmp_path / "music"),
        config_path=config_path,
    )

    content = config_path.read_text()
    for tag in TAGS:
        assert f"[tags.{tag.name}]" in content, f"missing section for tag {tag.name!r}"
        assert f"# {tag.description}" in content
