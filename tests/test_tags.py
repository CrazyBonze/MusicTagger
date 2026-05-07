"""Unit tests for tag helper functions and registry shape."""

from __future__ import annotations

import pytest

from musictagger.tags import (
    TAGS,
    _check_bpm,
    _check_electronic,
    _check_key,
    _check_keys,
    _check_mood,
    _check_mood_dance,
    _check_mood_happy,
    _check_timbre_brightness,
    _check_tonality,
)


def test_check_keys_requires_a_present_truthy_value() -> None:
    assert _check_keys({}, "TBPM") is False
    assert _check_keys({"TBPM": []}, "TBPM") is False
    assert _check_keys({"TBPM": ["120"]}, "TBPM") is True


@pytest.mark.parametrize(
    ("check_fn", "key"),
    [
        (_check_bpm, "TBPM"),
        (_check_bpm, "bpm"),
        (_check_bpm, "tmpo"),
        (_check_mood_happy, "TXXX:MOOD_HAPPY"),
        (_check_mood_happy, "----:com.apple.iTunes:MOOD_HAPPY"),
        (_check_mood_dance, "MOOD_DANCEABILITY"),
        (_check_mood_dance, "----:com.apple.iTunes:MOOD_DANCE"),
        (_check_mood, "TMOO"),
        (_check_mood, "----:com.apple.iTunes:MOOD"),
        (_check_electronic, "TXXX:ELECTRONIC"),
        (_check_electronic, "----:com.apple.iTunes:MOOD_ELECTRONIC"),
        (_check_timbre_brightness, "TIMBRE_BRIGHTNESS"),
        (_check_timbre_brightness, "TXXX:TIMBRE"),
        (_check_tonality, "tonality"),
        # Musical key aliases
        (_check_key, "TKEY"),
        (_check_key, "INITIALKEY"),
        (_check_key, "initialkey"),
        (_check_key, "key"),
        (_check_key, "----:com.apple.iTunes:initialkey"),
        (_check_key, "WM/InitialKey"),
    ],
)
def test_check_functions_accept_supported_aliases(check_fn: object, key: str) -> None:
    assert check_fn({key: ["value"]}) is True


def test_check_key_returns_false_when_absent() -> None:
    assert _check_key({}) is False
    assert _check_key({"TBPM": ["120"]}) is False


def test_tags_registry_has_unique_names_and_nonempty_descriptions() -> None:
    assert len(TAGS) == len({tag.name for tag in TAGS})
    assert all(tag.description for tag in TAGS)
    assert all(callable(tag.check_fn) for tag in TAGS)


def test_tags_registry_includes_key() -> None:
    names = [tag.name for tag in TAGS]
    assert "key" in names
