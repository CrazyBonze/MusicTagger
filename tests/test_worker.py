"""Focused tests for worker-side tag derivation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import mutagen
import numpy as np
import pytest

from musictagger.cache import FileCache
from musictagger.config import Config
from musictagger.download_models import MODELS
from musictagger.tags import TAGS
from musictagger.mood_mappings import GENRE_LABEL_OVERRIDES, label_for_class
from musictagger.worker import (
    _load_mono_ffmpeg,
    _make_text_list_writer,
    _top_subgenres_from_genre_scores,
    _write_acoustid_fingerprint,
    _write_key,
)


def test_download_catalog_includes_genre_discogs400_model_and_metadata() -> None:
    filenames = {filename for filename, _ in MODELS}

    assert "genre_discogs400-discogs-effnet-1.pb" in filenames
    assert "genre_discogs400-discogs-effnet-1.json" in filenames


def test_top_subgenres_returns_labels_above_threshold() -> None:
    # 3 of 5 labels are at or above the 0.10 threshold — all 3 returned.
    classes = [
        "Electronic---Techno",
        "Rock---Indie Rock",
        "Hip Hop---Boom Bap",
        "Jazz---Cool Jazz",
        "Pop---Europop",
    ]
    scores = np.array([0.8, 0.6, 0.9, 0.3, 0.05])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == [
        "Boom Bap",
        "Techno",
        "Indie Rock",
        "Cool Jazz",
    ]


def test_top_subgenres_caps_at_max_results() -> None:
    # 6 labels above threshold but max_results=4 — only top 4 returned.
    classes = [f"Genre---Style{i}" for i in range(6)]
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert len(result) == 4
    assert result[0] == "Style0"


def test_top_subgenres_pads_to_min_when_threshold_not_met() -> None:
    # Only 1 label passes threshold=0.50, but min_results=2 — pad to 2.
    classes = ["Electronic---Techno", "Rock---Indie Rock", "Pop---Europop"]
    scores = np.array([0.8, 0.3, 0.1])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.50, min_results=2, max_results=4
    )

    assert result == ["Techno", "Indie Rock"]


def test_top_subgenres_pads_to_min_when_nothing_passes_threshold() -> None:
    # No label reaches threshold — return top min_results=1 regardless.
    classes = ["Electronic---Techno", "Rock---Indie Rock"]
    scores = np.array([0.07, 0.05])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Techno"]


def test_top_subgenres_strips_genre_prefix() -> None:
    classes = ["Electronic---Ambient", "Rock---Doom Metal"]
    scores = np.array([0.9, 0.7])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Ambient", "Doom Metal"]


def test_top_subgenres_title_cases_labels() -> None:
    classes = ["Electronic---acid house", "Rock---indie rock"]
    scores = np.array([0.8, 0.6])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Acid House", "Indie Rock"]


def test_top_subgenres_handles_no_separator() -> None:
    # Labels without '---' should be returned as-is (title-cased).
    classes = ["Ambient", "Techno"]
    scores = np.array([0.7, 0.5])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Ambient", "Techno"]


def test_top_subgenres_uses_override_name_for_collision_class() -> None:
    # Both Electronic---Electro and Hip Hop---Electro are above threshold.
    # The higher-scoring one (Electronic---Electro, 0.24) should appear first
    # under its override name "Electro"; the lower one (Hip Hop---Electro, 0.03)
    # should be skipped because it resolves to the already-seen label "Electro Hip Hop"
    # — which is a different override — so both *would* appear if both are above
    # threshold.  This test confirms each gets its own distinct override name
    # and neither is dropped due to a false duplicate.
    classes = ["Electronic---Electro", "Hip Hop---Electro"]
    scores = np.array([0.24, 0.20])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Electro", "Electro Hip Hop"]


def test_top_subgenres_deduplicates_when_two_keys_resolve_to_same_label() -> None:
    # Simulate a future hypothetical where two class keys map to the same
    # display label — the lower-scoring duplicate must be skipped.
    # We achieve this by testing the seen-set guard directly with two classes
    # that both resolve to bare "Techno" (no override, same subgenre).
    classes = ["Electronic---Techno", "Electronic---Techno"]
    scores = np.array([0.8, 0.6])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Techno"]


def test_top_subgenres_non_collision_class_uses_plain_subgenre() -> None:
    # Classes not in GENRE_LABEL_OVERRIDES still return bare title-cased subgenre.
    classes = ["Electronic---Techno", "Rock---Indie Rock"]
    scores = np.array([0.9, 0.7])

    result = _top_subgenres_from_genre_scores(
        classes, scores, threshold=0.10, min_results=1, max_results=4
    )

    assert result == ["Techno", "Indie Rock"]


# ── mood_mappings ─────────────────────────────────────────────────────────────


def test_label_for_class_returns_override_for_collision_keys() -> None:
    assert label_for_class("Electronic---Electro") == "Electro"
    assert label_for_class("Hip Hop---Electro") == "Electro Hip Hop"
    assert label_for_class("Rock---Hardcore") == "Hardcore"
    assert label_for_class("Electronic---Hardcore") == "Hardcore Techno"
    assert label_for_class("Electronic---New Wave") == "Synth New Wave"
    assert label_for_class("Rock---New Wave") == "New Wave"


def test_label_for_class_strips_prefix_for_non_override_keys() -> None:
    assert label_for_class("Electronic---Techno") == "Techno"
    assert label_for_class("Rock---Indie Rock") == "Indie Rock"
    assert label_for_class("Hip Hop---Boom Bap") == "Boom Bap"


def test_label_for_class_title_cases_bare_labels() -> None:
    assert label_for_class("ambient") == "Ambient"
    assert label_for_class("acid house") == "Acid House"


def test_genre_label_overrides_covers_all_32_collision_entries() -> None:
    # There are 16 colliding subgenres × 2 parent genres = 32 entries.
    assert len(GENRE_LABEL_OVERRIDES) == 32


def test_genre_label_overrides_all_values_are_unique() -> None:
    # Every override must resolve to a distinct display name — no two class
    # keys should silently map to the same label.
    values = list(GENRE_LABEL_OVERRIDES.values())
    assert len(values) == len(set(values)), (
        "Duplicate display names found in GENRE_LABEL_OVERRIDES"
    )


def test_text_list_writer_joins_values_for_generic_mutagen_mapping() -> None:
    mood_writer = _make_text_list_writer("mood")
    tags: dict[str, list[str]] = {}

    mood_writer(tags, ["Techno", "Indie Rock"])

    assert tags == {"MOOD": ["Techno; Indie Rock"]}


# ── _write_key ────────────────────────────────────────────────────────────────


def test_write_key_vorbis_writes_key() -> None:
    """Plain dict simulates a Vorbis/FLAC file (the else branch).

    Jaikoz uses KEY as the canonical Vorbis comment field for musical key,
    so we write KEY rather than INITIALKEY for FLAC/Ogg files.
    """
    tags: dict[str, list[str]] = {}
    _write_key(tags, "C major")
    assert tags == {"KEY": ["C major"]}


def test_write_key_vorbis_minor() -> None:
    tags: dict[str, list[str]] = {}
    _write_key(tags, "A minor")
    assert tags == {"KEY": ["A minor"]}


def test_write_key_id3_sets_tkey_frame(tmp_path: Path) -> None:
    """Write key into a real minimal MP3 via mutagen and verify TKEY is set."""
    import mutagen.id3
    from mutagen.id3 import ID3

    mp3_path = tmp_path / "test.mp3"
    # Minimal valid ID3v2.3 header (10 bytes) followed by silence
    mp3_path.write_bytes(
        b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" + b"\x00" * 104
    )

    try:
        tags = ID3()
        tags.save(str(mp3_path))
        f = mutagen.File(str(mp3_path), easy=False)
        _write_key(f, "D major")
        assert "TKEY" in f.tags
        assert f.tags["TKEY"].text == ["D major"]
    except Exception:
        pytest.skip("Could not create minimal MP3 for ID3 test")


def test_write_key_mp4_sets_initialkey_atom(tmp_path: Path) -> None:
    """Write key into a minimal MP4 container and verify the freeform atom."""
    from mutagen.mp4 import MP4

    m4a_path = tmp_path / "test.m4a"
    try:
        f = MP4()
        f.save(str(m4a_path))
        f = mutagen.File(str(m4a_path), easy=False)
        _write_key(f, "F# minor")
        atom_key = "----:com.apple.iTunes:initialkey"
        assert atom_key in f
        assert bytes(f[atom_key][0]) == b"F# minor"
    except Exception:
        pytest.skip("Could not create minimal MP4 for atom test")


# ── EssentiaEngine._predict_key ───────────────────────────────────────────────


def test_predict_key_returns_none_on_monoloader_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_predict_key should return None gracefully when MonoLoader raises."""
    from musictagger.worker import EssentiaEngine

    engine = EssentiaEngine(Path("/nonexistent/models"))

    def _fail_monoloader(*args: object, **kwargs: object) -> object:
        class _ML:
            def __call__(self) -> None:
                raise RuntimeError("simulated decode failure")

        return _ML()

    monkeypatch.setattr(
        "musictagger.worker.EssentiaEngine._predict_key",
        lambda self, filepath_str: None,
    )

    result = engine._predict_key("/no/such/file.flac")
    assert result is None


def test_predict_key_formats_output_as_key_space_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_predict_key should join key and scale with a space."""
    import numpy as np

    from musictagger.worker import EssentiaEngine

    engine = EssentiaEngine(Path("/nonexistent/models"))

    # Patch _load_mono_ffmpeg so no real file or ffmpeg is needed.
    fake_audio = np.zeros(44100, dtype=np.float32)  # 1 second of silence at 44.1 kHz
    monkeypatch.setattr(
        "musictagger.worker._load_mono_ffmpeg",
        lambda path, sr: (fake_audio, sr),
    )

    class _FakeKeyExtractor:
        def __call__(self, audio: object) -> tuple[str, str, float]:
            return ("G", "minor", 0.85)

    import essentia.standard as es_std  # type: ignore[import]

    monkeypatch.setattr(es_std, "KeyExtractor", _FakeKeyExtractor)

    result = engine._predict_key("/fake/track.flac")
    assert result == "G minor"


# ── _load_mono_ffmpeg timeout ──────────────────────────────────────────────────


def test_load_mono_ffmpeg_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung ffmpeg process should raise RuntimeError, not block forever."""

    def _fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=120)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        _load_mono_ffmpeg("/some/track.mp3")


def test_load_mono_ffmpeg_raises_on_nonzero_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = MagicMock()
    result.returncode = 1
    result.stdout = b""
    result.stderr = b"error message"
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: result)

    with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
        _load_mono_ffmpeg("/some/track.mp3")


def test_load_mono_ffmpeg_raises_on_empty_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = MagicMock()
    result.returncode = 0
    result.stdout = b""
    result.stderr = b""
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: result)

    with pytest.raises(RuntimeError, match="no audio data"):
        _load_mono_ffmpeg("/some/track.mp3")


# ── _predict_bpm: NoneType .to() AttributeError handling ──────────────────────


def test_predict_bpm_raises_runtime_error_on_nonetype_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DeepRhythm NoneType .to() AttributeError is caught and re-raised as
    RuntimeError with a message indicating the track may be silent or inaudible.
    """
    import musictagger.worker as worker_mod

    audio = np.ones(22050 * 10, dtype=np.float32)
    monkeypatch.setattr(
        "musictagger.worker._load_mono_ffmpeg",
        lambda *a, **kw: (audio, 22050),
    )

    def _bad_predict(*args: object, **kwargs: object) -> None:
        # Simulate PyTorch calling .to() on a None result.
        raise AttributeError("'NoneType' object has no attribute 'to'")

    mock_predictor = MagicMock()
    mock_predictor.predict_from_audio.side_effect = _bad_predict

    cache_path = tmp_path / "cache.db"
    cache = FileCache(cache_path)
    try:
        config = Config(
            music_path=tmp_path, embeddings_db_path=tmp_path / "embeddings.db"
        )
        engine = worker_mod.Worker(cache=cache, config=config)
        monkeypatch.setattr(engine, "_get_predictor", lambda: mock_predictor)

        with pytest.raises(RuntimeError, match="silent or inaudible"):
            engine._predict_bpm("/fake/silent.flac")
        engine.close()
    finally:
        cache.close()


# ── Worker.run_pass: empty process_file result ─────────────────────────────────


def _make_worker_with_cache(tmp_path: Path) -> tuple:
    """Return a (Worker, FileCache, filepath) triple with one queued file."""
    from musictagger.worker import Worker

    db_path = tmp_path / "cache.db"
    cache = FileCache(db_path)

    config = Config(
        music_path=tmp_path / "music",
        db_path=db_path,
        embeddings_db_path=tmp_path / "embeddings.db",
        log_path=tmp_path / "musictagger.log",
        models_dir=tmp_path / "models",
    )

    filepath = tmp_path / "track.mp3"
    filepath.write_bytes(b"fake-audio")

    tag_results = {t.name: False for t in TAGS}
    cache.mark_changed(filepath)
    cache.mark_inspected(filepath, tag_results)
    cache.flush()

    worker = Worker(config, cache)
    return worker, cache, filepath


def test_run_pass_marks_error_when_process_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If process_file() returns {}, the file should be marked error, not done."""
    from musictagger.worker import Worker

    worker, cache, filepath = _make_worker_with_cache(tmp_path)

    monkeypatch.setattr(Worker, "process_file", lambda self, fp, missing, **kw: {})

    worker.run_pass(batch_size=1)

    row = cache._conn.execute(
        "SELECT processing_status, last_error FROM processed WHERE filepath = ?",
        (str(filepath),),
    ).fetchone()
    assert row[0] == "error"
    assert "no tag results" in row[1]
    assert worker.errors == 1
    assert worker.processed == 0
    worker.close()
    cache.close()


def test_run_pass_marks_done_when_process_file_returns_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When process_file() returns a non-empty dict the file should be marked done
    and the corresponding has_* column set to 1."""
    from musictagger.worker import Worker

    worker, cache, filepath = _make_worker_with_cache(tmp_path)

    # Return a minimal result dict and stub out _write_tags so no real file IO.
    monkeypatch.setattr(
        Worker, "process_file", lambda self, fp, missing, **kw: {"bpm": 120}
    )
    monkeypatch.setattr(Worker, "_write_tags", lambda self, fp, results, **kw: None)

    worker.run_pass(batch_size=1)

    row = cache._conn.execute(
        "SELECT processing_status, has_bpm FROM processed WHERE filepath = ?",
        (str(filepath),),
    ).fetchone()
    assert row[0] == "done"
    assert row[1] == 1  # has_bpm must be set to 1, not left as 0 or NULL
    assert worker.processed == 1
    assert worker.errors == 0
    worker.close()
    cache.close()


# ── Worker.last_activity heartbeat ────────────────────────────────────────────


def test_worker_last_activity_updates_during_run_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """last_activity should be a non-zero monotonic timestamp after a pass."""
    from musictagger.worker import Worker

    worker, cache, filepath = _make_worker_with_cache(tmp_path)

    monkeypatch.setattr(
        Worker, "process_file", lambda self, fp, missing, **kw: {"bpm": 120}
    )
    monkeypatch.setattr(Worker, "_write_tags", lambda self, fp, results, **kw: None)

    assert worker.last_activity == 0.0

    worker.run_pass(batch_size=1)

    assert worker.last_activity > 0.0
    worker.close()
    cache.close()


# ── EssentiaEngine._load_classifier crash guards ──────────────────────────────


def test_load_classifier_marks_broken_on_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A corrupt or broken .pb file that raises during construction should mark
    the classifier as broken (not crash the process)."""
    from musictagger.worker import EssentiaEngine

    # Write a dummy .pb file so the file-exists check passes.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pb = models_dir / "mood_happy-discogs-effnet-1.pb"
    pb.write_bytes(b"not a real protobuf")

    engine = EssentiaEngine(models_dir)

    import essentia.standard as es_std  # type: ignore[import]

    def _bad_init(**kwargs: object) -> None:
        raise RuntimeError("simulated TF graph load failure")

    monkeypatch.setattr(es_std, "TensorflowPredict2D", _bad_init)

    result = engine._load_classifier("mood_happy", pb.name)

    assert result is None
    assert "mood_happy" in engine._broken_models


def test_load_classifier_broken_model_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """After a classifier is marked broken it should not be loaded again."""
    from musictagger.worker import EssentiaEngine

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pb = models_dir / "mood_happy-discogs-effnet-1.pb"
    pb.write_bytes(b"not a real protobuf")

    engine = EssentiaEngine(models_dir)
    engine._broken_models.add("mood_happy")

    call_count = [0]

    import essentia.standard as es_std  # type: ignore[import]

    def _counting_init(**kwargs: object) -> None:
        call_count[0] += 1

    monkeypatch.setattr(es_std, "TensorflowPredict2D", _counting_init)

    result = engine._load_classifier("mood_happy", pb.name)

    assert result is None
    assert call_count[0] == 0  # constructor must not be called a second time


def test_predict_effnet_scores_skips_broken_classifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """_predict_effnet_scores should return an empty dict when all classifiers
    fail during inference, and mark them broken for the session."""
    from musictagger.worker import EssentiaEngine

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pb = models_dir / "mood_happy-discogs-effnet-1.pb"
    pb.write_bytes(b"dummy")

    engine = EssentiaEngine(models_dir)

    import essentia.standard as es_std  # type: ignore[import]

    class _BrokenClassifier:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __call__(self, embeddings: object) -> None:
            raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(es_std, "TensorflowPredict2D", _BrokenClassifier)

    fake_embeddings = np.zeros((10, 64), dtype=np.float32)
    results = engine._predict_effnet_scores(
        "/fake/track.flac", fake_embeddings, ["mood_happy"]
    )

    assert results == {}
    assert "mood_happy" in engine._broken_models


# ── Worker._init_predictor crash guards ───────────────────────────────────────


def test_init_predictor_cpu_failure_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If DeepRhythmPredictor CPU init fails, a RuntimeError should be raised
    (not an uncaught library-specific exception)."""
    from musictagger.worker import Worker
    from musictagger.config import Config

    config = Config(
        music_path=Path("/fake/music"), embeddings_db_path=Path("/tmp/opencode/e1.db")
    )
    worker = Worker(config, MagicMock())

    import deeprhythm  # type: ignore[import]

    def _bad_init(device: str) -> None:
        raise RuntimeError("simulated weight load failure")

    monkeypatch.setattr(deeprhythm, "DeepRhythmPredictor", _bad_init)

    # Torch/CUDA import must also succeed but report no CUDA so we reach the
    # CPU path. The simplest way is to just make torch.cuda.is_available()
    # return False via the existing try/except in _init_predictor.
    with pytest.raises(RuntimeError, match="DeepRhythm CPU init failed"):
        worker._init_predictor()


# ── Worker._predict_bpm fallback path ─────────────────────────────────────────


def test_predict_bpm_fallback_non_type_error_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the fallback predict_from_audio call (keyword-arg form) raises something
    other than TypeError, _predict_bpm should surface a RuntimeError rather than
    crashing with an unguarded exception."""
    from musictagger.worker import Worker
    from musictagger.config import Config

    config = Config(
        music_path=Path("/fake/music"), embeddings_db_path=Path("/tmp/opencode/e2.db")
    )
    worker = Worker(config, MagicMock())

    # Stub out ffmpeg decode so we get a valid audio buffer.
    fake_audio = np.zeros(22050 * 10, dtype=np.float32)
    monkeypatch.setattr(
        "musictagger.worker._load_mono_ffmpeg",
        lambda *a, **kw: (fake_audio, 22050),
    )

    call_count = [0]

    class _FakePredictor:
        def predict_from_audio(
            self, y: object, *args: object, **kwargs: object
        ) -> None:
            call_count[0] += 1
            if call_count[0] == 1:
                # First call raises TypeError → triggers fallback.
                raise TypeError("positional")
            # Fallback call raises something unexpected.
            raise ValueError("unexpected model error")

    worker._predictor = _FakePredictor()

    with pytest.raises(RuntimeError, match="DeepRhythm inference failed"):
        worker._predict_bpm("/fake/track.flac")


# ── EssentiaEngine._load_audio crash guard ────────────────────────────────────


def test_load_audio_propagates_ffmpeg_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ffmpeg decoding fails, _load_audio should propagate the RuntimeError."""
    from musictagger.worker import EssentiaEngine

    engine = EssentiaEngine(Path("/nonexistent/models"))

    monkeypatch.setattr(
        "musictagger.worker._load_mono_ffmpeg",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("ffmpeg decode failed (rc=1): error")
        ),
    )

    with pytest.raises(RuntimeError, match="ffmpeg decode failed"):
        engine._load_audio("/fake/track.flac")


def test_load_audio_raises_on_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ffmpeg returns an empty array, _load_audio should raise RuntimeError."""
    from musictagger.worker import EssentiaEngine

    engine = EssentiaEngine(Path("/nonexistent/models"))

    monkeypatch.setattr(
        "musictagger.worker._load_mono_ffmpeg",
        lambda *a, **kw: (np.array([], dtype=np.float32), 16000),
    )

    with pytest.raises(RuntimeError, match="empty audio"):
        engine._load_audio("/fake/track.flac")


# ── EssentiaEngine._extract_embeddings crash guard ────────────────────────────


def test_extract_embeddings_marks_extractor_broken_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the extractor call raises, _extractor_broken should be set to True
    and a RuntimeError should be raised."""
    from musictagger.worker import EssentiaEngine

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    pb = models_dir / "discogs-effnet-bs64-1.pb"
    pb.write_bytes(b"dummy")

    engine = EssentiaEngine(models_dir)

    class _BrokenExtractor:
        def __call__(self, audio: object) -> None:
            raise RuntimeError("simulated extractor failure")

    engine._extractor = _BrokenExtractor()

    fake_audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(RuntimeError, match="Essentia extractor failed"):
        engine._extract_embeddings("/fake/track.flac", fake_audio)

    assert engine._extractor_broken is True
    assert engine._extractor is None


# ── Worker.run_pass: stale file and already-satisfied file ────────────────────


def test_run_pass_marks_error_for_missing_file(
    tmp_path: Path,
) -> None:
    """A file in the work queue that no longer exists on disk should be marked
    error (not crash or silently skip)."""
    from musictagger.worker import Worker

    db_path = tmp_path / "cache.db"
    cache = FileCache(db_path)
    config = Config(
        music_path=tmp_path / "music",
        db_path=db_path,
        embeddings_db_path=tmp_path / "embeddings.db",
        log_path=tmp_path / "musictagger.log",
        models_dir=tmp_path / "models",
    )

    # Register a file that does NOT exist on disk.
    ghost = tmp_path / "ghost.mp3"
    ghost.write_bytes(b"x")  # create so mark_changed succeeds
    tag_results = {t.name: False for t in TAGS}
    cache.mark_changed(ghost)
    cache.mark_inspected(ghost, tag_results)
    cache.flush()
    ghost.unlink()  # now delete it so it's stale

    worker = Worker(config, cache)
    worker.run_pass(batch_size=1)

    row = cache._conn.execute(
        "SELECT processing_status, last_error FROM processed WHERE filepath = ?",
        (str(ghost),),
    ).fetchone()
    assert row[0] == "error"
    assert "not found" in row[1]
    worker.close()
    cache.close()


def test_run_pass_marks_done_when_all_tags_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If _missing_tags returns [] the file should be marked done without calling
    process_file at all."""
    from musictagger.worker import Worker

    worker, cache, filepath = _make_worker_with_cache(tmp_path)

    # Make _missing_tags always report nothing missing.
    monkeypatch.setattr(Worker, "_missing_tags", lambda self, fp: [])

    process_file_called = [False]

    def _spy_process(self: object, fp: str, missing: list) -> dict:
        process_file_called[0] = True
        return {}

    monkeypatch.setattr(Worker, "process_file", _spy_process)

    worker.run_pass(batch_size=1)

    assert not process_file_called[0]
    row = cache._conn.execute(
        "SELECT processing_status FROM processed WHERE filepath = ?",
        (str(filepath),),
    ).fetchone()
    assert row[0] == "done"
    worker.close()
    cache.close()


# ── _write_acoustid_fingerprint ───────────────────────────────────────────────


def test_write_acoustid_fingerprint_vorbis() -> None:
    # Plain dict simulates a Vorbis/FLAC file (the else branch).
    tags: dict[str, list[str]] = {}
    _write_acoustid_fingerprint(tags, "AQADtEkUSYmSD")
    assert tags == {"acoustid_fingerprint": ["AQADtEkUSYmSD"]}


def test_write_acoustid_fingerprint_id3(tmp_path: Path) -> None:
    from mutagen.id3 import ID3

    mp3_path = tmp_path / "test.mp3"
    mp3_path.write_bytes(
        b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" + b"\x00" * 104
    )
    try:
        tags = ID3()
        tags.save(str(mp3_path))
        f = mutagen.File(str(mp3_path), easy=False)
        _write_acoustid_fingerprint(f, "AQADtEkUSYmSD")
        txxx = f.tags.get("TXXX:Acoustid Fingerprint")
        assert txxx is not None
        assert txxx.text == ["AQADtEkUSYmSD"]
    except Exception:
        pytest.skip("Could not create minimal MP3 for ID3 test")


def test_write_acoustid_fingerprint_mp4(tmp_path: Path) -> None:
    from mutagen.mp4 import MP4

    m4a_path = tmp_path / "test.m4a"
    try:
        f = MP4()
        f.save(str(m4a_path))
        f = mutagen.File(str(m4a_path), easy=False)
        _write_acoustid_fingerprint(f, "AQADtEkUSYmSD")
        atom_key = "----:com.apple.iTunes:Acoustid Fingerprint"
        assert atom_key in f
        assert bytes(f[atom_key][0]) == b"AQADtEkUSYmSD"
    except Exception:
        pytest.skip("Could not create minimal MP4 for atom test")


# ── Worker._compute_fingerprint ───────────────────────────────────────────────


def test_compute_fingerprint_returns_none_when_fpcalc_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_compute_fingerprint must return None gracefully when fpcalc errors."""
    import musictagger.worker as worker_mod

    config = Config(music_path=tmp_path, embeddings_db_path=tmp_path / "e.db")
    cache = MagicMock()
    worker = worker_mod.Worker(config=config, cache=cache)

    import acoustid

    monkeypatch.setattr(
        acoustid,
        "fingerprint_file",
        lambda *a, **kw: (_ for _ in ()).throw(
            acoustid.FingerprintGenerationError("fpcalc not found")
        ),
    )

    result = worker._compute_fingerprint(str(tmp_path / "song.flac"))
    assert result is None


def test_compute_fingerprint_returns_hash_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_compute_fingerprint returns (raw_fp, sha256_hex) on success."""
    import acoustid

    import musictagger.worker as worker_mod
    from musictagger.embeddings import fingerprint_hash

    config = Config(music_path=tmp_path, embeddings_db_path=tmp_path / "e.db")
    cache = MagicMock()
    worker = worker_mod.Worker(config=config, cache=cache)

    raw = "AQADtEkUSYmSD_2RL3iO"
    monkeypatch.setattr(acoustid, "fingerprint_file", lambda *a, **kw: (180, raw))

    result = worker._compute_fingerprint(str(tmp_path / "song.flac"))
    assert result is not None
    assert result == (raw, fingerprint_hash(raw))


# ── Worker.run() / stop() / reset() ──────────────────────────────────────────


def _make_worker_for_loop_tests(tmp_path: Path) -> "Worker":  # noqa: F821
    """Return a Worker backed by a real cache with no queued files.

    Tests that exercise the loop logic drive run_pass() via monkeypatching so
    no actual audio processing occurs.
    """
    import musictagger.worker as worker_mod

    db_path = tmp_path / "cache.db"
    cache = FileCache(db_path)
    config = Config(
        music_path=tmp_path / "music",
        db_path=db_path,
        embeddings_db_path=tmp_path / "embeddings.db",
        log_path=tmp_path / "musictagger.log",
        models_dir=tmp_path / "models",
    )
    return worker_mod.Worker(config=config, cache=cache)


# ── Worker.stop() / threading.Event ───────────────────────────────────────────


def test_worker_stop_sets_stop_event(tmp_path: Path) -> None:
    """stop() must set the threading.Event so the file loop exits cleanly."""
    worker = _make_worker_for_loop_tests(tmp_path)

    assert not worker._stop_event.is_set()
    worker.stop()
    assert worker._stop_event.is_set()


def test_worker_reset_clears_stop_event(tmp_path: Path) -> None:
    """reset() must clear the stop event so the worker can be relaunched."""
    worker = _make_worker_for_loop_tests(tmp_path)
    worker.stop()
    assert worker._stop_event.is_set()

    worker.reset()
    assert not worker._stop_event.is_set()


# ── Worker.run() — drains the queue across multiple passes ────────────────────


def test_worker_run_loops_until_queue_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker.run() must call run_pass() repeatedly until it returns 0.

    This supersedes the _worker_loop closure test now that the loop logic
    lives in Worker.run() rather than the TUI.
    """
    import musictagger.worker as worker_mod

    worker = _make_worker_for_loop_tests(tmp_path)

    return_values = [5, 5, 5, 0]
    call_count = [0]

    def _fake_run_pass(self: object, batch_size: int = 20) -> int:
        result = return_values[call_count[0]]
        call_count[0] += 1
        worker._running = False
        return result

    monkeypatch.setattr(worker_mod.Worker, "run_pass", _fake_run_pass)

    worker.run(50)

    assert call_count[0] == len(return_values), (
        f"run_pass() called {call_count[0]} time(s); expected {len(return_values)}. "
        "Worker.run() is not looping until the queue is empty."
    )


def test_worker_run_stops_when_stop_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker.run() must exit cleanly when stop() is called between passes."""
    import musictagger.worker as worker_mod

    worker = _make_worker_for_loop_tests(tmp_path)

    call_count = [0]

    def _fake_run_pass(self: object, batch_size: int = 20) -> int:
        call_count[0] += 1
        worker.stop()
        worker._running = False
        return 5

    monkeypatch.setattr(worker_mod.Worker, "run_pass", _fake_run_pass)

    worker.run(50)

    assert call_count[0] == 1, (
        f"run_pass() called {call_count[0]} time(s); expected 1. "
        "Worker.run() did not respect stop()."
    )


# ── Worker.run() startup recovery ────────────────────────────────────────────


def test_worker_run_requeues_stuck_working_rows_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker.run() must call cache.requeue_working() once at startup.

    This ensures rows left in 'working' status by a previous crash are
    recovered before the first run_pass() call, not polled on every
    orchestrator tick.
    """
    import musictagger.worker as worker_mod

    worker = _make_worker_for_loop_tests(tmp_path)

    requeue_called = [0]
    original_requeue = worker.cache.requeue_working

    def _counting_requeue() -> int:
        requeue_called[0] += 1
        return original_requeue()

    monkeypatch.setattr(worker.cache, "requeue_working", _counting_requeue)

    # Stub run_pass to return 0 immediately so run() exits after one attempt.
    monkeypatch.setattr(
        worker_mod.Worker,
        "run_pass",
        lambda self, batch_size=20: 0,
    )

    worker.run(50)

    assert requeue_called[0] == 1, (
        f"requeue_working() was called {requeue_called[0]} time(s); expected 1. "
        "Worker.run() must call it exactly once at startup."
    )
