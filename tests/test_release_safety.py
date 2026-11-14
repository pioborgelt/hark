"""Release hardening: pinned weights, pickle refusal, enrollment cap, add-on entrypoint."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from hark import streaming
from hark.wyoming_server import HarkWyomingState


def _pin(monkeypatch, key: str, filename: str, payload: bytes) -> None:
    spec = dict(streaming.MODEL_SPECS[key], filename=filename,
                sha256=hashlib.sha256(payload).hexdigest(), local_fallbacks=[])
    monkeypatch.setitem(streaming.MODEL_SPECS, key, spec)


def test_cached_release_file_must_match_pin(tmp_path, monkeypatch):
    _pin(monkeypatch, "stage1", "s1.pt", b"good-1")
    _pin(monkeypatch, "stage2", "s2.pt", b"good-2")
    (tmp_path / "s1.pt").write_bytes(b"good-1")
    (tmp_path / "s2.pt").write_bytes(b"tampered")
    downloads = []

    def fake_download(repo, tag, filename, dest, expected_sha256=None):
        downloads.append(filename)
        (dest.parent / (dest.name + ".part")).write_bytes(b"still-wrong")
        return streaming._finish_download(dest.parent / (dest.name + ".part"), dest, expected_sha256)

    monkeypatch.setattr(streaming, "_download_release_asset", fake_download)
    with pytest.raises(streaming.ChecksumError):
        streaming.resolve_checkpoints(cache_dir=tmp_path)
    # The bad cache entry is discarded and a fresh download is verified, not trusted.
    assert downloads == ["s2.pt"]
    assert not (tmp_path / "s2.pt").exists()
    assert not (tmp_path / "s2.pt.part").exists()


def test_valid_cache_needs_no_download(tmp_path, monkeypatch):
    _pin(monkeypatch, "stage1", "s1.pt", b"good-1")
    _pin(monkeypatch, "stage2", "s2.pt", b"good-2")
    (tmp_path / "s1.pt").write_bytes(b"good-1")
    (tmp_path / "s2.pt").write_bytes(b"good-2")
    monkeypatch.setattr(streaming, "_download_release_asset",
                        lambda *a, **k: pytest.fail("unexpected download"))
    assert streaming.resolve_checkpoints(cache_dir=tmp_path) == (tmp_path / "s1.pt", tmp_path / "s2.pt")


class _Payload:
    def __reduce__(self):
        return (print, ("pickled code ran",))


def test_checkpoint_with_pickled_objects_is_refused(tmp_path):
    safe, unsafe = tmp_path / "safe.pt", tmp_path / "unsafe.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "vocab": {"a": 1}}, safe)
    torch.save({"model": {}, "extra": _Payload()}, unsafe)
    streaming.assert_weights_only(safe)
    with pytest.raises(ValueError, match="weights_only"):
        streaming.assert_weights_only(unsafe)


class _Engine:
    def __init__(self):
        self.calls = []

    def enroll(self, phrase, language="en", name=None, **_):
        if phrase == "bad":
            raise ValueError("Untrained phonemes")
        self.calls.append(phrase)
        return streaming.EnrolledPhrase(
            name=name or streaming.slugify_phrase(phrase, language), phrase=phrase,
            language=language, ipa_variants=("x",), stage1_token_sets=((1,),),
            stage2_token_sets=((1,),), s1_threshold=0.0, v_weight=0.5,
            combined_threshold=0.0, temperature=1.0, offset=0.0)


def test_dynamic_enrollment_is_capped_and_errors_fall_back():
    engine = _Engine()
    base = engine.enroll("Okay Haus", "de")
    state = HarkWyomingState(engine, [base], max_dynamic_phrases=2)
    engine.calls.clear()
    names = [t.name for t in state.resolve_targets(["one:en", "two:en", "three:en"])]
    assert names == ["one_en", "two_en"]
    assert len(state.phrases) == 3
    # Beyond the cap nothing new is enrolled; the stream falls back to all known phrases.
    assert state.resolve_targets(["four:en"]) == list(state.phrases.values())
    assert engine.calls == ["one", "two"]

    # Unpronounceable requests are logged and skipped instead of breaking the connection.
    fresh = HarkWyomingState(engine, [base])
    assert fresh.resolve_targets(["bad", "okay_haus_de"]) == [base]
    assert fresh.dynamic_enrollments == 0


def _load_run_py():
    path = Path(__file__).parents[1] / "hark_addon" / "run.py"
    spec = importlib.util.spec_from_file_location("hark_addon_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_addon_options_to_argv():
    run = _load_run_py()
    argv = run.build_argv({"phrases": ["Okay Haus:de", " ", "Hey Jarvis:en"], "sensitivity": "strict",
                           "threads": 3, "log_near_misses": False, "save_debug_clips": True})
    assert argv[:7] == ["wyoming-hark", "--uri", "tcp://0.0.0.0:10400",
                        "--sensitivity", "strict", "--threads", "3"]
    assert argv.count("--phrase") == 2
    assert "--no-near-miss-logs" in argv
    assert argv[-2:] == ["--debug-clip-dir", "/data/debug_clips"]


def test_addon_discovery_without_supervisor_is_skipped(monkeypatch, capsys):
    run = _load_run_py()
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    run.register_discovery()
    assert "skipping discovery" in capsys.readouterr().out


def test_stream_status_log_reports_level_and_quiet_audio(caplog):
    import logging

    import numpy as np

    from hark import wyoming_server

    handler = wyoming_server.HarkEventHandler.__new__(wyoming_server.HarkEventHandler)
    handler._reset_stats()
    caplog.set_level(logging.INFO, logger="wyoming_hark")
    second = np.full(wyoming_server.RATE, 3277, dtype="<i2").tobytes()  # -20 dBFS
    for _ in range(int(wyoming_server.STATUS_INTERVAL_SECONDS)):
        handler._track_audio(second)
    assert "peak level -20.0 dBFS" in caplog.text
    assert "nearly silent" not in caplog.text

    caplog.clear()
    silent = np.zeros(wyoming_server.RATE, dtype="<i2").tobytes()
    for _ in range(int(wyoming_server.STATUS_INTERVAL_SECONDS)):
        handler._track_audio(silent)
    assert "nearly silent" in caplog.text
    handler._log_stream_end("client disconnected")
    assert "Audio stream ended (client disconnected) after 120.0 s" in caplog.text


def test_addon_stop_by_supervisor_is_clean_exit():
    import signal

    run = _load_run_py()
    assert run.exit_code(-signal.SIGTERM, stop_requested=True) == 0
    assert run.exit_code(0, stop_requested=True) == 0
    # Real crashes stay visible: own non-zero codes and unrequested kills.
    assert run.exit_code(1, stop_requested=True) == 1
    assert run.exit_code(-signal.SIGKILL, stop_requested=False) == 137


def test_wake_word_names_shown_in_home_assistant_are_short():
    engine = _Engine()
    state = HarkWyomingState(engine, [engine.enroll("Okay Jarvis", "en"), engine.enroll("Okay Haus", "de")])
    models = state.build_info().wake[0].models
    assert [(m.name, m.description, m.phrase) for m in models] == [
        ("okay_jarvis_en", "Okay Jarvis (EN)", "Okay Jarvis"),
        ("okay_haus_de", "Okay Haus (DE)", "Okay Haus"),
    ]


def test_phrase_specs_reject_nonsense_and_unknown_languages():
    from hark.streaming import parse_phrase_spec

    assert parse_phrase_spec(" Hallo  Kühlschrank : DE ") == ("Hallo Kühlschrank", "de")
    assert parse_phrase_spec("de:Okay Haus") == ("Okay Haus", "de")
    assert parse_phrase_spec("Computer 3000:en") == ("Computer 3000", "en")
    for bad in ["Okay Haus:fr", ":de", "!!!:en", "你好:en", "Привет:de", "🔥:en", "Okay:Haus"]:
        with pytest.raises(ValueError):
            parse_phrase_spec(bad)


def test_untagged_phrase_warns_about_default_language(caplog):
    import logging

    from hark.streaming import parse_phrase_spec

    caplog.set_level(logging.WARNING, logger="hark")
    assert parse_phrase_spec("Hallo Haus") == ("Hallo Haus", "en")
    assert 'Write "Hallo Haus:de"' in caplog.text


def test_server_skips_invalid_and_duplicate_phrases(caplog):
    import logging

    from hark.wyoming_server import enroll_phrases

    engine = _Engine()
    caplog.set_level(logging.WARNING, logger="wyoming_hark")
    targets = enroll_phrases(engine, ["Okay Haus:de", "Okay Haus:fr", "bad:en", "Okay Haus:de", "Hey Jarvis:en"])
    assert [t.name for t in targets] == ["okay_haus_de", "hey_jarvis_en"]
    assert "Skipping wake phrase \"Okay Haus:fr\"" in caplog.text
    assert "Skipping wake phrase \"bad:en\"" in caplog.text
    assert "Skipping duplicate wake phrase" in caplog.text


def test_weights_come_from_pinned_huggingface_revision_then_github(tmp_path, monkeypatch):
    calls = []

    def fake_url(url, dest, expected, token):
        calls.append(url)
        raise OSError("offline")

    def fake_github(repo, tag, filename, dest, expected=None):
        calls.append(f"github:{repo}@{tag}/{filename}")
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(streaming, "_download_url", fake_url)
    monkeypatch.setattr(streaming, "_download_github_asset", fake_github)
    streaming._download_release_asset("pioborgelt/hark", "v0", "w.pt", tmp_path / "w.pt", None)
    assert calls[0] == f"https://huggingface.co/{streaming.HF_REPO}/resolve/{streaming.HF_REVISION}/w.pt"
    assert len(streaming.HF_REVISION) == 40
    assert calls[1] == "github:pioborgelt/hark@v0/w.pt"


def test_checksum_mismatch_from_huggingface_is_not_masked_by_fallback(tmp_path, monkeypatch):
    def bad(url, dest, expected, token):
        raise streaming.ChecksumError("tampered")

    monkeypatch.setattr(streaming, "_download_url", bad)
    monkeypatch.setattr(streaming, "_download_github_asset", lambda *a, **k: pytest.fail("no fallback"))
    with pytest.raises(streaming.ChecksumError):
        streaming._download_release_asset("r", "t", "w.pt", tmp_path / "w.pt", "abc")
