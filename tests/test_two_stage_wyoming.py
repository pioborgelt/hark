"""Unit and integration tests for HarkEngine, HarkStream, CLI, and Wyoming Protocol server."""
from __future__ import annotations

import asyncio
import functools
from pathlib import Path

import numpy as np
import pytest
import torch
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer
from wyoming.wake import Detect, NotDetected

from hark.app import main as cli_main
from hark.model import AcousticModel
from hark.streaming import HarkEngine, parse_phrase_list, parse_phrase_spec, resolve_sensitivity
from hark.verifier import VERIFIER_KIND, BiPhonemeVerifier
from hark.wyoming_server import HarkEventHandler, HarkWyomingState


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("hark_tiny_ckpts")
    # Build minimal valid IPA vocab covering English + German test phrases
    from hark.text import phones

    syms = sorted(
        set(phones("Okay Haus", "de", "espeak-case-v2", "espeak-v1"))
        | set(phones("Okay Haus", "de", "espeak-case-v2", "de-lexicon-v1"))
        | set(phones("Hey Jarvis", "en", "espeak-case-v2", "espeak-v1"))
        | {"a", "ɛ", "ɪ", "o", "u", "ə", "ɐ", "ɚ", "ɜ", "t", "ɾ", "s", "z", "k", "p"}
    )
    vocab = {sym: idx + 1 for idx, sym in enumerate(syms)}

    torch.manual_seed(0)
    s1 = AcousticModel(len(vocab) + 1, hidden=32, causal_layers=2, dropout=0.0, bins=64)
    s1_path = root / "stage1.pt"
    torch.save(
        {
            "model": s1.state_dict(),
            "vocab": vocab,
            "hidden": 32,
            "causal_layers": 2,
            "dropout": 0.0,
            "bins": 64,
            "text_version": "espeak-case-v2",
            "posterior_temperature": 1.0,
        },
        s1_path,
    )

    s2 = BiPhonemeVerifier(
        len(vocab) + 1,
        hidden=32,
        gru_hidden=32,
        gru_layers=1,
        attn_dim=32,
        attn_heads=2,
        dropout=0.0,
        bins=64,
    )
    s2_path = root / "stage2.pt"
    torch.save(
        {
            "model": s2.state_dict(),
            "model_kind": VERIFIER_KIND,
            "vocab": vocab,
            "hidden": 32,
            "gru_hidden": 32,
            "gru_layers": 1,
            "attn_dim": 32,
            "attn_heads": 2,
            "bins": 64,
            "dropout": 0.0,
            "text_version": "espeak-case-v2",
        },
        s2_path,
    )
    return s1_path, s2_path


def test_phrase_parsing_and_sensitivity():
    assert parse_phrase_spec("Okay Haus:de") == ("Okay Haus", "de")
    assert parse_phrase_spec("de:Okay Haus") == ("Okay Haus", "de")
    assert parse_phrase_spec("Hey Jarvis", default_language="en") == ("Hey Jarvis", "en")
    assert parse_phrase_list("Okay Haus:de; Hey Jarvis:en") == [
        ("Okay Haus", "de"),
        ("Hey Jarvis", "en"),
    ]
    cfg_strict = resolve_sensitivity("strict")
    cfg_sens = resolve_sensitivity("sensitive")
    assert cfg_strict["combined_threshold"] > cfg_sens["combined_threshold"]
    cfg_num = resolve_sensitivity(0.5, threshold_override=-0.25)
    assert cfg_num["combined_threshold"] == -0.25


def test_engine_enrollment_and_streaming(tiny_checkpoints):
    s1_path, s2_path = tiny_checkpoints
    eng = HarkEngine(stage1_checkpoint=s1_path, stage2_checkpoint=s2_path, threads=1)
    de_target = eng.enroll("Okay Haus", language="de")
    en_target = eng.enroll("Hey Jarvis", language="en")
    assert len(de_target.ipa_variants) >= 2  # espeak-v1 + de-lexicon-v1 for "Okay"
    assert en_target.language == "en"

    stream = eng.create_stream([de_target, en_target])
    noise = (np.random.RandomState(1).randn(16000) * 0.01).astype(np.float32)
    evs = stream.push(noise) + stream.flush()
    assert isinstance(evs, list)


def test_cli_phones():
    assert cli_main(["phones", "Okay Haus:de", "Hey Jarvis:en"]) == 0


def test_wyoming_describe_and_stream(tiny_checkpoints):
    s1_path, s2_path = tiny_checkpoints

    async def _run():
        eng = HarkEngine(stage1_checkpoint=s1_path, stage2_checkpoint=s2_path, threads=1)
        st = HarkWyomingState(eng, [eng.enroll("Okay Haus", "de"), eng.enroll("Hey Jarvis", "en")])
        srv = AsyncServer.from_uri("tcp://127.0.0.1:0")
        task = asyncio.create_task(srv.run(functools.partial(HarkEventHandler, st)))
        try:
            for _ in range(30):
                if getattr(srv, "_server", None) is not None:
                    break
                await asyncio.sleep(0.02)
            port = srv._server.sockets[0].getsockname()[1]

            async with AsyncTcpClient("127.0.0.1", port) as client:
                await client.write_event(Describe().event())
                info = Info.from_event(await client.read_event())
                assert info.wake[0].name == "hark"
                assert {m.name for m in info.wake[0].models} == {"okay_haus_de", "hey_jarvis_en"}

            pcm = (np.zeros(8000, dtype=np.int16)).tobytes()
            async with AsyncTcpClient("127.0.0.1", port) as client:
                await client.write_event(Detect(names=["okay_haus_de"]).event())
                await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())
                await client.write_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
                await client.write_event(AudioStop().event())
                reply = await client.read_event()
                assert NotDetected.is_type(reply.type)
        finally:
            task.cancel()

    asyncio.run(_run())
