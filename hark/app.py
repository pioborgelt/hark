"""User-facing CLI for Hark v0.3 (`hark listen`, `hark detect`, `hark phones`, `hark serve`)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import sys
import time

import numpy as np

from .audio import RATE, read_audio
from .streaming import HarkEngine, parse_phrase_list, parse_phrase_spec
from .text import phones


def _cmd_phones(args: argparse.Namespace) -> int:
    for raw in args.phrases:
        phrase_txt, lang = parse_phrase_spec(raw, default_language=args.language)
        base_ipa = "".join(phones(phrase_txt, lang, "espeak-case-v2", "espeak-v1"))
        variants = [f"/{base_ipa}/"]
        if lang == "de":
            lex_ipa = "".join(phones(phrase_txt, "de", "espeak-case-v2", "de-lexicon-v1"))
            if lex_ipa != base_ipa:
                variants.append(f"/{lex_ipa}/")
        print(json.dumps({"phrase": phrase_txt, "language": lang, "ipa": variants}, ensure_ascii=False))
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    engine = HarkEngine(
        stage1_checkpoint=args.stage1,
        stage2_checkpoint=args.stage2,
        threads=args.threads,
        cache_dir=args.cache_dir,
    )
    parsed = parse_phrase_list(args.phrases, default_language=args.language)
    targets = [
        engine.enroll(
            p_txt,
            language=lang,
            sensitivity=args.sensitivity,
            threshold_override=args.threshold,
        )
        for p_txt, lang in parsed
    ]

    near_misses: list[dict] = []

    def _on_near_miss(info: dict) -> None:
        near_misses.append(info)
        if not args.quiet_near_miss:
            print(json.dumps({"event": "near_miss", **info}, ensure_ascii=False), file=sys.stderr, flush=True)

    stream = engine.create_stream(targets=targets, on_near_miss=_on_near_miss)
    audio = read_audio(args.wav)
    chunk_samples = max(160, int(RATE * args.chunk_ms / 1000))

    t0 = time.perf_counter()
    detections: list[dict] = []
    for offset in range(0, len(audio), chunk_samples):
        evs = stream.push(audio[offset : offset + chunk_samples])
        for ev in evs:
            detections.append(ev)
            print(json.dumps({"event": "detected", **ev}, ensure_ascii=False), flush=True)
    for ev in stream.flush():
        detections.append(ev)
        print(json.dumps({"event": "detected", **ev}, ensure_ascii=False), flush=True)

    elapsed = time.perf_counter() - t0
    dur = len(audio) / RATE if len(audio) else 0.0
    summary = {
        "event": "summary",
        "detections": len(detections),
        "near_misses": len(near_misses),
        "audio_seconds": round(dur, 3),
        "wall_seconds": round(elapsed, 4),
        "real_time_factor": round(elapsed / dur, 4) if dur > 0 else None,
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _cmd_listen(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "hark listen requires `sounddevice` (install with `pip install 'hark-ww[microphone]'`).",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    engine = HarkEngine(
        stage1_checkpoint=args.stage1,
        stage2_checkpoint=args.stage2,
        threads=args.threads,
        cache_dir=args.cache_dir,
    )
    parsed = parse_phrase_list(args.phrases, default_language=args.language)
    targets = [
        engine.enroll(
            p_txt,
            language=lang,
            sensitivity=args.sensitivity,
            threshold_override=args.threshold,
        )
        for p_txt, lang in parsed
    ]

    def _on_near_miss(info: dict) -> None:
        if not args.quiet_near_miss:
            print(
                f"[near-miss] \"{info['phrase']}\" ({info['language']}) "
                f"score={info['combined_score']:.2f} < {info['threshold']:.2f} "
                f"(s1={info['s1_score']:.2f}, s2={info['s2_logit']:.2f}, rms={info['rms_dbfs']:.1f}dBFS)",
                flush=True,
            )

    stream = engine.create_stream(targets=targets, on_near_miss=_on_near_miss)
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    blocksize = max(320, int(RATE * args.chunk_ms / 1000))

    def _callback(indata, _frames, _time_info, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr, flush=True)
        audio_q.put(indata[:, 0].copy())

    print(
        f"Listening on microphone ({RATE} Hz) for: "
        + ", ".join(f'"{t.phrase}" [{t.language}]' for t in targets)
        + "  (Ctrl+C to stop)",
        flush=True,
    )
    started = time.monotonic()
    try:
        with sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=args.device,
            callback=_callback,
        ):
            while True:
                if args.seconds is not None and (time.monotonic() - started) >= args.seconds:
                    break
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                for det in stream.push(chunk):
                    print(json.dumps({"event": "detected", **det}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hark",
        description="Hark: Zero-shot text-enrolled EN/DE streaming wake-word engine.",
    )
    p.add_argument("--version", action="version", version="hark 0.3.0-alpha")
    sub = p.add_subparsers(dest="command", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--phrase",
            "-p",
            action="append",
            dest="phrases",
            required=True,
            help='Wake phrase with optional :en / :de tag, e.g. "Okay Haus:de" (repeatable)',
        )
        sp.add_argument("--language", "-l", choices=["en", "de"], default="en", help="Default language if untagged")
        sp.add_argument(
            "--sensitivity",
            "-s",
            default="balanced",
            help="strict | balanced | sensitive | float in [0.0, 1.0]",
        )
        sp.add_argument("--threshold", type=float, default=None, help="Explicit combined score threshold override")
        sp.add_argument("--stage1", default=None, help="Optional Stage-1 checkpoint path")
        sp.add_argument("--stage2", default=None, help="Optional Stage-2 verifier checkpoint path")
        sp.add_argument("--cache-dir", default=None, help="Model cache directory")
        sp.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads (default: 2)")
        sp.add_argument("--chunk-ms", type=int, default=80, help="Streaming audio chunk size in ms (default: 80)")
        sp.add_argument("--quiet-near-miss", action="store_true", help="Suppress Stage-2 near-miss log lines")

    det = sub.add_parser("detect", help="Run streaming wake-word detection on an audio file")
    det.add_argument("wav", help="Path to WAV/FLAC audio file")
    _add_common(det)

    lis = sub.add_parser("listen", help="Listen on a local microphone in real time")
    _add_common(lis)
    lis.add_argument("--device", default=None, help="Optional sounddevice input device index/name")
    lis.add_argument("--seconds", type=float, default=None, help="Optional duration limit in seconds")

    ph = sub.add_parser("phones", help="Inspect IPA phonemization for one or more phrases")
    ph.add_argument("phrases", nargs="+", help='Phrases, e.g. "Okay Haus:de" "Hey Jarvis:en"')
    ph.add_argument("--language", "-l", choices=["en", "de"], default="en")

    srv = sub.add_parser("serve", help="Start the Home Assistant Wyoming Protocol wake-word server")
    srv.add_argument("wyoming_args", nargs=argparse.REMAINDER, help="Arguments forwarded to wyoming-hark")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "phones":
        return _cmd_phones(args)
    if args.command == "detect":
        return _cmd_detect(args)
    if args.command == "listen":
        return _cmd_listen(args)
    if args.command == "serve":
        from .wyoming_server import main as wyoming_main

        forwarded = [a for a in args.wyoming_args if a != "--"]
        wyoming_main(forwarded)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
