"""Home Assistant Wyoming Protocol server for Hark (`wyoming-hark`)."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from functools import partial
from pathlib import Path

import soundfile as sf
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, WakeModel, WakeProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.wake import Detect, Detection, NotDetected

from .audio import RATE
from .streaming import EnrolledPhrase, HarkEngine, HarkStream, parse_phrase_list, parse_phrase_spec

LOGGER = logging.getLogger("wyoming_hark")
VERSION = "0.3.0-alpha"


class HarkWyomingState:
    def __init__(
        self,
        engine: HarkEngine,
        phrases: list[EnrolledPhrase],
        sensitivity: str | float = "balanced",
        threshold_override: float | None = None,
        log_near_misses: bool = True,
        debug_clip_dir: Path | None = None,
        max_debug_clips: int = 30,
    ):
        self.engine = engine
        self.phrases: dict[str, EnrolledPhrase] = {p.name: p for p in phrases}
        self.sensitivity = sensitivity
        self.threshold_override = threshold_override
        self.log_near_misses = log_near_misses
        self.debug_clip_dir = debug_clip_dir
        self.max_debug_clips = max_debug_clips
        self.lock = asyncio.Lock()
        if self.debug_clip_dir is not None:
            self.debug_clip_dir.mkdir(parents=True, exist_ok=True)

    def build_info(self) -> Info:
        attribution = Attribution(
            name="Pio Borgelt / Hark Contributors",
            url="https://github.com/pioborgelt/hark",
        )
        models = [
            WakeModel(
                name=p.name,
                description=f'Zero-shot {p.language.upper()} wake phrase "{p.phrase}" (IPA: /{p.ipa_variants[0]}/)',
                phrase=p.phrase,
                attribution=attribution,
                installed=True,
                languages=[p.language],
                version=VERSION,
            )
            for p in self.phrases.values()
        ]
        return Info(
            wake=[
                WakeProgram(
                    name="hark",
                    description="Zero-shot text-enrolled EN/DE two-stage wake-word engine",
                    attribution=attribution,
                    installed=True,
                    version=VERSION,
                    models=models,
                )
            ]
        )

    def resolve_targets(self, names: list[str] | None) -> list[EnrolledPhrase]:
        if not names:
            return list(self.phrases.values())
        resolved: list[EnrolledPhrase] = []
        for req in names:
            if req in self.phrases:
                resolved.append(self.phrases[req])
                continue
            by_phrase = next(
                (p for p in self.phrases.values() if p.phrase.lower() == req.strip().lower()),
                None,
            )
            if by_phrase is not None:
                resolved.append(by_phrase)
                continue
            # Support dynamic zero-shot enrollment directly from Detect(names=["Okay Küche:de"])
            phrase_txt, lang = parse_phrase_spec(req, default_language="en")
            enrolled = self.engine.enroll(
                phrase_txt,
                language=lang,
                name=req if ":" not in req else None,
                sensitivity=self.sensitivity,
                threshold_override=self.threshold_override,
            )
            self.phrases[enrolled.name] = enrolled
            resolved.append(enrolled)
        return resolved

    def save_debug_clip(self, kind: str, audio_ring, meta: dict) -> None:
        if self.debug_clip_dir is None or len(audio_ring) == 0:
            return
        try:
            existing = sorted(self.debug_clip_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            while len(existing) >= self.max_debug_clips:
                old = existing.pop(0)
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
            stamp = f"{int(time.time() * 1000)}_{kind}_{meta['name']}"
            wav_path = self.debug_clip_dir / f"{stamp}.wav"
            sf.write(str(wav_path), audio_ring[-RATE * 3 :], RATE)
            wav_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        except Exception as exc:
            LOGGER.warning("Failed to write debug clip: %s", exc)


class HarkEventHandler(AsyncEventHandler):
    def __init__(self, state: HarkWyomingState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self.converter = AudioChunkConverter(rate=RATE, width=2, channels=1)
        self.stream: HarkStream | None = None
        self.active_names: list[str] | None = None
        self.detected_in_utterance = False

    def _ensure_stream(self) -> HarkStream:
        if self.stream is None:
            targets = self.state.resolve_targets(self.active_names)
            self.stream = self.state.engine.create_stream(
                targets=targets,
                on_near_miss=self._on_near_miss,
            )
        return self.stream

    def _on_near_miss(self, info: dict) -> None:
        if self.state.log_near_misses:
            LOGGER.info(
                'Near-miss rejected by Stage-2: phrase="%s" (%s) combined=%.2f < %.2f '
                "(s1=%.2f, s2=%.2f, rms=%.1fdBFS)",
                info["phrase"],
                info["language"],
                info["combined_score"],
                info["threshold"],
                info["s1_score"],
                info["s2_logit"],
                info["rms_dbfs"],
            )
        if self.stream is not None:
            self.state.save_debug_clip("nearmiss", self.stream.audio_ring, info)

    async def _emit_detections(self, detections: list[dict]) -> None:
        for det in detections:
            self.detected_in_utterance = True
            LOGGER.info(
                'DETECTED: phrase="%s" (%s) name=%s combined=%.2f >= %.2f '
                "(s1=%.2f, s2=%.2f, rms=%.1fdBFS)",
                det["phrase"],
                det["language"],
                det["name"],
                det["combined_score"],
                det["threshold"],
                det["s1_score"],
                det["s2_logit"],
                det["rms_dbfs"],
            )
            if self.stream is not None:
                self.state.save_debug_clip("detected", self.stream.audio_ring, det)
            await self.write_event(
                Detection(
                    name=det["name"],
                    timestamp=int(det["time"] * 1000),
                ).event()
            )

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.state.build_info().event())
            return True

        if Detect.is_type(event.type):
            detect = Detect.from_event(event)
            self.active_names = detect.names or None
            self.stream = None
            self.detected_in_utterance = False
            return True

        if AudioStart.is_type(event.type):
            self.detected_in_utterance = False
            if self.stream is not None:
                self.stream.reset()
            return True

        if AudioChunk.is_type(event.type):
            chunk = self.converter.convert(AudioChunk.from_event(event))
            stream = self._ensure_stream()
            async with self.state.lock:
                detections = stream.push_pcm16(chunk.audio)
            if detections:
                await self._emit_detections(detections)
            return True

        if AudioStop.is_type(event.type):
            if self.stream is not None:
                async with self.state.lock:
                    detections = self.stream.flush()
                if detections:
                    await self._emit_detections(detections)
                self.stream.reset()
            if not self.detected_in_utterance:
                await self.write_event(NotDetected().event())
            return False

        return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wyoming-hark",
        description="Wyoming Protocol wake-word server for Home Assistant powered by Hark.",
    )
    p.add_argument(
        "--uri",
        default=os.environ.get("HARK_URI", "tcp://0.0.0.0:10400"),
        help="Server URI, e.g. tcp://0.0.0.0:10400",
    )
    p.add_argument(
        "--phrase",
        action="append",
        dest="phrases",
        default=[],
        help='Wake phrase with language tag, e.g. "Okay Haus:de" or "Hey Jarvis:en" (repeatable)',
    )
    p.add_argument(
        "--default-language",
        choices=["en", "de"],
        default=os.environ.get("HARK_LANGUAGE", "en"),
    )
    p.add_argument(
        "--sensitivity",
        default=os.environ.get("HARK_SENSITIVITY", "balanced"),
        help="strict | balanced | sensitive | float in [0.0, 1.0]",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ["HARK_THRESHOLD"]) if os.environ.get("HARK_THRESHOLD") else None,
        help="Optional explicit combined Stage-1+Stage-2 score threshold override",
    )
    p.add_argument("--stage1", default=os.environ.get("HARK_STAGE1_MODEL"), help="Path to Stage-1 checkpoint")
    p.add_argument("--stage2", default=os.environ.get("HARK_STAGE2_MODEL"), help="Path to Stage-2 checkpoint")
    p.add_argument("--cache-dir", default=os.environ.get("HARK_CACHE_DIR"), help="Model download cache directory")
    p.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("HARK_THREADS", "2")),
        help="PyTorch CPU inference threads (default: 2)",
    )
    p.add_argument(
        "--no-near-miss-logs",
        action="store_true",
        default=os.environ.get("HARK_LOG_NEAR_MISSES", "true").lower() in ("0", "false", "no"),
        help="Disable logging Stage-1 candidates rejected by Stage-2",
    )
    p.add_argument(
        "--debug-clip-dir",
        default=os.environ.get("HARK_DEBUG_CLIP_DIR", ""),
        help="Optional directory to save last 30 detected/near-miss 3s WAV clips locally",
    )
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    return p


async def async_main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_phrases = list(args.phrases)
    env_phrases = os.environ.get("HARK_PHRASES", "").strip()
    if env_phrases:
        raw_phrases.append(env_phrases)
    if not raw_phrases:
        raw_phrases = ["Hey Jarvis:en", "Okay Haus:de"]

    parsed = parse_phrase_list(raw_phrases, default_language=args.default_language)
    engine = HarkEngine(
        stage1_checkpoint=args.stage1,
        stage2_checkpoint=args.stage2,
        threads=args.threads,
        cache_dir=args.cache_dir,
    )
    enrolled = [
        engine.enroll(
            phrase=p_txt,
            language=lang,
            sensitivity=args.sensitivity,
            threshold_override=args.threshold,
        )
        for p_txt, lang in parsed
    ]

    state = HarkWyomingState(
        engine=engine,
        phrases=enrolled,
        sensitivity=args.sensitivity,
        threshold_override=args.threshold,
        log_near_misses=not args.no_near_miss_logs,
        debug_clip_dir=Path(args.debug_clip_dir) if args.debug_clip_dir else None,
    )

    server = AsyncServer.from_uri(args.uri)
    LOGGER.info(
        "wyoming-hark %s listening on %s (%d wake phrases ready)",
        VERSION,
        args.uri,
        len(enrolled),
    )
    await server.run(partial(HarkEventHandler, state))


def main(argv: list[str] | None = None) -> None:
    try:
        asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
