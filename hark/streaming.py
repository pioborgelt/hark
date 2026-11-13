"""Unified two-stage streaming wake-word engine (Causal CTC Stage 1 + BiPhonemeVerifier Stage 2).

Supports zero-shot English and German text enrollment with multi-pronunciation expansion,
shared Stage-1 acoustic encoding across multiple phrases, a short post-trigger lookahead
ringbuffer (~320 ms) for Stage-2 bidirectional span verification, and automatic checkpoint
downloading from GitHub Releases.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .audio import RATE, FeatureStream
from .residual_acoustic_model import load_model
from .supported_detect import Matcher
from .text import encode, encode_multi, phones
from .verifier import load_verifier, score_trigger_queries

LOGGER = logging.getLogger("hark")

RELEASE_TAG = "v0.3.0-alpha"
DEFAULT_REPO = os.environ.get("HARK_MODEL_REPO", "pioborgelt/hark")

MODEL_SPECS = {
    "stage1": {
        "filename": "hark-stage1-r221-h384.pt",
        "local_fallbacks": [
            "models/hark-stage1-r221-h384.pt",
            "artifacts/r221-cleanhybrid-20261113-071351/h384-step200000.pt",
        ],
    },
    "stage2": {
        "filename": "hark-stage2-r222-v384.pt",
        "local_fallbacks": [
            "models/hark-stage2-r222-v384.pt",
            "artifacts/r222-verifier-20261113-071450/v384-step055000.pt",
        ],
    },
}

SENSITIVITY_PRESETS = {
    "strict": {
        "s1_threshold": -1.10,
        "v_weight": 0.50,
        "combined_threshold": 0.85,
        "short_phrase_bonus": 0.45,
    },
    "balanced": {
        "s1_threshold": -1.25,
        "v_weight": 0.50,
        "combined_threshold": 0.15,
        "short_phrase_bonus": 0.45,
    },
    "sensitive": {
        "s1_threshold": -1.45,
        "v_weight": 0.50,
        "combined_threshold": -0.45,
        "short_phrase_bonus": 0.35,
    },
}

STAGE1_GATES = {
    "min_blank_probability": 0.02,
    "min_frame_probability": 0.02,
    "min_phone_probability": 0.0005,
    "phone_margin": 0.01,
    "mean_phone_probability": 0.03,
    "max_seconds_per_phone": 0.40,
}

LANGUAGE_CALIBRATION = {
    "en": (1.25, 2.25),
    "de": (1.25, 1.45),
}


def calibrate_posteriors(lp: np.ndarray, temperature: float, offset: float) -> np.ndarray:
    x = np.asarray(lp, dtype=np.float32).copy()
    x[:, 1:] += offset
    x /= temperature
    x -= x.max(axis=1, keepdims=True)
    x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
    return np.ascontiguousarray(x, dtype=np.float32)


def slugify_phrase(phrase: str, language: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower()).strip("_")
    return f"{slug}_{language}" if slug else f"wake_{language}"


def parse_phrase_spec(spec: str, default_language: str = "en") -> tuple[str, str]:
    """Parse `'Okay Haus:de'`, `'de:Okay Haus'`, or `'Hey Jarvis'` -> `(phrase, lang)`."""
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty wake phrase specification")
    if ":" in raw:
        left, right = raw.split(":", 1)
        if left.strip().lower() in ("de", "en") and right.strip():
            return right.strip(), left.strip().lower()
        if right.strip().lower() in ("de", "en") and left.strip():
            return left.strip(), right.strip().lower()
    if default_language not in ("de", "en"):
        raise ValueError(f"Unsupported language '{default_language}' (use 'en' or 'de')")
    return raw, default_language


def parse_phrase_list(raw: str | Sequence[str], default_language: str = "en") -> list[tuple[str, str]]:
    if isinstance(raw, str):
        items = [part.strip() for part in re.split(r"[;\n]+", raw) if part.strip()]
    else:
        items = []
        for entry in raw:
            items.extend(part.strip() for part in str(entry).split(";") if part.strip())
    return [parse_phrase_spec(item, default_language=default_language) for item in items]


def resolve_sensitivity(
    sensitivity: str | float = "balanced",
    threshold_override: float | None = None,
) -> dict[str, float]:
    if isinstance(sensitivity, str) and sensitivity in SENSITIVITY_PRESETS:
        cfg = dict(SENSITIVITY_PRESETS[sensitivity])
    else:
        try:
            val = float(sensitivity)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sensitivity must be one of {tuple(SENSITIVITY_PRESETS)} or a float in [0.0, 1.0]"
            ) from exc
        val = max(0.0, min(1.0, val))
        # 0.0 -> strict (+1.00), 0.5 -> balanced (+0.15), 1.0 -> sensitive (-0.70)
        cfg = {
            "s1_threshold": -1.05 - 0.50 * val,
            "v_weight": 0.50,
            "combined_threshold": 1.00 - 1.70 * val,
            "short_phrase_bonus": 0.45 - 0.15 * val,
        }
    if threshold_override is not None:
        cfg["combined_threshold"] = float(threshold_override)
        cfg["short_phrase_bonus"] = 0.0
    return cfg


def _download_release_asset(repo: str, tag: str, filename: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
    LOGGER.info("Downloading %s from %s ...", filename, url)

    headers = {"User-Agent": "hark-ww/0.3"}
    if token:
        # Support authenticated download via GitHub API for private repos
        import json as _json

        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        req = urllib.request.Request(
            api_url,
            headers={**headers, "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rel = _json.loads(resp.read().decode("utf-8"))
            asset_url = next((a["url"] for a in rel.get("assets", []) if a["name"] == filename), None)
            if asset_url:
                dl_req = urllib.request.Request(
                    asset_url,
                    headers={
                        **headers,
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/octet-stream",
                    },
                )
                with urllib.request.urlopen(dl_req, timeout=180) as dl, open(tmp, "wb") as out:
                    while chunk := dl.read(1024 * 256):
                        out.write(chunk)
                tmp.replace(dest)
                return dest
        except Exception as exc:
            LOGGER.debug("GitHub API asset lookup fallback (%s), trying direct URL...", exc)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as dl, open(tmp, "wb") as out:
        while chunk := dl.read(1024 * 256):
            out.write(chunk)
    tmp.replace(dest)
    return dest


def resolve_checkpoints(
    stage1_path: str | Path | None = None,
    stage2_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Locate Stage-1 and Stage-2 checkpoints locally or download from GitHub Releases."""
    s1_arg = stage1_path or os.environ.get("HARK_STAGE1_MODEL")
    s2_arg = stage2_path or os.environ.get("HARK_STAGE2_MODEL")
    cache_root = Path(
        cache_dir
        or os.environ.get("HARK_CACHE_DIR")
        or (Path.home() / ".cache" / "hark" / RELEASE_TAG)
    )

    resolved: list[Path] = []
    for key, explicit in (("stage1", s1_arg), ("stage2", s2_arg)):
        if explicit:
            p = Path(explicit)
            if not p.is_file():
                raise FileNotFoundError(f"Configured {key} checkpoint not found: {p}")
            resolved.append(p)
            continue
        spec = MODEL_SPECS[key]
        found = next((Path(cand) for cand in spec["local_fallbacks"] if Path(cand).is_file()), None)
        if found is not None:
            resolved.append(found)
            continue
        cached = cache_root / spec["filename"]
        if cached.is_file():
            resolved.append(cached)
            continue
        resolved.append(_download_release_asset(DEFAULT_REPO, RELEASE_TAG, spec["filename"], cached))
    return resolved[0], resolved[1]


@dataclass(frozen=True)
class EnrolledPhrase:
    name: str
    phrase: str
    language: str
    ipa_variants: tuple[str, ...]
    stage1_token_sets: tuple[tuple[int, ...], ...]
    stage2_token_sets: tuple[tuple[int, ...], ...]
    s1_threshold: float
    v_weight: float
    combined_threshold: float
    temperature: float
    offset: float


@dataclass
class PendingCandidate:
    target: EnrolledPhrase
    end_frame: int
    start_frame: int
    s1_score: float
    rms_dbfs: float


class HarkEngine:
    """Shared read-only Stage-1 + Stage-2 model holder; spawns lightweight `HarkStream` sessions."""

    def __init__(
        self,
        stage1_checkpoint: str | Path | None = None,
        stage2_checkpoint: str | Path | None = None,
        device: str = "cpu",
        threads: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        if threads is not None and threads > 0:
            torch.set_num_threads(int(threads))
        s1_path, s2_path = resolve_checkpoints(stage1_checkpoint, stage2_checkpoint, cache_dir=cache_dir)
        self.device = device
        self.stage1_path = s1_path
        self.stage2_path = s2_path
        self.stage1, self.s1_ck = load_model(str(s1_path), device=device)
        self.stage1.eval()
        self.verifier, self.s2_ck = load_verifier(str(s2_path), device=device)
        self.verifier.eval()
        self.bins = int(self.s1_ck.get("bins", 64))
        self.s1_vocab = self.s1_ck["vocab"]
        self.s2_vocab = self.s2_ck["vocab"]
        self.s1_text_ver = self.s1_ck.get("text_version", "espeak-case-v2")
        self.s2_text_ver = self.s2_ck.get("text_version", "espeak-case-v2")
        self._warm_feats, self._warm_state = self._build_warm_state()

    @torch.no_grad()
    def _build_warm_state(self) -> tuple[np.ndarray, object]:
        rng = np.random.RandomState(0)
        lead_in = (rng.randn(int(RATE * 0.48)) * 1e-4).astype(np.float32)
        fs = FeatureStream(self.bins)
        feats = fs.push(lead_in)
        _, state = self.stage1(feats[None].to(self.device), None)
        return feats.detach().cpu().numpy().astype(np.float32), state

    def clone_warm_state(self):
        st = self._warm_state
        if st is None:
            return None
        if isinstance(st, torch.Tensor):
            return st.clone()
        if isinstance(st, dict):
            out = {}
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.clone()
                elif isinstance(v, list):
                    out[k] = [t.clone() for t in v]
                else:
                    out[k] = self._clone_nested(v)
            return out
        return st

    @staticmethod
    def _clone_nested(obj):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        if isinstance(obj, dict):
            return {k: HarkEngine._clone_nested(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [HarkEngine._clone_nested(x) for x in obj]
        return obj

    def enroll(
        self,
        phrase: str,
        language: str = "en",
        name: str | None = None,
        sensitivity: str | float = "balanced",
        threshold_override: float | None = None,
        multi_pronunciation: bool = True,
    ) -> EnrolledPhrase:
        if language not in ("en", "de"):
            raise ValueError(f"Unsupported language '{language}' (expected 'en' or 'de')")
        cfg = resolve_sensitivity(sensitivity, threshold_override=threshold_override)

        if language == "en" and multi_pronunciation:
            s1_seqs = encode_multi(phrase, "en", self.s1_vocab, self.s1_text_ver, "espeak-v1")
            s2_seqs = encode_multi(phrase, "en", self.s2_vocab, self.s2_text_ver, "espeak-v1")
        elif language == "de" and multi_pronunciation:
            s1_seqs = [encode(phrase, "de", self.s1_vocab, self.s1_text_ver, "espeak-v1")]
            s2_seqs = [encode(phrase, "de", self.s2_vocab, self.s2_text_ver, "espeak-v1")]
            s1_lex = encode(phrase, "de", self.s1_vocab, self.s1_text_ver, "de-lexicon-v1")
            s2_lex = encode(phrase, "de", self.s2_vocab, self.s2_text_ver, "de-lexicon-v1")
            if s1_lex not in s1_seqs:
                s1_seqs.append(s1_lex)
            if s2_lex not in s2_seqs:
                s2_seqs.append(s2_lex)
        else:
            s1_seqs = [encode(phrase, language, self.s1_vocab, self.s1_text_ver, "espeak-v1")]
            s2_seqs = [encode(phrase, language, self.s2_vocab, self.s2_text_ver, "espeak-v1")]

        inv = {idx: sym for sym, idx in self.s1_vocab.items()}
        ipa_variants = tuple("".join(inv[t] for t in seq) for seq in s1_seqs)
        min_len = min(len(seq) for seq in s1_seqs)
        short_guard = cfg["short_phrase_bonus"] if min_len <= 5 else 0.0
        temp, offset = LANGUAGE_CALIBRATION[language]

        enrolled = EnrolledPhrase(
            name=name or slugify_phrase(phrase, language),
            phrase=phrase.strip(),
            language=language,
            ipa_variants=ipa_variants,
            stage1_token_sets=tuple(tuple(int(t) for t in seq) for seq in s1_seqs),
            stage2_token_sets=tuple(tuple(int(t) for t in seq) for seq in s2_seqs),
            s1_threshold=float(cfg["s1_threshold"]),
            v_weight=float(cfg["v_weight"]),
            combined_threshold=round(float(cfg["combined_threshold"] + short_guard), 4),
            temperature=temp,
            offset=offset,
        )
        LOGGER.info(
            'Enrolled phrase="%s" lang=%s name=%s ipa=%s threshold=%.2f',
            enrolled.phrase,
            enrolled.language,
            enrolled.name,
            ["/" + v + "/" for v in enrolled.ipa_variants],
            enrolled.combined_threshold,
        )
        return enrolled

    def create_stream(
        self,
        targets: Sequence[EnrolledPhrase],
        cooldown_seconds: float = 1.2,
        lookahead_frames: int = 16,
        on_near_miss: Callable[[dict], None] | None = None,
    ) -> "HarkStream":
        return HarkStream(
            engine=self,
            targets=targets,
            cooldown_seconds=cooldown_seconds,
            lookahead_frames=lookahead_frames,
            on_near_miss=on_near_miss,
        )


class HarkStream:
    """Stateful streaming session for one audio stream (supports 1..N enrolled phrases)."""

    def __init__(
        self,
        engine: HarkEngine,
        targets: Sequence[EnrolledPhrase],
        cooldown_seconds: float = 1.2,
        lookahead_frames: int = 16,
        max_ring_frames: int = 350,
        on_near_miss: Callable[[dict], None] | None = None,
    ):
        if not targets:
            raise ValueError("At least one EnrolledPhrase target is required")
        self.engine = engine
        self.targets = list(targets)
        self.cooldown_frames = max(1, int(round(cooldown_seconds / 0.02)))
        self.lookahead_frames = max(4, int(lookahead_frames))
        self.max_ring_frames = max(120, int(max_ring_frames))
        self.on_near_miss = on_near_miss
        self.reset()

    def reset(self) -> None:
        self.frontend = FeatureStream(self.engine.bins)
        self.state = self.engine.clone_warm_state()
        self.global_frame = 0
        warm_f = self.engine._warm_feats.copy()
        self.ring_start_frame = -len(warm_f)
        self.feat_ring = warm_f
        self.audio_ring = np.zeros(0, dtype=np.float32)
        self.max_audio_samples = RATE * 6
        self.matchers: dict[str, list[Matcher]] = {
            t.name: [
                Matcher(
                    np.asarray(tok, dtype=np.int64),
                    threshold=t.s1_threshold,
                    cooldown=0.45,
                    **STAGE1_GATES,
                )
                for tok in t.stage1_token_sets
            ]
            for t in self.targets
        }
        self.pending: dict[str, PendingCandidate] = {}
        self.last_emitted_frame: dict[str, int] = {t.name: -100000 for t in self.targets}

    def push_pcm16(self, pcm_bytes: bytes) -> list[dict]:
        if not pcm_bytes:
            return []
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        return self.push(samples)

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> list[dict]:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []
        rms = float(np.sqrt(np.mean(np.square(x)) + 1e-12))
        rms_dbfs = round(20.0 * math.log10(max(rms, 1e-6)), 1)

        self.audio_ring = np.concatenate([self.audio_ring, x])
        if self.audio_ring.size > self.max_audio_samples:
            self.audio_ring = self.audio_ring[-self.max_audio_samples :]

        feats = self.frontend.push(x)
        if len(feats) == 0:
            return []

        feats_np = feats.detach().cpu().numpy().astype(np.float32)
        self.feat_ring = np.concatenate([self.feat_ring, feats_np], axis=0)
        if self.feat_ring.shape[0] > self.max_ring_frames:
            drop = self.feat_ring.shape[0] - self.max_ring_frames
            self.feat_ring = self.feat_ring[drop:]
            self.ring_start_frame += drop

        lp_t, self.state = self.engine.stage1(feats[None].to(self.engine.device), self.state)
        raw_lp = lp_t[0].detach().cpu().numpy().astype(np.float32)

        calibrated_cache: dict[tuple[float, float], np.ndarray] = {}
        for target in self.targets:
            cal_key = (target.temperature, target.offset)
            cal_lp = calibrated_cache.get(cal_key)
            if cal_lp is None:
                cal_lp = calibrate_posteriors(raw_lp, target.temperature, target.offset)
                calibrated_cache[cal_key] = cal_lp

            t_matchers = self.matchers[target.name]
            for local_i in range(len(cal_lp)):
                frame_idx = self.global_frame + local_i
                frame_lp = cal_lp[local_i]
                for m in t_matchers:
                    ev = m.push(frame_lp)
                    if ev is not None:
                        start_frame = int(round(ev["start"] / 0.02))
                        s1_score = float(ev["score"])
                        prev = self.pending.get(target.name)
                        if prev is None or s1_score > prev.s1_score:
                            self.pending[target.name] = PendingCandidate(
                                target=target,
                                end_frame=frame_idx,
                                start_frame=start_frame,
                                s1_score=s1_score,
                                rms_dbfs=rms_dbfs,
                            )

        self.global_frame += len(raw_lp)
        return self._evaluate_ready(force=False)

    def flush(self) -> list[dict]:
        return self._evaluate_ready(force=True)

    @torch.no_grad()
    def _evaluate_ready(self, force: bool = False) -> list[dict]:
        if not self.pending:
            return []
        ready_names = [
            name
            for name, cand in self.pending.items()
            if force or self.global_frame >= cand.end_frame + self.lookahead_frames
        ]
        if not ready_names:
            return []

        queries = []
        cand_list: list[PendingCandidate] = []
        for name in ready_names:
            cand = self.pending.pop(name)
            if cand.end_frame - self.last_emitted_frame[name] <= self.cooldown_frames:
                continue
            local_end = cand.end_frame - self.ring_start_frame
            local_start = cand.start_frame - self.ring_start_frame
            if local_end < 2 or self.feat_ring.shape[0] < 6:
                continue
            q = (local_end, max(0, local_start), cand.target.stage2_token_sets)
            queries.append(q)
            cand_list.append((cand, q))

        if not queries:
            return []

        scores = score_trigger_queries(
            self.engine.verifier,
            self.feat_ring,
            queries,
            is_features=True,
            device=self.engine.device,
        )

        detections: list[dict] = []
        for cand, q in cand_list:
            s2_logit = float(scores.get(q, -20.0))
            vw = cand.target.v_weight
            combined = (1.0 - vw) * cand.s1_score + vw * s2_logit
            info = {
                "name": cand.target.name,
                "phrase": cand.target.phrase,
                "language": cand.target.language,
                "time": round(0.025 + cand.end_frame * 0.02, 3),
                "start": round(cand.start_frame * 0.02, 3),
                "s1_score": round(cand.s1_score, 3),
                "s2_logit": round(s2_logit, 3),
                "combined_score": round(combined, 3),
                "threshold": cand.target.combined_threshold,
                "rms_dbfs": cand.rms_dbfs,
            }
            if combined >= cand.target.combined_threshold:
                self.last_emitted_frame[cand.target.name] = cand.end_frame
                detections.append(info)
            elif self.on_near_miss is not None:
                self.on_near_miss(info)
        return detections
