# Hark Two-Stage Open-Vocabulary Wake-Word Architecture

Hark is a bilingual (**English + German**), zero-shot text-enrolled wake-word engine designed for continuous 16 kHz audio streams at strict false-alarm budgets (0 to 0.1 FA/hour).

## Overview

Any wake phrase (e.g. "Okay PAI", "Alexa", "Smart Mirror") is enrolled purely from text—no audio templates or per-phrase model training are required:

1. **Deterministic IPA Phonemization (hark/text.py):**
   - encode() and encode_multi() map arbitrary English or German text into IPA codepoint sequences using espeak-ng (piper-tts), optionally generating length-guarded multi-pronunciation variants (US/UK vowels, alveolar flap ɾ → t, final unstressed vowel realizations).
2. **Stage 1 — Always-On Causal Streaming Encoder + Confident-Frame CTC Matcher (hark/model.py, hark/decoder.py):**
   - **Frontend (hark/audio.py):** 64-bin 16 kHz log-mel filterbank (25 ms window, 10 ms hop).
   - **Encoder (AcousticModel):** Three causal 1D convolutions (stride 2 → 50 Hz frame rate) followed by a 2-layer causal GRU (hidden=384), trained on continuous 14-second noise/music/speech stream beds with carried recurrent state across boundaries.
   - **Streaming Decoder (decode_sets):** Monotonic CTC trellis matcher with confident-frame gating (min_blank_probability, min_frame_probability, mean_phone_probability, max_seconds_per_phone).
3. **Stage 2 — Event-Triggered Bidirectional Phoneme Verifier (hark/verifier.py):**
   - Invoked **only** when Stage 1 emits a candidate trigger (< 0.05 % duty cycle on continuous 24-hour audio).
   - **BiPhonemeVerifier:** Evaluates a short ~2.2 s local context window (left_ctx=16, right_ctx=14 frames at 50 Hz) with a 2-layer **bidirectional** GRU (hidden=384) seeing both pre-onset and post-offset coarticulation.
   - **Causal-Lag Span Alignment:** Compensates for Stage-1 causal GRU latency (span_shifts = ((-3, -1), (-4, 1))), computes exact length-normalized forward CTC forced-alignment log-likelihood over the target IPA sequence alongside phone-margin, min-phone, and boundary-blank features, and outputs a calibrated verification logit via a compact MLP head.
