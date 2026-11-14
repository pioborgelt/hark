<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/hark-wordmark-white.svg">
    <img alt="Hark" src="docs/brand/hark-wordmark-black.svg" width="300">
  </picture>
</p>

**A zero-shot wake word engine for your own Jarvis: type a phrase, done. No recordings, no training.**

I originally built Hark for my own voice assistant, PAI. Its old wake-word engine (sherpa-onnx) is solid in English but basically deaf to "Okay PAI" in German, and every alternative wanted me to either record myself 50 times or burn two hours of Colab per word. Somehow that "quick weekend fix" turned into 220+ training runs, so I figured I might as well open-source it.

> **Alpha Preview (v0.3.0-alpha).** Runs on Linux x86_64 and ARM64 (Raspberry Pi 4/5, N100, mini PCs). It isn't optimised yet; faster builds, and maybe one day an ESP32 version, are on the list.

---

## Install (Home Assistant & Python)

**Before you start:** Hark only provides the wake word. Your voice assistant also needs
speech-to-text and text-to-speech (e.g. the **Whisper** or **Speech-to-Phrase** and **Piper**
apps), and a device that streams audio to Home Assistant. Stuck? See the [FAQ](docs/FAQ.md)!

### Option 1: Home Assistant OS app (recommended)

1. **Settings → Apps → Install app → ⋮ (top right) → Repositories → Add** →
   `https://github.com/pioborgelt/hark`
2. Search the store for **hark**, open **Hark Wake Word** → **Install**.
3. **Configuration**: add phrases like `Hallo Kühlschrank:de` or `Hey Jarvis:en` → **Save** →
   **Info → Start**.
4. **Settings → Devices & services → Discovered → Hark Wake Word → Add.**
5. **Settings → Voice assistants →** your assistant **→ ⋮ → Add streaming wake word** →
   engine **hark**, wake word = your phrase.

6. Say the phrase, pause for a beat, then your command.

(If the menu points differ here or you can't follow the guide, please [open an issue](https://github.com/pioborgelt/hark/issues/new). I try to keep hark as accessible as possible.)

No host port is opened, so it runs next to the openWakeWord app.

### Option 2: Docker + Home Assistant (Container, Core, or a separate machine)

Run the Wyoming server with whatever English or German phrases you want (separated by `;`):

```bash
docker run -d --name wyoming-hark --restart unless-stopped \
  -p 10400:10400 \
  -v ./hark-data:/data \
  -e HARK_PHRASES="Hallo Kühlschrank:de;Hey Jarvis:en" \
  -e HARK_SENSITIVITY="balanced" \
  ghcr.io/pioborgelt/hark:alpha
```

*(Or clone this repo and run `docker compose up -d`. The ~31 MB model weights download
automatically on first start into `./hark-data`.)*

Then in Home Assistant:
1. **Settings → Devices & services → Add integration → Wyoming Protocol** → host = the IP of
   the machine running the container, port **10400**.
2. **Settings → Voice assistants →** your assistant **→ ⋮ → Add streaming wake word** →
   **hark** + your phrase.
3. *(Voice PE / ESPHome devices: set the wake word location from "On device" to "In Home
   Assistant".)*

### Option 3: Python CLI & API

Needs `espeak-ng` and `libsndfile1` (`sudo apt install espeak-ng libsndfile1`).

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"
pip install "hark-ww[wyoming,microphone] @ git+https://github.com/pioborgelt/hark.git"

# Test live on your mic with custom German + English phrases at the same time
hark listen -p "Okay Haus:de" -p "Hey Jarvis:en"

# Or start the Home Assistant Wyoming server directly
wyoming-hark --uri tcp://0.0.0.0:10400 --phrase "Okay Haus:de" --phrase "Hey Jarvis:en"
```

<details>
<summary><b>Python API (6 lines)</b></summary>

```python
from hark.streaming import HarkEngine

engine = HarkEngine(threads=2)
stream = engine.create_stream([
    engine.enroll("Okay Haus", language="de"),
    engine.enroll("Hey Jarvis", language="en"),
])

for hit in stream.push(audio_float32_16khz):
    print("Triggered:", hit["phrase"])
```
</details>

> **Quick tip on picking phrases:** 3–4 syllables with clear consonants ("Okay Haus", "Hey Jarvis", "Hallo Computer", "Smart Mirror") work way better than ultra-short words like "Ey" or "Yo" — unless you enjoy talking to your toaster by accident.

---

## Does it actually work? (Benchmarks)

Every Hark score below is **100% zero-shot**, and the model has never seen audio or synthetic TTS of the target wake word. I just typed the text in and streamed ~25–28 hours of continuous speech, music, and real background noise through it without resetting state.

### 1. Picovoice 24h Wake-Word Benchmark (Loud Background Noise)

Tested across all 6 standard benchmark words (*alexa, computer, jarvis, smart mirror, snowboy, view glass*) mixed with real noise at 10 dB SNR, capped at a strict **0.1 false alarms per hour** (~2 false triggers in 24 hours):

![24h Wake-Word Benchmark at 10 dB Noise](docs/charts/wwb_10db_pareto.png)

| Engine | How you add a new word | Languages | Hit Rate (10 dB Noise) | Hit Rate (Clean) |
|---|---|---|---:|---:|
| **Hark v0.3 (Open-Source)** | **Type text (instant)** | **EN + DE** | **78.1%** | **91.0%** |
| Kitt.ai Snowboy | Record voice samples | Any | 68.1% | – |
| sherpa-onnx (Zipformer 3.3M) | Type text (instant) | EN only | 66.0% | 77.4% |
| CMU PocketSphinx | Type text | EN | 48.0% | – |
| Vosk (small) | Type text | EN, DE | 32.7% | – |
| openWakeWord 0.6 | Hours of TTS training per word | Fixed EN | *63.3% (jarvis) / 0% (computer)* | – |
| Picovoice Porcupine v4 | Closed-source + Cloud API key | Proprietary | 97.3% | – |

 **+12 percentage points ahead of sherpa-onnx** (the strongest existing open-source zero-shot engine I could find), winning on 5 out of 6 words.

**Beats pre-trained openWakeWord single-word models zero-shot** on both *jarvis* (78.4% vs 63.3%) and *computer* (64.2% vs 0%, where openWakeWord blows past the false-alarm limit).

Closing in on commercial per-word engines without needing an account, API key, or per-device license. (We'll get there ;))

### 2. The test I actually built it for: "Okay PAI" in English & German

28 hours of real English speech, German podcasts and music as negatives, then "Okay PAI" / "Hey PAI" spoken clean, in reverberant rooms and over loud music. Fair warning: the positives here are **synthetic Piper TTS voices**, so treat this as "does German work at all" rather than "how good is it on your mic" (benchmark #1 above is the real-voice one).

| Language | Hark (0 false alarms in 28 h) | Hark (2 false alarms) | sherpa-onnx | Vosk |
|---|---:|---:|---:|---:|
| **English** | **85.9%** | **92.0%** | 81.6% | 48.2% (needs ≥4 FA) |
| **German** | **57.0%** | **62.8%** *(68.7% @ 5 FA)* | no German support | 5,276+ false alarms (rip) |

German is clearly the weaker language right now. If you have a few real recordings of a German wake word (yours, your kids', your Bavarian uncle's), that's the most useful thing you could send me.

<p align="center">
  <img src="docs/charts/pai_bench_de_pareto.png" width="49%" alt="German Benchmark" />
  <img src="docs/charts/pai_bench_en_pareto.png" width="49%" alt="English Benchmark" />
</p>

Full nerd tables and per-word numbers are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## How it works

Normal streaming wake-word models only look *backwards* in time so they stay fast. The catch: in a noisy room, "he collects a..." sounds identical to "Alexa" until you hear the syllable *after* it.

Hark splits the job into two tiny models ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)):

1. **Text → Phonemes:** `espeak-ng` turns your text into IPA phonemes (plus common US/UK/German pronunciation variants so you don't have to speak like a news anchor).
2. **Stage 1 — The Bouncer (8.7 MB, runs 24/7):** A tiny causal Conv-GRU scans incoming audio 25x faster than real-time on CPU (~4% load). Most of the hour, Stage 2 stays completely asleep.
3. **Stage 2 — The Double-Checker (22 MB, <2 ms per check):** When Stage 1 thinks it heard your phrase, Hark waits 0.3 seconds of extra context and runs a bidirectional phoneme cross-attention model forward *and* backward over that 2-second snippet. That kills >95% of look-alike false alarms for basically zero extra CPU.

---

## Alpha Heads-Up & Feedback

This is a really early alpha release. I mainly want to find out where it breaks on real microphones, weird room acoustics, and regional accents before polishing v0.4. Also, I'm building this in my free time, so keep your expectations relaxed: this isn't Porcupine (for now...).

- Runs on your Home Assistant server / Pi CPU (like `wyoming-openwakeword`), for now NOT directly on an ESP32 chip.
- If a phrase feels deaf or false-triggers when your washing machine beeps, check the container logs (Hark logs near-misses rejected by Stage 2) and [open an issue](https://github.com/pioborgelt/hark/issues/new) with your **phrase**, **language**, and **mic setup**.

## License

- **Code:** [MIT](LICENSE)
- **v0.3-alpha Weights** ([Hugging Face](https://huggingface.co/Pioborgelt/hark)): **CC-BY-NC-4.0** (free for personal use, Home Assistant setups, tinkering, and research). Trained on open speech/noise datasets (MLS, LibriSpeech, VoxPopuli, Common Voice, MUSAN) + synthetic speech generated with Google Gemini TTS.
