# Wake-Word Benchmark Results (Hark v0.3)

All Hark numbers below use **pure text enrollment** (`hark/text.py`) with **zero audio samples** of the target wake phrase and **uninterrupted continuous streaming inference** (`reset = 0`).

---

## 1. Picovoice 24-Hour Wake-Word Benchmark (`10 dB` DEMAND Noise & Clean)

Evaluated across all 6 official benchmark wake words (`alexa`, `computer`, `jarvis`, `smart mirror`, `snowboy`, `view glass`), ~24.8 hours of continuous background speech mixed with 17 DEMAND noise environments at `10 dB` SNR per word (~149 hours per SNR condition), at the official false-alarm ceiling of **`<= 0.1 FA / hour` (`<= 2.4 FA / 24 h`)**.

| Engine | Enrollment / Type | Languages | Hit Rate @ `10 dB` (`<= 2.4 FA/24h`) | Miss Rate @ `10 dB` | Hit Rate Clean (`99 dB`) |
|---|---|---|---:|---:|---:|
| **Hark v0.3 (`R221 + mp + R222`)** | **Zero-Shot Text (Open-Source)** | **EN + DE** | **`78.14 %`** *(75.41 % single global setting)* | **`21.86 %`** | **`91.01 %`** *(8.99 % miss)* |
| **Hark (`R217 + mp + R218`)** | **Zero-Shot Text (Open-Source)** | **EN + DE** | **`71.45 %`** | `28.55 %` | `85.08 %` |
| **Hark Stage-1 Only (`R221`)** | **Zero-Shot Text (Causal GRU)** | **EN + DE** | **`66.72 %`** | `33.28 %` | `80.31 %` |
| `Kitt.ai Snowboy` | Recorded Audio Templates | Template | `68.10 %` *(published)* | `31.90 %` | – |
| `sherpa-onnx` 3.3M (Zipformer KWS) | Zero-Shot Text (Open-Source) | EN only | `66.00 %` | `34.00 %` | – |
| `CMU PocketSphinx` 5.1.1 | Open-Vocab CMUdict (Open-Source) | EN | `48.00 %` *(published)* | `52.00 %` | – |
| `Vosk` 0.3.45 (`small-en-us-0.15`) | Open-Vocab ASR Grammar | EN, DE | `32.69 %` *(4/6 words exceed FA cap)* | `67.31 %` | – |
| `openWakeWord` 0.6.0 | Fixed Pre-Trained Single-Word ONNX | Fixed EN | `alexa` `87.9 %`, `jarvis` `63.3 %`, `computer` `0.0 %` | *(3/6 unavailable)* | – |
| `microWakeWord` 2.5.0 | Fixed Pre-Trained Single-Word TFLite | Fixed EN | `alexa` `89.5 %` | *(5/6 unavailable)* | – |
| `Picovoice Porcupine` v4 | Closed-Source Fixed Per-Word Models | Proprietary | `97.30 %` *(published)* | `2.70 %` | – |

### Per-Keyword Hit Rates at `10 dB` SNR (`<= 0.1 FA / hour`)

| Keyword | **Hark v0.3 (`10 dB`)** | **Hark v0.3 (`Clean`)** | `sherpa-onnx` 3.3M (`10 dB`) | Hark Stage-1 (`R221`) | `openWakeWord` 0.6 | `microWakeWord` 2.5 | `Vosk` 0.3.45 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smart mirror` | **`85.09 %`** *(+19.0 pp)* | **`92.95 %`** | 66.1 % | 87.8 % | n/a | n/a | 98.6 % |
| `snowboy` | **`83.54 %`** *(+14.2 pp)* | **`99.00 %`** | 69.3 % | 72.1 % | n/a | n/a | 97.5 % |
| `view glass` | **`79.20 %`** *(+22.1 pp)* | **`91.98 %`** | 57.1 % | 76.9 % | n/a | n/a | 0.0 % *(>2.4 FA)* |
| `alexa` | **`78.41 %`** | **`86.67 %`** | 82.9 % | 36.8 % | 87.9 % *(fixed)* | 89.5 % *(fixed)* | 0.0 % *(>2.4 FA)* |
| `jarvis` | **`78.39 %`** *(+20.8 pp)* | **`89.32 %`** | 57.6 % | 72.7 % | 63.3 % *(fixed)* | n/a | 0.0 % *(>2.4 FA)* |
| `computer` | **`64.23 %`** *(+0.9 pp)* | **`86.13 %`** | 63.3 % | 54.0 % | 0.0 % *(>2.4 FA)* | n/a | 0.0 % *(>2.4 FA)* |
| **6-Word Mean** | **`78.14 %`** | **`91.01 %`** | `66.0 %` | `66.7 %` | *partial (3/6)* | *partial (1/6)* | `32.7 %` |

---

## 2. PAI-Bench Bilingual Zero-Shot Benchmark (`"Okay PAI"` / `"Hey PAI"` vs. 28.2h Negative Streams)

Evaluated across clean speech, reverberant rooms, `10 dB` and `0 dB` music interference, and embedded continuous-speech contexts against **28.19 hours** of negative English speech, German speech, and full music tracks.

### PAI-Bench English (`820` positive trials)

| Engine | `0 FA` in 28.2h | `1 FA` | `2 FA` | `5 FA` | Notes |
|---|---:|---:|---:|---:|---|
| **Hark v0.3 (`R221 + R222`)** | `85.9 %` (`704/820`) | **`89.1 %` (`731`)** | **`92.0 %` (`754`)** | **`92.8 %` (`761/820`)** | `164/164` (`100 %`) Clean & `164/164` (`100 %`) Context |
| **Hark (`R214 + R218`)** | **`86.6 %` (`710/820`)** | `86.6 %` (`710`) | `86.6 %` (`710`) | `88.4 %` (`725/820`) | `164/164` (`100 %`) on Clean at `0 FA` |
| **Hark (`R214 + R222`)** | `83.3 %` (`683/820`) | `85.9 %` (`704`) | `87.9 %` (`721`) | `91.5 %` (`750/820`) | `164/164` (`100 %`) Clean |
| `sherpa-onnx` 3.3M | `81.6 %` (`669/820`) | – | – | – | English-only model |
| `Vosk` 0.3.45 (`small-en-us-0.15`) | *unreachable (`min 4 FA`)* | *unreachable* | *unreachable* | `48.2 %` (`395/820`) | Confidence saturates at `1.0` |
| `CMU PocketSphinx` 5.1.1 | `0.0 %` (`0/820`) | `0.0 %` | `0.0 %` | `0.0 %` | Negative FA score exceeds positives |

### PAI-Bench German (`1,020` positive trials)

| Engine | `0 FA` in 28.2h | `1 FA` | `2 FA` | `5 FA` | Notes |
|---|---:|---:|---:|---:|---|
| **Hark (`R214 + R222`)** | `64.8 %` (`661/1020`) | `65.5 %` (`668`) | **`69.3 %` (`707`)** | **`73.0 %` (`745/1020`)** | New German record at `2–5 FA` |
| **Hark (`R214 + R218`)** | **`65.7 %` (`670/1020`)** | **`67.0 %` (`683`)** | `67.5 %` (`689`) | `70.9 %` (`723/1020`) | `+21.3 pp` over Stage-1 alone at `0 FA` |
| **Hark v0.3 (`R221 + R222`)** | `57.0 %` (`581/1020`) | `60.9 %` (`621`) | `62.8 %` (`641`) | `68.7 %` (`701/1020`) | Universal EN+DE Stage-1 checkpoint |
| `Vosk` 0.3.45 (`small-de-0.15`) | *unreachable (`min 5,276 FA`)* | *unreachable* | *unreachable* | *unreachable* | Closed-lexicon OOV neighbor triggers |
| `sherpa-onnx` / `Porcupine` / `openWakeWord` | n/a | n/a | n/a | n/a | No zero-shot German text enrollment |
