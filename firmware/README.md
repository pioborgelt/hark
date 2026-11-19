# Hark Satellite Firmware

> Just want to flash your Voice PE or ReSpeaker Lite? Use the [web installer](https://pioborgelt.github.io/hark/firmware/) and the [step-by-step guide](../docs/ESP32_SATELLITE.md).

This directory holds the ESPHome firmware sources, CI manifests, and web installer for [issue #1](https://github.com/pioborgelt/hark/issues/1): letting ESP32 voice satellites stream their microphone to Home Assistant so Hark can run zero-shot wake word detection there.

## Layout

| Path | What's in it |
| --- | --- |
| `voice-pe/` | Home Assistant Voice PE firmware (official `26.11.0` + Hark streaming switch). See [voice-pe/README.md](voice-pe/README.md). |
| `respeaker/` | ReSpeaker Lite firmware (formatBCE `2026.11.1` + Hark streaming switch), plus `hark-respeaker-test.yaml` with HA simulation buttons. |
| `patch_voice_assistant.py` | Reproducible patch for ESPHome's `voice_assistant` component (stale `RUN_END` fix, explained below). |
| `site/index.html` | ESP Web Tools installer published to [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/). |
| `make_manifest.py` | Generates `manifest.json` for both ESP Web Tools (USB flash) and ESPHome's `http_request` OTA updater. |

## How it works

Both devices share the exact same streaming state machine (`hark-streaming.yaml`). It adds a **Wake word engine location** select entity (`On device` / `In Home Assistant`) without touching stock behavior when left on `On device`.

When switched to **In Home Assistant**:

- **Deferred ducking:** the satellite streams audio continuously, but keeps media playback at full volume until Home Assistant signals a real wake-word match, then un-ducks when the reply finishes (with a safety reset in case a run aborts mid-way).
- **Full control parity:** center button, mute switch, timers, and the local "stop" word work like stock. On-device wake words ("Okay Nabu", etc.) are ignored while streaming so they don't collide with Hark.
- **Backoff on errors:** if Hark or Home Assistant is down, reconnect retries back off (`5 s → 10 s → 20 s → 40 s → 60 s`) instead of spinning in a tight loop.
- **Diagnostics:** exposes `Voice phase`, `Streaming status`, `Streaming to Home Assistant`, `Music ducked`, `In conversation`, and `Last voice error` in Home Assistant so you can see exactly what the state machine is doing.

### The `voice_assistant` patch

While testing continuous streaming on real hardware, pressing the center button to interrupt a conversation repeatedly wedged the next run.

The root cause is in ESPHome's `voice_assistant` component: Home Assistant sends pipeline events (`RUN_START` … `RUN_END`) sequentially over one TCP stream, without tagging them with a run ID. If the device stops a run and Home Assistant doesn't process the stop before the next start arrives, HA closes the old run right as the new one starts — so the stale `RUN_END` arrives *before* the new `RUN_START`. Unpatched ESPHome treats that as the end of the new run and silently stops the microphone without telling HA, leaving HA's run open and repeating the cycle on every subsequent start.

`patch_voice_assistant.py` adds a single `run_started_` flag so any `RUN_END` arriving before `RUN_START` of the current request is ignored. Both `voice-pe/components/voice_assistant` and `respeaker/components/voice_assistant` carry this patch on top of upstream commit `92723c9`.

## Building from source

Pinned to ESPHome `2026.11.0b4` (matching `.github/workflows/firmware.yml`):

```bash
pip install "esphome==2026.11.0b4"

# Voice PE (16 MB)
cd firmware/voice-pe
esphome -s hark_fw_version 26.11.0-dev compile home-assistant-voice.hark.yaml

# ReSpeaker Lite
cd ../respeaker/config
esphome -s hark_fw_version 2026.11.1-dev compile hark-respeaker.yaml
```

## Releases and OTA updates

Pushing a tag like `firmware-hark3` triggers `.github/workflows/firmware.yml`:

1. Compiles both targets (`.factory.bin` for USB, `.ota.bin` for Home Assistant updates).
2. Creates a GitHub prerelease holding the binaries (binaries are never committed to Git).
3. Dispatches `.github/workflows/pages.yml`, which pulls the newest `firmware-hark*` release, runs `make_manifest.py`, and publishes `/firmware/<device>/manifest.json` alongside `site/index.html`.

Both firmwares point their `update` entity at `https://pioborgelt.github.io/hark/firmware/<device>/manifest.json`, so OTA updates stay on the Hark track and keep the switch.

## Hardware test status

- **ReSpeaker Lite (XIAO ESP32-S3, XMOS 1.1.0):** verified end-to-end on real hardware — fresh browser USB flash (ESP Web Tools `10.5.2`), Improv Wi-Fi setup, HA auto-discovery, full Hark → Whisper → Piper round-trip, button/mute/stop-word/timer handling, error backoff, device + HA restarts, and OTA upgrade (`hark2 → hark3`).
- **Home Assistant Voice PE (16 MB):** passes `esphome config`, builds cleanly, and produces a bootloader and partition table byte-identical to official `26.11.0`. Not yet tested on physical Voice PE hardware (audible music ducking in particular still needs a real-device check).
- **Not yet covered:** the 8 MB flash variant of the Voice PE (`home-assistant-voice.8mb.yaml` upstream).
