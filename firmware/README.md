# Hark satellite firmware

Looking to flash your device? Use the [web installer](https://pioborgelt.github.io/hark/firmware/) and the [setup guide](../docs/ESP32_SATELLITE.md).

This folder has the ESPHome configs, patch, manifest script, and web installer for [issue #1](https://github.com/pioborgelt/hark/issues/1): continuous mic streaming from Voice PE and ReSpeaker Lite to Home Assistant so Hark can handle wake word detection.

| Path | Contents |
| --- | --- |
| `voice-pe/` | Home Assistant Voice PE (`26.11.0` + Hark streaming switch). See [voice-pe/README.md](voice-pe/README.md). |
| `respeaker/` | ReSpeaker Lite (formatBCE `2026.11.1` + Hark streaming switch) plus `hark-respeaker-test.yaml` with HA simulation buttons. |
| `patch_voice_assistant.py` | Patch for ESPHome's `voice_assistant` component (stale `RUN_END` fix). |
| `site/index.html` | ESP Web Tools installer at [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/). |
| `make_manifest.py` | Builds `manifest.json` for ESP Web Tools and ESPHome `http_request` OTA updates. |

## How it works

Both targets share the same `hark-streaming.yaml` state machine. It adds a **Wake word engine location** select (`On device` / `In Home Assistant`). Left on `On device`, behavior is unchanged.

In **In Home Assistant** mode:
- Audio streams continuously, while music stays at full volume until HA confirms a wake word and un-ducks after the reply.
- Center button, mute switch, timers, and the local "stop" word work like stock. Built-in wake words ("Okay Nabu", etc.) are ignored.
- Errors back off (`5s, 10s, 20s, 40s, 60s`) instead of looping tightly.
- Diagnostic entities (`Voice phase`, `Streaming status`, `Streaming to Home Assistant`, `Music ducked`, `In conversation`, `Last voice error`) expose the state in HA.

### `voice_assistant` patch

Home Assistant sends pipeline events (`RUN_START` ... `RUN_END`) over one TCP stream without run IDs. If the device cancels a run and HA doesn't process the stop before the next start arrives, HA closes the old run right as the new one begins, so the old `RUN_END` arrives before the new `RUN_START`. Unpatched ESPHome treats that as the end of the new run and drops the mic without telling HA, wedging every following start.

`patch_voice_assistant.py` adds a `run_started_` guard so a `RUN_END` arriving before `RUN_START` is ignored. Applied to upstream `esphome/esphome@92723c9` in both `voice-pe/` and `respeaker/`.

## Building

Pinned to ESPHome `2026.11.0b4`:

```bash
pip install "esphome==2026.11.0b4"

cd firmware/voice-pe
esphome -s hark_fw_version 26.11.0-dev compile home-assistant-voice.hark.yaml

cd ../respeaker/config
esphome -s hark_fw_version 2026.11.1-dev compile hark-respeaker.yaml
```

Pushing a tag like `firmware-hark3` triggers `.github/workflows/firmware.yml`, builds both targets, creates a GitHub prerelease with the binaries, and dispatches `pages.yml` to update the installer and OTA manifests.

## Test status

- **ReSpeaker Lite (XIAO ESP32-S3, XMOS 1.1.0):** tested end-to-end (browser flash, Improv Wi-Fi, HA discovery, Hark/Whisper/Piper round-trip, button/mute/stop/timer cases, backoff, restarts, OTA update).
- **Voice PE (16 MB):** builds cleanly with bootloader and partition table byte-identical to official `26.11.0`; not yet tested on physical hardware (especially audible music ducking). 8 MB Voice PE isn't built yet.
