# Voice PE firmware (Hark streaming switch)

> **Untested on physical Voice PE hardware (16 MB model only).**
> Builds cleanly with bootloader and partition table byte-identical to official `26.11.0`. The same `hark-streaming.yaml` logic is tested end-to-end on a ReSpeaker Lite (same XMOS front end, DAC, button and ducking logic), except audible music ducking over Wi-Fi. 8 MB Voice PE variants aren't supported yet. Please report results in [issue #1](https://github.com/pioborgelt/hark/issues/1).

Based on official Voice PE `26.11.0` (ESPHome `2026.11.0b4`, upstream `esphome/esphome@92723c9` plus the stale-`RUN_END` fix in `../patch_voice_assistant.py`).

## Install

Full guide: [docs/ESP32_SATELLITE.md](../../docs/ESP32_SATELLITE.md).

1. Connect the Voice PE via USB-C to a computer with Chrome or Edge.
2. Flash from [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/) (erases Wi-Fi, prompts at the end).
3. In Home Assistant, attach Hark to the device's assistant (**Add streaming wake word** in the assistant's three-dot menu).
4. On the device page, set **Wake word engine location** to **In Home Assistant**.

Updates arrive through Home Assistant as normal firmware updates. To return to stock, flash from the [official Voice PE installer](https://esphome.github.io/home-assistant-voice-pe/).

## What to test

Playing music while testing is the main thing not yet verified:

- Music playing, nobody speaks: volume stays normal.
- Wake word: music ducks right away, LED ring reacts, volume returns after the reply.
- Center button: stops timer / conversation / answer / announcement, pauses music, or starts the assistant directly, then resumes listening.
- Mute switch: `Streaming status` shows `off (muted)`, resumes when unmuted.
- "Stop" during a long answer stops it and resumes listening.
- Stopping Hark backs off cleanly and recovers when Hark starts again.
- Restarting Voice PE or Home Assistant resumes streaming on its own.
- Switching back to **On device** restores stock "Okay Nabu" behavior.

If anything breaks, the device log (ESPHome device page > Logs) around that timestamp helps most.

## Files

- `home-assistant-voice.hark.yaml`: build entry point (includes `hark-streaming.yaml` and Hark OTA manifest).
- `home-assistant-voice.yaml`: upstream config with Hark hooks marked `Hark:` (diff in `hark-voice-pe.diff`).
- `hark-streaming.yaml`: streaming state machine, shared with ReSpeaker Lite.
- `components/voice_assistant/`: `esphome/esphome@92723c9` with the stale-`RUN_END` patch.
