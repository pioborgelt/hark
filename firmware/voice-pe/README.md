# Voice PE Firmware (Hark Streaming Switch)

> **Status: untested on physical Voice PE hardware (16 MB model only).**
> It builds cleanly and matches the official `26.11.0` bootloader and partition layout byte-for-byte. The exact same `hark-streaming.yaml` state machine has been verified end-to-end on a ReSpeaker Lite (same XMOS front end, same DAC, same button and ducking logic) — wake word through Hark, button handling, stop word, mute, backoff when Hark stops, and restarts. The one thing not tested yet is audible music ducking while music plays over Wi-Fi.
>
> Only for the standard **16 MB Voice PE** (the 8 MB flash variant isn't built yet). Feedback in [issue #1](https://github.com/pioborgelt/hark/issues/1) is very welcome.

Based on official Voice PE firmware **26.11.0 (beta)**, compiled with ESPHome `2026.11.0b4` and upstream external components (`esphome/esphome@92723c9`, with a small stale-`RUN_END` fix in `voice_assistant` — see [firmware/README.md](../README.md)).

## Install & setup

Full walkthrough: [docs/ESP32_SATELLITE.md](../../docs/ESP32_SATELLITE.md). Short version:

1. Connect the Voice PE via USB-C to a computer with Chrome or Edge.
2. Open [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/), pick **Home Assistant Voice PE**, and click **Install Hark firmware**. USB flashing wipes Wi-Fi settings, so the page prompts for your Wi-Fi at the end (or use Bluetooth setup from the HA app).
3. Add the device in Home Assistant if needed, attach Hark to its assistant (**Add streaming wake word** in the three-dot menu under Settings > Voice assistants), and make sure the Voice PE's **Assistant** points to it.
4. Set **Wake word engine location** to **In Home Assistant**.

Future Hark updates show up as normal firmware updates in Home Assistant. To go back to stock anytime, flash from the [official Voice PE installer](https://esphome.github.io/home-assistant-voice-pe/).

## What would help most to test

Playing music on the Voice PE while testing is the main thing I couldn't test on the ReSpeaker Lite:

- Music playing, nobody speaks: volume stays normal.
- Say your wake word: music ducks right away, LED ring reacts, volume returns after the reply.
- Center button: stops a ringing timer / active conversation / answer / announcement, pauses music, or starts the assistant directly — and resumes listening afterwards.
- Mute switch on: `Streaming status` shows `off (muted)`; off again: streaming resumes.
- Say "Stop" during a long answer: cuts it off and goes back to listening.
- Stop the Hark app: backs off cleanly without crashing; start Hark again: recovers on its own.
- Restart the Voice PE or Home Assistant: streaming resumes automatically.
- Switch back to **On device**: "Okay Nabu" behaves just like stock.

If anything acts weird, the device log (ESPHome device page > Logs) around that timestamp is the most helpful thing to share.

## Files

- `home-assistant-voice.hark.yaml` — build entry point (replaces `home-assistant-voice.factory.yaml`, wires in `hark-streaming.yaml` and the Hark OTA manifest).
- `home-assistant-voice.yaml` — upstream file with Hark hooks marked `Hark:` (full diff in `hark-voice-pe.diff`).
- `hark-streaming.yaml` — streaming state machine, identical to the ReSpeaker build.
- `components/voice_assistant/` — upstream `esphome/esphome@92723c9` with the stale-`RUN_END` patch (`../patch_voice_assistant.py`).
