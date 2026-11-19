# Setting up an ESP32 satellite for Hark

This guide covers getting a **Home Assistant Voice PE** or a **ReSpeaker Lite** talking to Hark. For installing the Hark app itself, start with the [app documentation](../hark_addon/DOCS.md).

> **Roadmap heads-up:** Right now the satellite streams its microphone to Home Assistant continuously and Hark listens there. The plan for a future release is to run Hark's tiny first stage directly on the ESP32, so the device only streams audio once it thinks it heard your phrase and Home Assistant runs the second-stage check. Less Wi-Fi traffic, less server load — and when that lands, this page gets a lot shorter.

## Why custom firmware?

Stock Voice PE and ReSpeaker Lite firmware only run wake word detection on the device itself — there is no switch to hand the audio stream off to Home Assistant.

The Hark firmware takes the upstream firmware for your board and adds one setting: **Wake word engine location**.

- **On device** (default): behaves identically to the stock firmware.
- **In Home Assistant**: streams the microphone to Home Assistant so Hark can listen for whatever phrase you typed.

It isn't just a quick config tweak — a few things had to be wired properly so the satellite still feels native:

- **Proper audio ducking:** music stays at normal volume while the device is listening, ducks only after Hark actually confirms the wake word, and fades back up once the reply ends.
- **All controls still work:** center button (stop a timer, cancel an answer or announcement, pause music, or push-to-talk), mute switch (cuts the stream completely), timers, and the local "stop" word behave just like stock. Built-in wake words ("Okay Nabu", etc.) stay quiet in this mode.
- **Clean recovery:** if Hark or Home Assistant is unreachable, the device backs off gracefully (5 s up to 1 min) instead of hammering your network. It also includes a small patch to ESPHome's `voice_assistant` component for an upstream race condition where a late `RUN_END` from a cancelled run could kill the next run on flaky Wi-Fi.
- **Safe to try, easy to undo:** bootloader and partition table match upstream byte-for-byte, future Hark updates arrive through Home Assistant's normal update button, and going back to stock over USB takes 30 seconds.

## Supported devices

- **Home Assistant Voice PE (16 MB flash — the standard model):** based on official firmware `26.11.0`. It uses the exact same streaming state machine as the ReSpeaker build, though it hasn't been run on a physical Voice PE yet — especially music ducking needs a real-world check. Please drop a note in [issue #1](https://github.com/pioborgelt/hark/issues/1) on how it goes! *(8 MB Voice PE variants aren't built yet.)*
- **ReSpeaker Lite (XIAO ESP32-S3):** based on formatBCE's `2026.11.1`. Tested end-to-end on real hardware.

## What you need

- The device and a USB-C **data** cable (plenty of charging cables leave out the data lines).
- A computer with Chrome or Edge. Firefox, Safari and phones don't support Web Serial.
- Home Assistant with the Hark app running and at least one phrase configured (steps 1–3 of the [app documentation](../hark_addon/DOCS.md)).

ReSpeaker Lite extras:

- **A 2.4 GHz U.FL antenna on the XIAO.** The XIAO ESP32-S3 has no onboard antenna, just a tiny U.FL socket. Continuous audio streaming (~256 kbit/s) needs a clean signal, so plug an antenna in.
- **XMOS firmware 1.1.0 or newer (48 kHz I2S).** Fresh boards ship with USB audio firmware on the XMOS chip, which leaves the ESP32 unable to read the mics. If you've already run formatBCE's firmware you're good; otherwise flash the [48 kHz I2S firmware](https://wiki.seeedstudio.com/xiao_respeaker/#flash-the-i2s-firmware) first via the USB port on the main board, not the XIAO (step 3 of [formatBCE's guide](https://github.com/formatBCE/Respeaker-ESPHome-integration/blob/main/readme/lite/README.md)).

## 1. Flash the firmware (once, over USB)

1. Plug the device into your computer. On the ReSpeaker Lite, use the USB-C port on the small **XIAO** board.
2. Open [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/) in Chrome or Edge.
3. Pick your device and click **Install Hark firmware**.
4. In the browser's serial port picker, select **USB JTAG/serial debug unit** and click **Connect**.
5. Click **Install**, confirm **Install**, and wait ~30 seconds.
6. When it finishes, click **Next**, pick your Wi-Fi network, enter the password, and click **Connect**.
7. Once it says "Device connected to the network", click **Skip**. You can unplug the device and put it wherever it lives.

Flashing over USB always wipes the device (including Wi-Fi credentials), which is why step 6 asks for them. Normal updates in Home Assistant afterwards keep all your settings.

## 2. Add it to Home Assistant

1. Go to Settings > Devices & services (Geräte & Dienste). Under **Discovered** (Entdeckt), you should see your device (for example `ReSpeaker Lite f6dd3c`). Click **Add**, then **Submit**.
2. Home Assistant launches the "Voice Satellite setup" wizard and asks you to say "Okay Nabu". You don't need that for Hark — click **Skip** in the top right. *(On the ReSpeaker Lite, the wizard may also claim the mic is muted when it isn't. Skip works anyway.)*
3. Assign an area (or skip).
4. Pick your processing mode (for a local setup, **Full local processing** with Whisper and Piper) and click **Next**.
5. On the last screen, note which assistant the device is assigned to (e.g. `Full local assistant`) and click **Done**.

## 3. Attach Hark to that assistant

1. Go to Settings > Voice assistants (Sprachassistenten) and open the assistant from step 2.5.
2. Click the three-dot menu in the top right of the dialog and pick **Add streaming wake word** (Streaming-Aktivierungswort hinzufügen).
3. Scroll down to the new section, pick **hark** as the engine and your phrase as the wake word, then click **Update**.

## 4. Flip the switch

1. Open the device under Settings > Devices & services > ESPHome > your device.
2. Under **Configuration**, set **Wake word engine location** to **In Home Assistant**.
3. A second later, **Streaming status** (under Diagnostic) should read `streaming, waiting for wake word`, and the Hark app log should show `Audio stream started, listening for: ...`.

Say your wake phrase, pause a beat, and give your command.

## Updates and going back

You only need USB once. New Hark firmware builds show up as regular firmware updates on the device page in Home Assistant. They pull from the Hark install page rather than upstream, so updating won't wipe your switch.

Want to go back to stock? Flash over USB the exact same way from the [official Voice PE installer](https://esphome.github.io/home-assistant-voice-pe/) or [formatBCE's ReSpeaker Lite installer](https://formatbce.github.io/Respeaker-ESPHome-integration/lite/).

## Troubleshooting

**Chrome doesn't show the device.** Swap the USB cable first — power-only cables are the #1 culprit. On a ReSpeaker Lite, you can also unplug it, hold the **BOOT** button on the XIAO while plugging it back in, and try again. On Linux, make sure your user can access serial ports: `sudo usermod -aG dialout $USER`, then log out and back in.

**Flashing fails halfway with "Failed to write compressed data to flash".** Reload the page and retry. If it keeps failing, plug directly into the computer instead of a USB hub.

**Nothing shows up in the Hark log when you speak.** Check three things: **Wake word engine location** is set to `In Home Assistant`, the device's **Assistant** points to the assistant where you added Hark (satellites don't follow your "preferred" assistant), and that assistant has speech-to-text configured.

**Hark reacts sluggishly or misses obvious wake words.** Almost always Wi-Fi packet loss. Check the Hark app log: it prints `Listening: 60 s audio received` once a minute. If those lines show up much slower than once per minute, audio frames are being dropped on the way. On a ReSpeaker Lite check the antenna; otherwise move the satellite closer to your access point.

**The first word of my command gets clipped.** Pause briefly after your wake phrase. Hark's second stage looks ~0.3 seconds past the end of the phrase to rule out look-alikes before handing off to speech-to-text.

**ReSpeaker Lite mute / user buttons don't respond.** Those two pins need the hardware solder jumpers (`USR→D2` and `MUTE→D3`) bridged on the ReSpeaker Lite board.

**Re-flashing over USB or changing Wi-Fi.** Run the installer page again. Because USB flashing wipes the flash, you'll re-enter Wi-Fi at the end, click **Reconfigure** > **Submit** on the device card in Home Assistant, and flip **Wake word engine location** back to **In Home Assistant**. Your chosen assistant stays intact.

---

Curious how the firmware is put together or want to build it yourself? See [firmware/README.md](../firmware/README.md).
