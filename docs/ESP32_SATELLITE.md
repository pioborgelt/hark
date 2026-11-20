# ESP32 satellites (Voice PE & ReSpeaker Lite)

Stock Voice PE and ReSpeaker Lite firmware only listen on the device itself. The Hark firmware adds one setting, **Wake word engine location**: on **On device** it behaves like stock, on **In Home Assistant** it streams the mic to Home Assistant so Hark can listen for any phrase you type.

Music only ducks once Hark confirms the wake word, and the button, mute switch, timers and "stop" word keep working as usual. It also patches a race condition in ESPHome's `voice_assistant` where a late `RUN_END` from a cancelled run could kill the next one.

> **Note:** Right now the satellite streams audio continuously. Later releases will run Hark's first stage directly on the ESP32 so audio only leaves the device on a candidate hit.

Supported boards:
- **Home Assistant Voice PE (16 MB, standard model):** based on official `26.11.0`. Same streaming logic as the ReSpeaker build, not yet tested on a physical Voice PE (8 MB models aren't built yet). Feedback in [issue #1](https://github.com/pioborgelt/hark/issues/1) helps a lot.
- **ReSpeaker Lite (XIAO ESP32-S3):** based on formatBCE's `2026.11.1`, tested on real hardware. Needs a 2.4 GHz U.FL antenna on the XIAO and [XMOS 48 kHz I2S firmware 1.1.0+](https://wiki.seeedstudio.com/xiao_respeaker/#flash-the-i2s-firmware) (flashed via the main board USB port; skip if you already ran formatBCE's firmware).

## Setup

You need a USB-C data cable, Chrome or Edge on a computer, and the Hark app running in Home Assistant (steps 1 to 3 of the [app docs](../hark_addon/DOCS.md)).

1. **Flash once over USB.** Plug the device into your computer (on ReSpeaker Lite, use the USB-C port on the small XIAO board). Open [pioborgelt.github.io/hark/firmware](https://pioborgelt.github.io/hark/firmware/), pick your device, click **Install Hark firmware**, select **USB JTAG/serial debug unit**, and install (~30 s). USB flashing wipes Wi-Fi, so enter your Wi-Fi at the end, then click **Skip**.
2. **Add to Home Assistant.** Go to Settings > Devices & services (Geräte & Dienste), click **Add** on the discovered device, then **Submit**. In the setup wizard, click **Skip** on the "Okay Nabu" screen (ignore any false muted warning on ReSpeaker), pick your local processing option, and note which assistant it uses.
3. **Attach Hark.** Open Settings > Voice assistants (Sprachassistenten), click that assistant, open the three-dot menu in the top right, choose **Add streaming wake word**, and pick **hark** + your phrase.
4. **Flip the switch.** Open Settings > Devices & services > ESPHome > your device, and set **Wake word engine location** to **In Home Assistant**. Under Diagnostic, **Streaming status** should switch to `streaming, waiting for wake word`.

Say your wake word, pause a beat, then give your command.

## Updates and going back

Future Hark builds show up as normal firmware updates in Home Assistant and keep your settings. To go back to stock, flash over USB from the [Voice PE installer](https://esphome.github.io/home-assistant-voice-pe/) or [formatBCE's ReSpeaker Lite installer](https://formatbce.github.io/Respeaker-ESPHome-integration/lite/).

## Troubleshooting

- **Device not listed in Chrome:** Try a different USB cable (many are charge-only). On ReSpeaker Lite, hold **BOOT** on the XIAO while plugging in. On Linux, run `sudo usermod -aG dialout $USER` and re-login.
- **Flash fails halfway:** Reload the page and try a direct USB port without a hub.
- **Nothing in the Hark log:** Check that **Wake word engine location** is `In Home Assistant`, the device's **Assistant** matches the one with Hark (satellites ignore the "preferred" assistant), and that assistant has speech-to-text set up.
- **Misses or lag:** Usually Wi-Fi drops. The Hark log prints `Listening: 60 s audio received` every minute; if those lines come much slower, audio is getting lost (check the ReSpeaker antenna or move closer to the router).
- **First word cut off:** Pause ~0.3 s after the wake word while Stage 2 confirms it.
- **ReSpeaker mute / user buttons dead:** Needs the `USR->D2` and `MUTE->D3` solder jumpers bridged on the board.
- **Reflashing over USB / changing Wi-Fi:** USB flashing resets the device, so re-enter Wi-Fi, click **Reconfigure** in Home Assistant, and flip **Wake word engine location** back to `In Home Assistant`.

Technical details and build steps are in [firmware/README.md](../firmware/README.md).
