# Hark Wake-Word Add-on Setup

1. Configure your desired English (`:en`) and/or German (`:de`) wake phrases under the **Configuration** tab, e.g.:
   - `Okay Haus:de`
   - `Hallo Computer:de`
   - `Hey Jarvis:en`
2. Start the Add-on and check the **Log** tab to verify the IPA pronunciation generated for each phrase.
3. Go to **Settings → Devices & Services → Add Integration → Wyoming Protocol**, enter host `core-hark-wakeword` (or `127.0.0.1`) and port `10400`.
4. Open **Settings → Voice Assistants**, edit your Assist pipeline, and select **hark** + your wake phrase under **Wake word**.
   *(Tip for ESP32 Voice PE / Atom Echo satellites: set wake-word processing on the satellite device to **Home Assistant** rather than **On device** so audio streams to Hark.)*
