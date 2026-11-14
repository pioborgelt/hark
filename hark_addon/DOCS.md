# Hark Wake Word

Hark detects any English or German wake phrase you type as text. No recordings, no training.

German UI terms are given in brackets: Home Assistant calls a wake word *Aktivierungswort*.

## Before you start

Hark only does the wake word. Your voice assistant also needs **speech-to-text** and
**text-to-speech**, otherwise Home Assistant silently refuses to start and Hark never hears
anything. Install them first, e.g. the **Whisper** (or **Speech-to-Phrase**) and **Piper** apps,
and add them via **Settings → Devices & services** when they show up as discovered.
German users: set *Language* to `de` in the Whisper app configuration (it defaults to `en`).

You also need a device that streams audio to Home Assistant (a *satellite*), see below.

## Setup

1. **Configuration** tab → *Wake phrases*: add your phrases, each ending in `:de` or `:en`,
   e.g. `Okay Haus:de`, `Hey Jarvis:en`. Press Enter after typing, then **Save**.
2. **Info** tab → **Start**. The first start takes a moment. The **Log** tab then lists every
   phrase with its pronunciation (IPA).
3. **Settings → Devices & services** (*Geräte & Dienste*): under **Discovered** (*Entdeckt*)
   you see *Hark Wake Word / Wyoming Protocol* → **Add** → **Submit**.
4. **Settings → Voice assistants** (*Sprachassistenten*) → open your assistant → **⋮** (top
   right of the dialog) → **Add streaming wake word** (*Streaming-Aktivierungswort hinzufügen*).
   Choose **hark** as *Wake word engine* and your phrase as
   *Wake word* (the first phrase is preselected) → **Update**.
5. Make sure your satellite uses this assistant: **Settings → Devices & services → your
   satellite device → Assistant**. Satellites do not follow the "preferred" assistant.

Then say the wake word, **pause briefly** (or wait for the satellite's chime) and say your
command. Hark confirms a wake word about 0.3 s after you finish it; words spoken in that moment
do not reach speech-to-text.

## Which devices can use it?

Hark runs on your Home Assistant machine, not on the microphone device, so the satellite has
to stream audio to Home Assistant:

- **Home Assistant Voice PE / ESPHome devices**: set the wake word location to
  **In Home Assistant** instead of *On device*.
- **wyoming-satellite** (Raspberry Pi, Linux PC): run it *without* a local wake word service.

These satellites stream microphone audio to Home Assistant continuously.

## Troubleshooting

- **Nothing in the log when you speak:** the log shows `Audio stream started, listening for:
  ...` as soon as Home Assistant sends audio, and a `Listening: ...` line with the audio level
  every minute. No such line means Home Assistant is not running a wake word pipeline for your
  device; check speech-to-text and the satellite's assistant (above).
- **Satellite stopped listening:** after this app restarts or updates, after you change the
  assistant, or after a failed command (e.g. "no text recognized"), Home Assistant does not
  always restart the satellite's pipeline. Reload the satellite: **Settings → Devices &
  services → Wyoming Protocol → ⋮ next to the satellite → Reload**, or restart the satellite.
  To do this automatically after every Home Assistant start, use the automation in the FAQ.
- **New phrase missing in the wake word list:** reload the *Hark Wake Word* entry the same way.
- **Only one phrase per assistant:** Home Assistant listens for the single wake word chosen in
  the assistant. Create a second assistant for a second phrase.
- **First word of the command is missing:** pause briefly after the wake word (see above).
- **No update button:** Home Assistant creates the update button for an app when its Supervisor
  integration starts. After the first install, restart Home Assistant once (or reload **Settings
  → Devices & services → Home Assistant Supervisor**); updates then appear on the app page and
  under **Settings**. *Rebuild* does not install a newer version.
- **"does not match regular expression" when saving:** a phrase is missing `:de` or `:en`.
- **The app stopped:** check the **Log** tab, and enable **Watchdog** on the Info tab so Home
  Assistant restarts it automatically.

More answers: https://github.com/pioborgelt/hark/blob/main/docs/FAQ.md

## Options

| Option | Meaning |
| --- | --- |
| Wake phrases | `Text:de` or `Text:en`. 3–4 syllables work best. |
| Sensitivity | `strict` (fewer false triggers), `balanced`, `sensitive` (fewer misses). |
| CPU threads | Threads for detection. |
| Log near misses | Logs phrases that were almost recognized, with their scores. |
| Save debug clips | Saves the last 30 detected/rejected 3-second clips to `/data/debug_clips`. **Records real room audio**; keep it off unless you are debugging. |

## Network

Home Assistant talks to the app over its internal network, so no host port is opened. This also
avoids clashing with the openWakeWord app, which uses the same port. Only set a host port under
**Configuration → Network** if a client outside Home Assistant needs the service.

## Status

Alpha. Report misses or false triggers with the phrase, language, sensitivity, microphone and
the near-miss lines from the log: [open an issue](https://github.com/pioborgelt/hark/issues/new)
