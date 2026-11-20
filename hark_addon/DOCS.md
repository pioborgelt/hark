# Hark Wake Word

Hark listens for any English or German wake phrase you type in. No recordings, no training.

If your Home Assistant is in German: it calls a wake word "Aktivierungswort". I've put the German menu names in brackets where they help.

## Before you start

Hark only does the wake word. Your assistant also needs speech-to-text and text-to-speech, for example the Whisper (or Speech-to-Phrase) and Piper apps. Install those first and add them under Settings > Devices & services once they show up there. Without speech-to-text, Home Assistant quietly refuses to start the assistant and Hark never hears anything.

German speakers: the Whisper app starts out in English, so set its language to `de` in its configuration.

You also need a device that streams its microphone to Home Assistant (a "satellite"), more on that below.

## Setup

1. On the Configuration tab, add your wake phrases. Each one ends with `:de` or `:en`, like `Hallo Computer:de` or `Hey Jarvis:en`. Press Enter after each phrase, then Save.
2. Start the app on the Info tab. The first start takes a moment. Afterwards the Log tab lists each phrase with its pronunciation.
3. Go to Settings > Devices & services (Geräte & Dienste). Under "Discovered" (Entdeckt) you'll see Hark Wake Word. Click Add, then Submit.
4. Open Settings > Voice assistants (Sprachassistenten) and click your assistant. The wake word setting is hidden in the three-dot menu in the top right of that dialog: "Add streaming wake word" (Streaming-Aktivierungswort hinzufügen). Choose hark as the engine and your phrase as the wake word, then Update.
5. Check that your satellite actually uses this assistant: Settings > Devices & services, open your satellite and look at its Assistant setting. Satellites don't follow the "preferred" assistant.

Now say the wake word, pause briefly (or wait for the chime) and then say your command. Hark confirms the wake word about 0.3 seconds after you finish it, and anything you say in that moment doesn't reach speech-to-text.

## Which devices work with it?

Hark runs on your Home Assistant machine, not on the microphone device, so the satellite has to send its audio to Home Assistant. Voice PE and ReSpeaker Lite need the Hark firmware for that ([ESP32 satellites](https://github.com/pioborgelt/hark/blob/main/docs/ESP32_SATELLITE.md)), which adds a "Wake word engine location" switch to set to "In Home Assistant". Devices like the ESP32-S3-Box or M5Stack Atom Echo already have this setting. If you use `wyoming-satellite` on a Raspberry Pi or Linux PC, run it without a local wake word service.

## When something doesn't work

**Nothing in the log when you speak.** As soon as Home Assistant sends audio, the log says `Audio stream started, listening for: ...`, and then once a minute shows a `Listening: ...` line with the audio level. If you never see those, Home Assistant isn't running a wake word pipeline for your device. Check speech-to-text and the satellite's assistant (see above).

**The satellite stopped listening.** After this app restarts or updates, after you switch assistants, or after a failed command like "no text recognized", Home Assistant doesn't always restart the satellite. Reload it under Settings > Devices & services > Wyoming Protocol, three-dot menu next to the satellite, "Reload". Or just restart the satellite. The FAQ has an automation that does this for you after every Home Assistant start.

**A new phrase is missing from the wake word list.** Reload the Hark Wake Word entry the same way.

**You want two phrases.** Home Assistant listens for one wake word per assistant. Create a second assistant for a second phrase.

**The first word of your command is cut off.** Pause briefly after the wake word.

**There's no update button.** Home Assistant creates it when its Supervisor integration starts, so it's missing right after the first install. Restart Home Assistant once, or reload Settings > Devices & services > Home Assistant Supervisor. "Rebuild" does not install a newer version.

**Saving fails with "does not match regular expression".** One of your phrases is missing `:de` or `:en`.

**The app stopped.** Check the Log tab, and turn on Watchdog on the Info tab so Home Assistant restarts it automatically.

More answers in the FAQ: https://github.com/pioborgelt/hark/blob/main/docs/FAQ.md

## Options

**Wake phrases.** `Text:de` or `Text:en`. Three or four syllables work best.

**Sensitivity.** `strict` gives fewer false triggers, `sensitive` fewer misses, `balanced` sits in between.

**CPU threads.** How many threads Hark uses for detection.

**Log near misses.** Logs phrases that were almost recognized, with their scores. Handy for tuning.

**Save debug clips.** Keeps the last 30 detected or rejected three-second clips in `/data/debug_clips`. This records real audio from your room, so leave it off unless you're debugging.

## Network

Home Assistant talks to the app over its internal network, so no port is opened on your host. That's also why it doesn't clash with the openWakeWord app, which uses the same port. Only set a host port under Configuration > Network if something outside Home Assistant needs to reach it.

## Status

This is an alpha. If it misses you or goes off when it shouldn't, please [open an issue](https://github.com/pioborgelt/hark/issues/new) with the phrase, language, sensitivity, your microphone and the near-miss lines from the log.
