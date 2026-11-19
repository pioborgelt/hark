# Hark FAQ

I set Hark up in a bunch of fresh Home Assistant installs, in English and in German, and wrote down everything that tripped me up. Menu names are from Home Assistant 2026.11. If your Home Assistant is in German, I put the German names in brackets.

- [Setup](#setup)
- [It doesn't hear me](#it-doesnt-hear-me)
- [It hears too much](#it-hears-too-much)
- [Phrases](#phrases)
- [Updates](#updates)
- [Devices and satellites](#devices-and-satellites)
- [Docker](#docker)
- [Privacy and resources](#privacy-and-resources)

## Setup

### There is no "Add-ons" menu. Where do I install it?

Home Assistant renamed add-ons to "Apps" a while ago. Go to Settings > Apps and click "Install app" in the bottom right. In the app store, open the three-dot menu in the top right, choose "Repositories" and add `https://github.com/pioborgelt/hark`. In German that's Einstellungen > Apps > "App installieren", then "Repositorys".

Hark ends up somewhere further down the store list, so just type "hark" into the search.

### It's installed and running. Now what?

Hark only gives you the wake word. Home Assistant needs three more things before anything happens:

1. Your assistant needs speech-to-text and text-to-speech, for example the Whisper (or Speech-to-Phrase) and Piper apps. This one is sneaky: without speech-to-text, Home Assistant quietly refuses to run the assistant at all, so Hark never gets any audio and you don't see an error anywhere.
2. Add the Wyoming connection: Settings > Devices & services, then under "Discovered" (Entdeckt) click Add on Hark Wake Word.
3. Turn on the wake word in your assistant. That option is hidden: open Settings > Voice assistants, click your assistant, open the three-dot menu in the top right of that dialog and choose "Add streaming wake word" (Streaming-Aktivierungswort hinzufügen). Then pick hark as the engine and your phrase as the wake word.

### Do I have to remove openWakeWord first?

No, both can run side by side. Each assistant uses one of them, and neither opens a port on your host, so they don't get in each other's way.

## It doesn't hear me

Open the Hark app and look at the Log tab (Protokoll). What you see there tells you where it's stuck:

- **Nothing after `listening on tcp://...`**: Home Assistant isn't sending any audio to Hark. The next question is for you.
- **`Audio stream started, listening for: "..."`**: audio arrives and Hark is listening for that phrase. If it's the wrong one, see [It listens for the wrong phrase](#it-listens-for-the-wrong-phrase).
- **`Listening: ... peak level -60 dBFS` or `nearly silent`**: the microphone is muted or extremely quiet. Check the mute switch on your satellite.
- **`Near-miss rejected by Stage-2 ... combined=0.12 < 0.15`**: that one was close. Try the `sensitive` setting, say the phrase the way you typed it, or pick a longer phrase.
- **`DETECTED: ...` but nothing happens**: Hark did its part, something later in the pipeline failed. Usually that's speech-to-text, see below.

### The log never says "Audio stream started"

The usual suspects, most likely first:

1. The assistant has no speech-to-text. Home Assistant gives up before it even gets to the wake word and doesn't tell you in the UI.
2. Your satellite uses a different assistant. Every satellite has its own assistant setting (Settings > Devices & services > your satellite), and it does not follow the "preferred" assistant.
3. The satellite went silent after a restart, an update or a failed command. That's the next question.

### My satellite stops listening after a reboot, an update or a failed command

This is how Home Assistant behaves with Linux satellites (`wyoming-satellite`). If one assistant run fails, say because an app is still starting after a reboot, Home Assistant doesn't start a new one, and the satellite stays quiet until you reload it. The same happens after "No text recognized" and whenever Hark restarts or updates.

The quick fix is to reload the satellite: Settings > Devices & services > Wyoming Protocol, then the three-dot menu next to the satellite and "Reload" (Neu laden).

To never think about it again, use an automation that reloads it after every start. I tested this one on HA 2026.11:

```yaml
alias: Reload voice satellites after start
triggers:
  - trigger: homeassistant
    event: start
actions:
  - delay:
      seconds: 90   # give Whisper, Piper and Hark time to start
  - action: homeassistant.reload_config_entry
    target:
      entity_id: assist_satellite.living_room   # your satellite
mode: single
```

Hark starts before Home Assistant itself, so it's ready when satellites connect. The Whisper app starts later though, which is why you still need the automation after a reboot. ESPHome satellites behave better here: on a ReSpeaker Lite with the Hark firmware, streaming comes back on its own after a Home Assistant restart (and backs off cleanly if Hark is temporarily down instead of getting stuck).

### The first word of my command is missing ("...time is it?")

Hark confirms the wake word about 0.3 seconds after you finish saying it, because the second stage listens to a little bit of audio after the phrase to rule out look-alikes. Home Assistant only hands audio from that point on to speech-to-text. So pause briefly after the wake word, or wait for the satellite's chime, and then talk.


### It listens for the wrong phrase

Home Assistant uses exactly one wake word per assistant, and it preselects the first phrase in the list. Pick your phrase in the assistant (see [Setup](#setup)) and make sure the satellite uses that assistant. For a second phrase, create a second assistant.

## It hears too much

### It goes off during normal conversation

Set the sensitivity to `strict`, and prefer phrases with three or four syllables that nobody says in passing ("Hallo Kühlschrank", "Hey Jarvis") over something like "Okay Haus". If you turn on "Log near misses", the log shows you how close each false trigger was.

### What do strict, balanced and sensitive actually do?

They move the score threshold: 0.85 for `strict`, 0.15 for `balanced` and -0.45 for `sensitive`. Very short phrases (five sounds or fewer) automatically get a stricter threshold. The near-miss lines in the log show the score next to the threshold, like `combined=0.84 < 0.85`.

## Phrases

### How do I write a phrase?

The text, a colon and the language: `Hallo Kühlschrank:de` or `Hey Jarvis:en`. The language decides how it's pronounced. `Hallo Haus:en` gets pronounced like English "Hallo House", which won't work well if you say it in German.

### Saving fails with "does not match regular expression ^.+:(de|en)$"

One of your phrases is missing the `:de` or `:en` at the end. Add it and save again.

### Which phrases work best?

Three or four syllables with clear consonants, like "Hallo Kühlschrank", "Hey Jarvis", "Okay Computer" or "Hallo Computer". Single short words ("Hey", "Computer") and phrases you'd say in normal conversation work worse.

### Can I use other languages, umlauts or numbers?

Umlauts and ß are fine, and numbers get read out ("Computer 3000"). Other languages and alphabets (French, Cyrillic, Chinese, emoji) are rejected with an error in the log. The model only knows German and English sounds.

### I added a phrase but it's not in the assistant's list

Restart the Hark app after saving. Then reload the Hark Wake Word entry under Settings > Devices & services > Wyoming Protocol (three-dot menu, "Reload").

## Updates

### It says "Update available" but there's no update button

Home Assistant only creates the update button for an app when its Supervisor integration starts, so right after the first install it's missing. Restart Home Assistant once (Settings, three-dot menu, "Restart Home Assistant") or reload the Home Assistant Supervisor entry under Settings > Devices & services. After that, updates show up on the app page and under Settings. "Rebuild" (Neu aufbauen) won't install a newer version, by the way.

### After the update nothing reacts anymore

The update restarts Hark, and Linux satellites go quiet until you reload them. See [My satellite stops listening](#my-satellite-stops-listening-after-a-reboot-an-update-or-a-failed-command).

## Devices and satellites

### Does Hark run on my ESP32 or Voice PE?

No, Hark runs on your Home Assistant machine and the satellite streams its audio there. Stock Voice PE and ReSpeaker Lite firmware only listen on-device, so flash the Hark firmware first — see [ESP32 satellites](ESP32_SATELLITE.md). It adds a "Wake word engine location" switch; flip it to "In Home Assistant" and you're set. Devices like the ESP32-S3-Box or M5Stack Atom Echo already have that setting out of the box. If you use `wyoming-satellite` on a Raspberry Pi or Linux PC, run it without a local wake word service.

### Can I use several phrases at once?

Hark can listen for all of them, but Home Assistant asks for one phrase per assistant. So make one assistant per phrase and give them to different satellites.

## Docker

### What host do I enter in the Wyoming dialog?

The IP address of the machine running the container, and port `10400`. You find the dialog under Settings > Devices & services > Add integration > Wyoming Protocol. `localhost` only works if Home Assistant itself runs in Docker with `network_mode: host` on the same machine. On Home Assistant OS, `localhost` is Home Assistant and not your Docker host, so you'll get "Failed to connect" (Verbindung fehlgeschlagen).

One more thing about that dialog that caught me: the port is prefilled with 0, so change it to 10400.

### The container runs, but Home Assistant can't connect

Check `docker logs wyoming-hark`. It should end with `listening on tcp://0.0.0.0:10400`. On the very first start Hark downloads about 31 MB of model weights, and the port only opens after that. Also make sure the port is published (`-p 10400:10400`) and no firewall is in the way. Small thing: the timestamps in the container log are UTC.

### Where are the weights, debug clips and settings?

Everything is in the `/data` volume (`./hark-data` in the examples). The weights are in `models/` and get checked against their known checksums on every start. Debug clips land in `debug_clips/` if you set `HARK_DEBUG_CLIP_DIR=/data/debug_clips`.

### How do I change the phrases?

Change `HARK_PHRASES` (separated by `;`), recreate the container, and then reload the hark entry under Settings > Devices & services > Wyoming Protocol so Home Assistant sees the new list. Phrases in an unsupported language or with odd characters are skipped with an error in the log.

### After restarting the container nothing reacts

Same story as with the app: Linux satellites go quiet until you reload them. See [My satellite stops listening](#my-satellite-stops-listening-after-a-reboot-an-update-or-a-failed-command).

## Privacy and resources

### Does Hark record or upload anything?

No, everything stays on your machine. Nothing gets saved unless you turn on "Save debug clips", which keeps the last 30 three-second clips in `/data/debug_clips` for troubleshooting. Keep in mind that satellites using wake word detection in Home Assistant stream their microphone to your Home Assistant all the time. That's how it works with any engine that runs there.

### How well does it work inside Home Assistant?

I ran recordings of real people from the Picovoice benchmark through a real Home Assistant pipeline. "Jarvis" was recognized 8 out of 8 times and "Computer" 6 out of 8, with no false triggers on the other speech. With synthetic German voices, "Hallo Kühlschrank" and "Okay Haus" worked, and one very low-quality voice got missed. In practice your microphone and your room matter more than anything else, so keep an eye on the near-miss lines in the log.

### How much CPU and RAM does it need?

In my x86 test VM it used about 390 MB of RAM and almost no CPU while idle. I haven't measured a Raspberry Pi yet. If you run it on one, please tell me how it goes.

### How long does the install take?

About a minute and a half on a fast x86 machine, mostly for downloading PyTorch. On a Raspberry Pi it takes a good while longer. Home Assistant only shows a spinner during that time, so be patient.
