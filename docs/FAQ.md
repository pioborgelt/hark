# Hark FAQ

Everything we stumbled over while setting Hark up in fresh Home Assistant installs, in English
and German, so you don't have to. Menu names are from Home Assistant 2026.11; German UI terms are
in brackets.

**Jump to:** [Setup](#setup) · [It doesn't hear me](#it-doesnt-hear-me) ·
[It hears too much](#it-hears-too-much) · [Phrases](#phrases) · [Updates](#updates) ·
[Devices & satellites](#devices--satellites) · [Docker](#docker-without-home-assistant-os) ·
[Privacy & resources](#privacy--resources)

---

## Setup

### Where do I install it? There is no "Add-ons" menu.

Home Assistant now calls add-ons **Apps**:

**Settings → Apps → Install app** (button, bottom right) **→ App store → ⋮ (top right) →
Repositories → Add**, then paste `https://github.com/pioborgelt/hark`.

> 🇩🇪 Einstellungen → Apps → **App installieren** → App-Store → ⋮ → **Repositorys** →
> **Hinzufügen**

Hark is further down the store list; type `hark` into the search field.

### I installed and started it. Now what?

Hark only provides the wake word. Home Assistant needs three more things:

1. **Speech-to-text and text-to-speech** in your assistant, e.g. the **Whisper** (or
   **Speech-to-Phrase**) and **Piper** apps. *Without speech-to-text Home Assistant silently
   refuses to run the assistant and Hark never receives audio.*
2. **The Wyoming connection:** **Settings → Devices & services → Discovered → Hark Wake Word →
   Add** (🇩🇪 Geräte & Dienste → **Entdeckt**).
3. **The wake word in your assistant**, which is hidden by default: **Settings → Voice
   assistants → your assistant → ⋮ (top right of the dialog) → Add streaming wake word**, then
   *Wake word engine* = **hark** and *Wake word* = your phrase.
   > 🇩🇪 Sprachassistenten → Assistent → ⋮ → **Streaming-Aktivierungswort hinzufügen** →
   > *Aktivierungswort-Modul* = hark, *Aktivierungswort* = deine Phrase.

### Why is it called "Aktivierungswort" in German?

That is Home Assistant's German word for wake word. Hark's options say "Aktivierungswörter
(Wake-Words)" so you can find both.

### Do I need openWakeWord removed first?

No. Both can run side by side; each assistant picks one engine. Neither opens a host port by
default, so they don't clash.

---

## It doesn't hear me

Open the Hark app → **Log** (🇩🇪 Protokoll) and check which line you see.

| Log shows | Meaning | Fix |
| --- | --- | --- |
| Nothing after `listening on tcp://…` | Home Assistant is not sending audio to Hark. | See the next questions. |
| `Audio stream started, listening for: "…"` | Audio arrives, Hark listens for that phrase. | If it's the wrong phrase, see [*It listens for the wrong phrase*](#it-listens-for-the-wrong-phrase). |
| `Listening: … peak level -60 dBFS` / `nearly silent` | The microphone is muted or very quiet. | Check the satellite's microphone / mute switch. |
| `Near-miss rejected by Stage-2 … combined=0.12 < 0.15` | Almost recognized. | Try `sensitivity: sensitive`, speak the phrase as configured, or pick a longer phrase. |
| `DETECTED: …` but nothing happens | Hark did its job; the rest of the pipeline failed. | Check speech-to-text (below). |

### The log never shows "Audio stream started"

In order of likelihood:

1. **The assistant has no speech-to-text.** Home Assistant aborts before the wake word stage
   (`the pipeline does not support speech-to-text`) and shows no error in the UI.
2. **The satellite uses a different assistant.** Every satellite has its own **Assistant**
   setting (**Settings → Devices & services → your satellite**). It does *not* follow the
   "preferred" assistant.
3. **The satellite went silent** after a restart, an update or a failed command, see below.

### My satellite stops listening after a reboot, an update or a failed command

This is a Home Assistant behaviour with Linux satellites (`wyoming-satellite`): if one
assistant run fails, for example because an app is still starting after a reboot, Home
Assistant does not start a new one and the satellite stays silent until it is reloaded. It also
happens after "No text recognized" or after Hark restarts or updates.

- **Quick fix:** **Settings → Devices & services → Wyoming Protocol → ⋮ next to the satellite
  → Reload** (🇩🇪 Neu laden).
- **Permanent fix:** an automation that reloads it after every start (tested on HA 2026.11):

  ```yaml
  alias: Reload voice satellites after start
  triggers:
    - trigger: homeassistant
      event: start
  actions:
    - delay:
        seconds: 90   # give Whisper/Piper/Hark time to start
    - action: homeassistant.reload_config_entry
      target:
        entity_id: assist_satellite.living_room   # your satellite
  mode: single
  ```

Hark itself starts before Home Assistant Core, so it is ready when satellites connect; the
Whisper app starts later, which is why the reboot case still needs the automation. We have
not tested ESPHome devices (Voice PE) yet; please report whether they recover on their own.

### The first word of my command is missing ("…time is it?")

Hark confirms a wake word about 0.3 s *after* you finish it: its second stage listens to a bit
of audio after the phrase to reject look-alikes. Home Assistant only passes audio from that
moment on to speech-to-text. **Pause briefly after the wake word**, or wait for the satellite's
chime, then speak.

### Speech-to-text understands nonsense in German

The Whisper app defaults to `language: en` and a small model. In the Whisper app's
configuration set **Language** to `de` and, if your hardware allows it, a bigger model
(e.g. `small-int8`). Also set the assistant's speech-to-text language to German.

### It listens for the wrong phrase

Home Assistant uses exactly **one** wake word per assistant, preselected to the first phrase in
the list. Choose the phrase in the assistant (see [Setup](#setup)), and make sure the satellite
uses that assistant. Create a second assistant for a second phrase.

---

## It hears too much

### It triggers on normal speech

Short, common phrases can appear inside normal sentences. In our tests "Okay Haus" triggered on
*"Okay, das Haus da drüben ist ziemlich groß"* with `balanced`, but not with `strict`.

- Set **sensitivity** to `strict`.
- Prefer 3–4 syllables that don't occur in everyday sentences ("Hallo Kühlschrank",
  "Hey Jarvis") over "Okay Haus".
- Turn on **Log near misses**; the scores show how close each false trigger was.

### What do strict / balanced / sensitive mean in numbers?

They set the score threshold: `strict` 0.85, `balanced` 0.15, `sensitive` −0.45 (one-syllable
very short phrases (5 sounds or fewer) get a stricter threshold automatically). Near-miss log lines show the score next to the
threshold, e.g. `combined=0.84 < 0.85`.

---

## Phrases

### What format do phrases need?

`Text:de` or `Text:en`, e.g. `Hallo Kühlschrank:de`, `Hey Jarvis:en`. The language decides
the pronunciation: `Hallo Haus:en` is pronounced as English "Hallo House" and works poorly for
German speakers.

### Saving fails with "does not match regular expression ^.+:(de|en)$"

A phrase is missing `:de` or `:en` (🇩🇪 *Speichern fehlgeschlagen*). Add it and save again.

### Which phrases work best?

3–4 syllables with clear consonants: "Hallo Kühlschrank", "Hey Jarvis", "Okay Computer",
"Hallo Computer". Avoid single short words ("Hey", "Computer") and phrases that occur in normal
speech.

### Can I use other languages, umlauts or numbers?

Umlauts and `ß` are fine, and digits are read out ("Computer 3000"). Other languages and
scripts (French, Cyrillic, Chinese, emoji) are rejected with an error in the log, because the
model only knows German and English sounds.

### I added a phrase but it isn't in the assistant's list

Restart the Hark app after saving, then reload the *Hark Wake Word*
entry under **Settings → Devices & services → Wyoming Protocol → ⋮ → Reload**.

---

## Updates

### "Update available" is shown but there is no update button

Home Assistant creates the update button for an app when its *Supervisor* integration starts.
Right after the first install it's missing. Restart Home Assistant once (**Settings → ⋮ →
Restart Home Assistant**) or reload **Settings → Devices & services → Home Assistant
Supervisor**. Afterwards updates appear on the app page and under **Settings**. **Rebuild**
(🇩🇪 Neu aufbauen) does not install a newer version.

### After updating, nothing reacts

Updating restarts Hark, and Linux satellites go silent until reloaded, see
[My satellite stops listening](#my-satellite-stops-listening-after-a-reboot-an-update-or-a-failed-command).

---

## Devices & satellites

### Does Hark run on my ESP32 / Voice PE?

No, Hark runs on the Home Assistant machine. The device streams audio to Home Assistant:

- **Home Assistant Voice PE / ESPHome:** set the wake word location to **In Home Assistant**
  instead of *On device*.
- **wyoming-satellite** (Raspberry Pi, Linux PC): run it without a local wake word service.

### Can I use several phrases at once?

Hark can listen for all configured phrases, but Home Assistant asks for one phrase per
assistant. Use one assistant per phrase and assign them to different satellites.

---

## Docker (without Home Assistant OS)

### Which host do I enter in the Wyoming dialog?

The IP address of the machine running the Hark container, and port `10400`
(**Settings → Devices & services → Add integration → Wyoming Protocol**; 🇩🇪 *Integration
hinzufügen*). `localhost` only works if Home Assistant itself runs in Docker with
`network_mode: host` on the same machine. On Home Assistant OS, `localhost` is Home Assistant,
not your Docker host, and you get **"Failed to connect"** (🇩🇪 *Verbindung fehlgeschlagen*).

Two things that trip people up in that dialog:

- The **port is prefilled with 0**. Change it to `10400`.
- **After an error, Home Assistant clears both fields.** Enter host *and* port again before
  retrying.

### The container started, but Home Assistant can't connect

- `docker logs wyoming-hark` must end with `listening on tcp://0.0.0.0:10400`. The first
  start downloads ~31 MB of model weights; until then the port is closed.
- The port must be published (`-p 10400:10400`) and not blocked by a firewall.
- Log timestamps inside the container are UTC.

### Where are the weights, debug clips and settings?

Everything lives in the mounted `/data` volume (`./hark-data` in the examples): weights in
`models/`, debug clips (if `HARK_DEBUG_CLIP_DIR=/data/debug_clips` is set) in
`debug_clips/`. Weights are checked against pinned SHA256 hashes on every start.

### How do I change phrases?

Edit `HARK_PHRASES` (separated by `;`), recreate the container, then reload the *hark* entry in
**Settings → Devices & services → Wyoming Protocol** so Home Assistant picks up the new list.
Phrases with an unsupported language or characters are skipped with an error in the log.

### After restarting the container nothing reacts

Same as the app: Linux satellites go silent until reloaded, see
[My satellite stops listening](#my-satellite-stops-listening-after-a-reboot-an-update-or-a-failed-command).

---

## Privacy & resources

### Does Hark record or upload audio?

No. Everything runs locally. Nothing is stored unless you turn on **Save debug clips**, which
keeps the last 30 three-second clips in `/data/debug_clips` for troubleshooting. Satellites
that use Home Assistant wake word detection stream microphone audio to your Home Assistant
continuously.

### How well does it work in Home Assistant?

In our end-to-end tests through a real Home Assistant pipeline with recordings of real people
(Picovoice wake word benchmark): "Jarvis" 8/8, "Computer" 6/8, no false triggers on other
speech. Synthetic German voices: "Hallo Kühlschrank" and "Okay Haus" were recognized; a very
low-quality voice was missed. Your microphone and room matter more than anything else, so
watch the near-miss lines in the log.

### How much CPU and RAM does it need?

In our x86 test VM: about 390 MB RAM and close to 0 % CPU while idle. Raspberry Pi numbers are
not measured yet. Please report yours.

### Is it hard to install?

About 1.5 minutes on a fast x86 machine (it downloads PyTorch); considerably longer on a
Raspberry Pi. The install button only shows a spinner meanwhile.
