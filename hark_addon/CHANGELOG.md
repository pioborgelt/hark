# Changelog

## 0.3.5-alpha

- Installs a ready-made image instead of building it on your device, so installing and updating
  takes seconds to minutes instead of a long build (especially on a Raspberry Pi).
- The default wake phrase is now just "Hey Jarvis:en". Add your own phrases on the
  Configuration tab.

## 0.3.4-alpha

- Starts before Home Assistant Core (`startup: services`), so satellites find the wake word
  engine right after a reboot instead of staying silent.
- Shorter, readable phrase check: Home Assistant's error now shows `^.+:(de|en)$`.
- German option names use Home Assistant's own term "Aktivierungswort".
- New icon and logo.

## 0.3.3-alpha

- Phrases must end in `:de` or `:en`; the configuration form rejects anything else.
- Invalid phrases (unsupported language, symbols, other scripts) are skipped with a clear log
  error instead of being enrolled as nonsense; duplicates are ignored.
- Readable option names and descriptions (English, German), app icon and logo.
- Documentation rewritten after real Home Assistant walkthroughs (menu names, prerequisites,
  hidden wake word setting, satellite reloads, update button).

## 0.3.2-alpha

- Short wake word names in Home Assistant ("Okay Jarvis (EN)" instead of the IPA string).
- Troubleshooting for assistants per satellite, reloading after restarts, updates and STT.

## 0.3.1-alpha

- Log when an audio stream starts and ends, plus the audio level every minute,
  so a silent pipeline is visible in the add-on log.

## 0.3.0-alpha

- First add-on release: zero-shot EN/DE text-enrolled wake words over Wyoming.
- Automatic Wyoming discovery; no host port opened by default.
- Model weights are pinned by SHA256 and loaded without pickled code.
