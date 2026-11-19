"""Write manifest.json for a Hark firmware (Voice PE or ReSpeaker Lite).

usage: python3 make_manifest.py <dir> <device> [release_url] [summary]

<device> is voice-pe or respeaker-lite. <dir> holds exactly one
hark-<device>-<version>.factory.bin and the matching .ota.bin; the version is
taken from the file name. The manifest serves two readers: ESP Web Tools
(install page: builds[].parts) and the firmware's own update entity (ESPHome
http_request update: version, builds[].ota.path/md5).
"""
import hashlib, json, pathlib, sys

# must match esphome.project.name in the device's YAML
PROJECT_NAMES = {
    'voice-pe': 'Hark.Home Assistant Voice PE',      # voice-pe/home-assistant-voice.hark.yaml
    'respeaker-lite': 'Hark.ReSpeaker Lite',         # respeaker/config/hark-respeaker.yaml
}

d = pathlib.Path(sys.argv[1]); device = sys.argv[2]
release_url = sys.argv[3] if len(sys.argv) > 3 else ''
summary = sys.argv[4] if len(sys.argv) > 4 else ''
if device not in PROJECT_NAMES:
    sys.exit(f'unknown device {device!r}, expected one of {", ".join(PROJECT_NAMES)}')

prefix = f'hark-{device}-'
factories = sorted(d.glob(f'{prefix}*.factory.bin'))
if len(factories) != 1:
    sys.exit(f'expected exactly one {prefix}<version>.factory.bin in {d}, found {len(factories)}')
factory = factories[0]
version = factory.name[len(prefix):-len('.factory.bin')]
ota = d / f'{prefix}{version}.ota.bin'
if not ota.is_file():
    sys.exit(f'missing {ota}')

def digests(p):
    b = p.read_bytes()
    return hashlib.md5(b).hexdigest(), hashlib.sha256(b).hexdigest()

ota_md5, ota_sha = digests(ota)
fac_md5, fac_sha = digests(factory)
manifest = {
    'name': PROJECT_NAMES[device],
    'version': version,
    'home_assistant_domain': 'esphome',
    # no erase prompt; ESP Web Tools still erases (Wi-Fi included) when the running firmware
    # reports another project name, and keeps the settings when it is already this firmware
    'new_install_prompt_erase': False,
    'builds': [{
        'chipFamily': 'ESP32-S3',
        'ota': {'path': ota.name, 'md5': ota_md5, 'sha256': ota_sha,
                'summary': summary, 'release_url': release_url},
        'parts': [{'path': factory.name, 'offset': 0, 'md5': fac_md5, 'sha256': fac_sha}],
    }],
}
(d / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'wrote {d / "manifest.json"} for {device} {version}')
