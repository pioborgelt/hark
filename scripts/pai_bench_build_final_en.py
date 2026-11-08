"""Sealed final set for the 'beat sherpa OKAY PIE' gate (English): 80 libritts_r speakers that are NOT in PAI-Bench v1.
Same calls, tails and tempo range as v1. Must stay unevaluated until one Hark candidate and its operating point are locked."""
import hashlib, json, random, wave
from pathlib import Path
import numpy as np
from piper import PiperVoice
from scripts.pai_bench_build import synth, call_end_seconds, CALLS, TAILS

ROOT = Path('data/pai-bench-final-en'); VOICE = 'en_US-libritts_r-medium'


def main():
    v1 = [json.loads(s) for s in Path('data/pai-bench-v1/manifest.jsonl').read_text().splitlines()]
    used = {r['speaker'] for r in v1 if r['voice'] == VOICE}
    rng = random.Random(202611082); (ROOT / 'wav').mkdir(parents=True, exist_ok=False)
    speakers = rng.sample(sorted(set(range(904)) - used), 80); voice = PiperVoice.load(f'data/voices/{VOICE}.onnx'); rows = []
    for spk in speakers:
        for call in CALLS['en']:
            tail = rng.choice(TAILS['en']); length = rng.uniform(.85, 1.2); text = call + tail
            x = synth(voice, text, spk, length); end = call_end_seconds(voice, call, spk, length, x) if tail else len(x) / 16000
            pid = hashlib.sha256(f'final|{spk}|{text}|{length:.4f}'.encode()).hexdigest()[:16]; path = ROOT / 'wav' / f'{pid}.wav'
            with wave.open(str(path), 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
            rows.append({'id': pid, 'language': 'en', 'voice': VOICE, 'speaker': spk, 'call': call, 'text': text, 'length_scale': length,
                         'seconds': len(x) / 16000, 'call_end_upper_seconds': min(end + .15, len(x) / 16000), 'path': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert not ({r['speaker'] for r in rows} & used); print(len(rows), len(speakers))


if __name__ == '__main__':
    main()
