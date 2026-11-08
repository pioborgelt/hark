"""PAI-Bench v2 (DE only; v1 DE was dominated by the low-quality multi-speaker de_DE-mls-medium voice): synthetic wakeword-in-use test set for 'Okay PAI' / 'Hey PAI' (DE and EN).

Test-only material. Multi-speaker Piper voices (de_DE-mls-medium 236 speakers, en_US-libritts_r-medium 904
speakers) plus all local single-speaker DE/EN voices. Each utterance is either the bare call or the call followed by
a short command, spoken with random tempo. Audio is stored once as 16 kHz WAV (~25 MB); conditions (music, room,
context) are applied deterministically at evaluation time. Never use for training or calibration of a candidate
that is then reported on this set without a separate split."""
import hashlib, json, math, random, wave
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from piper import PiperVoice, SynthesisConfig

ROOT = Path('data/pai-bench-v2-de')
CALLS = {'de': ['Okay PAI', 'Hey PAI'], 'en': ['Okay PAI', 'Hey PAI']}
TAILS = {'de': ['', '', ', wie spät ist es?', ', mach das Licht an.', ', wie wird das Wetter morgen?', ', spiel etwas Musik.'],
         'en': ['', '', ", what time is it?", ', turn on the lights.', ", what's the weather tomorrow?", ', play some music.']}
MULTI = {'de': ('de_DE-mls-medium', 236, 70), 'en': ('en_US-libritts_r-medium', 904, 70)}
SINGLE = {'de': ['de_DE-thorsten-medium', 'de_DE-kerstin-low', 'de_DE-eva_k-x_low', 'de_DE-ramona-low', 'de_DE-karlsson-low', 'de_DE-pavoque-low'],
          'en': ['en_US-lessac-medium', 'en_US-amy-medium', 'en_US-ryan-medium', 'en_US-hfc_female-medium', 'en_US-hfc_male-medium', 'en_US-kathleen-low']}


NOISE = {}


def synth(voice, text, speaker, length):
    ns, nw = NOISE.get('v', (None, None))
    chunks = list(voice.synthesize(text, SynthesisConfig(speaker_id=speaker, length_scale=length, noise_scale=ns, noise_w_scale=nw)))
    x = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32); rate = chunks[0].sample_rate
    if rate != 16000: g = math.gcd(rate, 16000); x = resample_poly(x, 16000 // g, rate // g).astype(np.float32)
    return x


def call_end_seconds(voice, call, speaker, length, full):
    """Duration of the bare call rendered with the same speaker/tempo (upper bound for where the call ends)."""
    return len(synth(voice, call, speaker, length)) / 16000


def main():
    rng = random.Random(20261108); (ROOT / 'wav').mkdir(parents=True, exist_ok=True); rows = []
    for lang in ['de']:
        jobs = []
        for v in SINGLE[lang]: jobs += [(v, None)] * 10
        for spk in range(8):
            if spk != 3: jobs += [('de_DE-thorsten_emotional-medium', spk)] * 6  # all styles except 'drunk'
        cache = {}
        for vname, spk in jobs:
            if vname not in cache: cache[vname] = PiperVoice.load(f'data/voices/{vname}.onnx')
            voice = cache[vname]
            for call in CALLS[lang]:
                tail = rng.choice(TAILS[lang]); length = rng.uniform(.8, 1.25); text = call + tail
                NOISE['v'] = (rng.uniform(.45, .9), rng.uniform(.5, 1.0))
                x = synth(voice, text, spk, length)
                end = call_end_seconds(voice, call, spk, length, x) if tail else len(x) / 16000
                NOISE['v'] = (None, None)
                pid = hashlib.sha256(f'{lang}|{vname}|{spk}|{text}|{length:.4f}'.encode()).hexdigest()[:16]
                path = ROOT / 'wav' / f'{pid}.wav'
                with wave.open(str(path), 'wb') as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
                rows.append({'id': pid, 'language': lang, 'voice': vname, 'speaker': spk, 'call': call, 'text': text,
                             'length_scale': length, 'seconds': len(x) / 16000, 'call_end_upper_seconds': min(end + .15, len(x) / 16000),
                             'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(json.dumps({'language': lang, 'utterances': sum(r['language'] == lang for r in rows)}), flush=True)
    (ROOT / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (ROOT / 'README.md').write_text(__doc__ + f'\nUtterances: {len(rows)}\n')


if __name__ == '__main__':
    main()
