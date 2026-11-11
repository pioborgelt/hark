"""Cheap local proxy for the Picovoice benchmark to compare decoder variants (not a replacement for wwb_eval on Colab).

Positives: every real keyword clip, peak-normalised to 0.3, with 0.5 s silence before and 1 s after, no noise.
Negatives: all of LibriSpeech test-clean (files without any benchmark keyword), streamed as one ~5.4 h signal.
Log-probabilities are computed once per model and cached as float16 in --cache; variants only re-run the decoder.
Reported per word and variant: hits at 0 false alarms and at <= --max-fa false alarms in the negative stream."""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hark.audio import FeatureStream, RATE
from hark.fast_supported import decode
from hark.residual_acoustic_model import load_model
from hark.text import encode
from scripts.wwb_eval import KEYWORDS, calibrate, readable

THRESHOLDS = [-8., -5., -3.5, -2.6, -2., -1.6, -1.3, -1.1, -.95, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.12, -.06]
GATE = {'min_blank_probability': .02, 'min_frame_probability': .02, 'min_phone_probability': .0005, 'phone_margin': .01, 'mean_phone_probability': .03, 'max_seconds_per_phone': .4}
REDUCED = set('ə ɐ ɚ ɜ ᵻ'.split())
VOWELS = set('a e i o u æ ø œ ɐ ɑ ɒ ɔ ə ɚ ɛ ɜ ɪ ʊ ʌ ʏ ᵻ'.split())
NARROW = [set('a ɐ ʌ ɑ'.split()), set('ə ɐ ɜ ɪ'.split()), set('ɛ e æ a'.split()), set('ɪ i e'.split()), set('ʊ u'.split()), set('ɔ o ɑ'.split()), set('s z'.split()),
          set('t d'.split()), set('k ɡ g'.split()), set('p b'.split()), set('f v'.split()), set('ɹ r ɾ ʁ'.split()), set('ʃ ʒ'.split())]
WIDE = [set('a ɐ ʌ ɑ ɒ ɔ ə'.split()), set('ɛ e æ a ɪ'.split()), set('ɪ i e ᵻ'.split()), set('ʊ u o'.split()), set('ɔ o ɑ ɒ'.split()), set('ɚ ɜ ə ɹ'.split()),
        set('s z ʃ'.split()), set('t d ɾ'.split()), set('k ɡ'.split()), set('p b'.split()), set('f v'.split()), set('ɹ r ɾ'.split()), set('ʃ ʒ'.split()), set('w v'.split()), set('j i'.split())]


def stream_lp(model, ck, chunks):
    front = FeatureStream(ck.get('bins', 64)); state = None; out = []
    with torch.no_grad():
        for x in chunks:
            for i in range(0, len(x), RATE * 8):
                f = front.push(x[i:i + RATE * 8])
                if len(f): lp, state = model(f[None], state); out.append(lp[0].numpy().astype(np.float16))
    return np.concatenate(out)


def negatives(root):
    for dirpath, _, names in sorted(os.walk(root)):
        for name in sorted(names):
            if name.endswith('.trans.txt'):
                for line in open(Path(dirpath) / name):
                    stem, text = line.split(' ', 1)
                    if not any(w in text for k in KEYWORDS for w in k.upper().split()): yield sf.read(str(Path(dirpath) / f'{stem}.flac'), dtype='float32')[0]


def clips(word, data):
    for f in sorted((data / 'repo/audio' / word).glob('*.*')):
        if not readable(f): continue
        x, r = sf.read(str(f), dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if r != RATE: continue
        yield np.concatenate([np.zeros(RATE // 2, np.float32), x / max(1e-4, np.abs(x).max()) * .3, np.zeros(RATE, np.float32)])


def merge(lp, ids, names, tokens, groups, rule):
    """Each target column becomes the log-sum of its group; rule 'reduced' widens only reduced vowels to all vowels."""
    y = lp.astype(np.float32)
    for tok in {int(t) for t in tokens}:
        p = names[tok]; group = {p}
        for g in groups:
            if p in g: group |= g
        if rule == 'reduced' and p in REDUCED: group |= VOWELS
        members = sorted(ids[q] for q in group if q in ids)
        if len(members) > 1: y[:, tok] = np.logaddexp.reduce(lp[:, members].astype(np.float32), axis=1)
    y -= np.logaddexp.reduce(y, axis=1, keepdims=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def variants(tokens):
    t = list(tokens); out = {'strict': ([t], None, None), 'narrow': ([t], NARROW, None), 'wide': ([t], WIDE, 'reduced')}
    edge = [t, t[1:], t[:-1]] if len(t) > 4 else [t]
    out['edge'] = (edge, None, None); out['wide+edge'] = (edge, WIDE, 'reduced'); return out


def events(x, token_sets, th):
    ev = sorted(e[0] for ts in token_sets for e in decode(x, np.asarray(ts), threshold=th, **GATE))
    out = []
    for e in ev:
        if not out or e - out[-1] > 50: out.append(e)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('models', nargs='+'); p.add_argument('--data', default='data/wwb'); p.add_argument('--keywords', default=','.join(KEYWORDS))
    p.add_argument('--cache', default='/tmp/wwb_proxy'); p.add_argument('--max-fa', type=int, default=1); p.add_argument('--threads', type=int, default=6); p.add_argument('--variants', default='strict,narrow,wide,edge,wide+edge')
    a = p.parse_args(); torch.set_num_threads(a.threads); data = Path(a.data); cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True); pool = ThreadPoolExecutor(a.threads); report = {}
    for spec in a.models:
        name, path = spec.split('=', 1); model, ck = load_model(path); model.eval(); ids = ck['vocab']; names = {i: q for q, i in ids.items()}
        neg_f = cache / f'{name}-neg.npy'
        if not neg_f.exists(): np.save(neg_f, stream_lp(model, ck, negatives(data / 'LibriSpeech/test-clean')))
        neg = calibrate(np.load(neg_f).astype(np.float32), 1.25, 2.5); hours = len(neg) / 50 / 3600
        for word in a.keywords.split(','):
            pos = [calibrate(stream_lp(model, ck, [c]).astype(np.float32), 1.25, 2.5) for c in clips(word, data)]
            tokens = encode(word, 'en', ids, ck.get('text_version'), 'espeak-v1')
            for vname, (token_sets, groups, rule) in variants(tokens).items():
                if vname not in a.variants.split(','): continue
                tf = (lambda x: merge(x, ids, names, tokens, groups, rule)) if groups else (lambda x: x)
                n = tf(neg); ps = [tf(x) for x in pos]; rows = []
                for th in THRESHOLDS:
                    fa = len(events(n, token_sets, th)); hits = sum(pool.map(lambda x: bool(events(x, token_sets, th)), ps)); rows.append((th, hits, fa))
                best0 = max([h for _, h, f in rows if f == 0], default=0); best1 = max([h for _, h, f in rows if f <= a.max_fa], default=0)
                report.setdefault(name, {}).setdefault(word, {})[vname] = {'clips': len(pos), 'hits_fa0': best0, f'hits_fa{a.max_fa}': best1, 'rows': rows}
                print(json.dumps({'model': name, 'word': word, 'variant': vname, 'clips': len(pos), 'neg_hours': round(hours, 2), 'miss_fa0': round(1 - best0 / len(pos), 3), f'miss_fa{a.max_fa}': round(1 - best1 / len(pos), 3)}), flush=True)
    Path('reports/wwb/proxy.json').write_text(json.dumps(report))


if __name__ == '__main__':
    main()
