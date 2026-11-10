"""Hark on Picovoice's public wake-word benchmark (real keyword recordings, LibriSpeech test-clean, DEMAND noise at 10 dB).

Reimplements mixer.py and dataset.py from Picovoice/wake-word-benchmark so that no downloaded code runs. The mixing recipe and seed are
the same. Deviations: words are excluded from the background case-correctly (upstream compares a lowercase keyword with uppercase
transcripts, so its exclusion does nothing), and each keyword counts as detected at most once. Metric: miss rate at 1 false alarm per
10 hours, averaged over the six keywords. Text enrollment only: no keyword audio is used for tuning except the detection threshold."""
import argparse, itertools, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hark.audio import FeatureStream, RATE
from hark.fast_supported import decode
from hark.residual_acoustic_model import load_model
from hark.text import encode, encode_multi

def calibrate(lp, t, o):
    x = np.asarray(lp, dtype=np.float32).copy(); x[:, 1:] += o; x /= t
    x -= x.max(1, keepdims=True); x -= np.log(np.exp(x).sum(1, keepdims=True))
    return np.ascontiguousarray(x, dtype=np.float32)


KEYWORDS = ['alexa', 'computer', 'jarvis', 'smart mirror', 'snowboy', 'view glass']
FRAME = 512


class Files:
    def __init__(self, paths): self.paths = sorted(paths); self.rng = np.random.RandomState(778)
    def get(self, i): x, rate = sf.read(self.paths[i], dtype='float32'); assert rate == RATE, self.paths[i]; return x
    def random(self): return self.get(self.rng.randint(low=0, high=len(self.paths)))


def background_files(root, word):
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith('.trans.txt'):
                for line in open(Path(dirpath) / name):
                    stem, text = line.split(' ', 1)
                    if word.upper() not in text: out.append(str(Path(dirpath) / f'{stem}.flac'))
    return out


def readable(path):
    """A few upstream FLAC files cannot be decoded by libsndfile; they are dropped and the keyword count reflects that."""
    try: return path.is_file() and sf.read(str(path), dtype='float32')[0].size > 0
    except Exception: return False


def energy(x):
    n = x.size // FRAME
    return float((x[:n * FRAME].reshape(n, FRAME) ** 2).sum(1).max()) if n else 0.


def build(word, data, hours, snr=10.):
    """snr in dB as in upstream (10); 99 or more means no noise at all."""
    rng = np.random.RandomState(778); keywords = Files([str(p) for p in (data / 'repo/audio' / word).iterdir() if readable(p)])
    background = Files(background_files(data / 'LibriSpeech/test-clean', word)); noise = Files([str(p / 'ch01.wav') for p in (data / 'demand').iterdir()])
    each = int(hours * 3600 * RATE) // (len(keywords.paths) + 1)

    def part_list():
        parts = []; total = 0
        while total < each:
            x = background.random(); total += x.size
            parts.append(x / max(np.abs(x).max(), 1e-9) if rng.uniform() < .2 else x.size)
        return parts

    parts = part_list(); times = []; position = sum(p.size if isinstance(p, np.ndarray) else p for p in parts)
    for i in rng.permutation(np.arange(len(keywords.paths))):
        x = keywords.get(i); times.append((position / RATE, (position + x.size) / RATE + .5)); position += x.size
        parts.append(x / max(np.abs(x).max(), 1e-9)); more = part_list(); parts += more; position += sum(p.size if isinstance(p, np.ndarray) else p for p in more)
    chunks = []; total = 0
    while total < position:
        x = noise.random(); x = x / max(np.abs(x).max(), 1e-9); chunks.append(x); total += x.size
    out = np.concatenate(chunks)[:position]; del chunks
    if snr >= 99: out[:] = 0.
    start = 0
    for p in parts:
        if isinstance(p, np.ndarray):
            segment = out[start:start + p.size]; speech = energy(p)
            if snr >= 99: segment += p * .3
            elif speech > 0: segment += p * np.sqrt(energy(segment) * 10 ** (snr / 10) / speech)
            start += p.size
        else: start += p
    out /= np.abs(out).max()
    return out, times, len(background.paths), len(keywords.paths)


def infer(model, ck, audio, device='cpu'):
    front = FeatureStream(ck.get('bins', 64)); state = None; out = []
    with torch.no_grad():
        for i in range(0, len(audio), RATE * 8):
            f = front.push(audio[i:i + RATE * 8])
            if len(f):
                lp, state = model(f[None].to(device), state); out.append(lp[0].cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def infer_blocked(model, ck, audio, warm, hop, device='cpu', batch=256):
    """Periodic state reset: every `hop` seconds a fresh model state starts `warm` seconds earlier; only its last `hop` seconds are kept.
    Equals running staggered streaming instances on device. Features are position-based, so blocks are slices of the full sequence."""
    front = FeatureStream(ck.get('bins', 64)); feats = []
    for i in range(0, len(audio), RATE * 8):
        f = front.push(audio[i:i + RATE * 8])
        if len(f): feats.append(f)
    f = torch.cat(feats); W = int(round(warm * 50)); H = int(round(hop * 50)); n = -(-len(f) // H)
    f = torch.cat([f.new_zeros(W, f.shape[1]), f, f.new_zeros(n * H - len(f), f.shape[1])])
    out = []
    with torch.no_grad():
        for b in range(0, n, batch):
            idx = torch.arange(b, min(n, b + batch)); win = torch.stack([f[i * H:i * H + W + H] for i in idx.tolist()]).to(device)
            lp, _ = model(win, None); out.append(lp[:, W:].reshape(-1, lp.shape[-1]).cpu().numpy().astype(np.float32))
    return np.concatenate(out)[:sum(len(x) for x in feats)]


SOFT = [set('a ɐ ʌ ɑ'.split()), set('ə ɐ ɜ ɪ'.split()), set('ɛ e æ a'.split()), set('ɪ i e'.split()), set('ʊ u'.split()), set('ɔ o ɑ'.split()), set('s z'.split()),
        set('t d'.split()), set('k ɡ g'.split()), set('p b'.split()), set('f v'.split()), set('ɹ r ɾ ʁ'.split()), set('ʃ ʒ'.split())]


def soften(lp, tokens, vocab):
    """Accept close phone variants: each target column becomes the summed probability of its confusable group."""
    ids = vocab if isinstance(vocab, dict) else {p: i for i, p in enumerate(vocab)}; names = {i: p for p, i in ids.items()}; y = lp.copy()
    for tok in {int(t) for t in tokens}:
        group = {names[tok]}
        for g in SOFT:
            if names[tok] in g: group |= g
        members = sorted(ids[p] for p in group if p in ids)
        if len(members) > 1: y[:, tok] = np.logaddexp.reduce(lp[:, members], axis=1)
    y -= np.logaddexp.reduce(y, axis=1, keepdims=True)
    return np.ascontiguousarray(y, dtype=np.float32)


def decode_sets(x, token_sets, **g):
    if len(token_sets) == 1: return decode(x, token_sets[0], **g)
    ev = sorted((e for tok in token_sets for e in decode(x, tok, **g)), key=lambda e: (float(e[0]), -float(e[2])))
    out = []
    for e in ev:
        if not out or e[0] - out[-1][0] > 50: out.append(e)
    return out


def score(events, times):
    seen = set(); fa = 0; j = 0; times_sorted = times
    for e in events:
        t = e[0] * .02 + .025
        while j < len(times_sorted) and times_sorted[j][1] < t: j += 1
        if j < len(times_sorted) and times_sorted[j][0] <= t <= times_sorted[j][1]: seen.add(j)
        else: fa += 1
    return len(seen), fa


def main():
    p = argparse.ArgumentParser(); p.add_argument('models', nargs='+', help='name=checkpoint'); p.add_argument('--data', default='data/wwb'); p.add_argument('--hours', type=float, default=24.)
    p.add_argument('--keywords', default=','.join(KEYWORDS)); p.add_argument('--threads', type=int, default=8); p.add_argument('--torch-threads', type=int, default=10); p.add_argument('--tag', default='full'); p.add_argument('--soft', action='store_true'); p.add_argument('--device', default='cpu'); p.add_argument('--snrs', default='10'); p.add_argument('--resets', default='0', help='comma list; 0 = continuous, warm:hop seconds = periodic state reset'); p.add_argument('--pron-modes', default='strict', help='comma list of strict,multi'); p.add_argument('--verifier', default='', help='optional Stage-2 BiPhonemeVerifier checkpoint'); a = p.parse_args()
    torch.set_num_threads(a.torch_threads); data = Path(a.data); out = Path('reports/wwb'); out.mkdir(parents=True, exist_ok=True)
    temps = [1.25]; offsets = [2.5, 4.]; thresholds = [-8., -5., -3.5, -2.6, -2., -1.6, -1.3, -1.1, -.95, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.12, -.06]
    s1_v_ths = [-3.5, -2.6, -2., -1.6, -1.3, -1.1, -.8, -.6]
    v_weights = [0.2, 0.35, 0.5, 0.75, 1.0]
    v_c_ths = [-2.6, -2.0, -1.5, -1.1, -0.8, -0.55, -0.35, -0.15, 0.0, 0.15, 0.3, 0.45, 0.65, 0.85, 1.05, 1.3, 1.55, 1.85, 2.2, 2.6, 3.1, 3.7, 4.4, 5.3, 6.4]
    gates = [{'min_blank_probability': .02, 'min_frame_probability': .02, 'min_phone_probability': .0005, 'phone_margin': .01, 'mean_phone_probability': .03, 'max_seconds_per_phone': .4}]
    loaded = {}; pool = ThreadPoolExecutor(a.threads); words = a.keywords.split(',')
    for spec in a.models:
        name, path = spec.split('=', 1); m, ck = load_model(path); loaded[name] = (m.eval().to(a.device), ck)
    verifier, vck, v_front = None, None, None
    if a.verifier:
        from hark.audio import LogMel
        from hark.verifier import load_verifier, score_trigger_queries
        verifier, vck = load_verifier(a.verifier, device=a.device)
        v_front = LogMel(vck.get('bins', 64)).to(a.device)
    snrs = [float(x) for x in a.snrs.split(',')]; resets = a.resets.split(','); pron_modes = a.pron_modes.split(',')
    mtag = lambda n, pm, ver=False: (n if pm == 'strict' else f'{n}+mp') + ('+v' if ver else '')
    label = lambda n, s, r='0': (n if r == '0' else f'{n}/r{r}') + ('' if snrs == [10.] else f'@{s:g}dB')
    ver_flags = [False, True] if verifier is not None else [False]
    results = {label(mtag(n, pm, vf), s, r): {} for n in loaded for pm in pron_modes for vf in ver_flags for s in snrs for r in resets}
    for word, snr in itertools.product(words, snrs):
        t0 = time.monotonic(); audio, times, n_bg, n_kw = build(word, data, a.hours, snr); hours = audio.size / RATE / 3600
        print(json.dumps({'word': word, 'snr': snr, 'hours': round(hours, 2), 'keywords': n_kw, 'background_files': n_bg, 'build_s': round(time.monotonic() - t0)}), flush=True)
        for (name, (model, ck)), reset in itertools.product(loaded.items(), resets):
            t1 = time.monotonic(); lp = infer(model, ck, audio, a.device) if reset == '0' else infer_blocked(model, ck, audio, *map(float, reset.split(':')), a.device)
            tsets_by_pm = {pm: [np.asarray(t) for t in (encode_multi(word, 'en', ck['vocab'], ck.get('text_version'), 'espeak-v1') if pm == 'multi' else [encode(word, 'en', ck['vocab'], ck.get('text_version'), 'espeak-v1')])] for pm in pron_modes}
            v_tsets_by_pm = {}
            if verifier is not None:
                for pm in pron_modes:
                    v_seqs = encode_multi(word, 'en', vck['vocab'], vck.get('text_version'), 'espeak-v1') if pm == 'multi' else [encode(word, 'en', vck['vocab'], vck.get('text_version'), 'espeak-v1')]
                    v_tsets_by_pm[pm] = tuple(tuple(int(z) for z in s) for s in v_seqs)
            rows_by_pm = {pm: [] for pm in pron_modes}
            v_evs_store = {pm: [] for pm in pron_modes}  # list of (t, o, pt, ev_list)
            for t, o in itertools.product(temps, offsets):
                x = calibrate(lp, t, o)
                grid = [{**g, 'threshold': th} for th, g in itertools.product(thresholds, gates)]
                for pm in pron_modes:
                    y = soften(x, tsets_by_pm[pm][0], ck['vocab']) if a.soft else x
                    tsets = tsets_by_pm[pm]
                    ev_lists = list(pool.map(lambda g, _y=y, _ts=tsets: decode_sets(_y, _ts, **g), grid))
                    for pt, ev in zip(grid, ev_lists):
                        hit, fa = score(ev, times)
                        rows_by_pm[pm].append({'T': t, 'offset': o, 'mean_phone_probability': pt['mean_phone_probability'], 'max_seconds_per_phone': pt['max_seconds_per_phone'], 'threshold': pt['threshold'], 'hits': hit, 'false_alarms': fa})
                        if verifier is not None and pt['threshold'] in s1_v_ths:
                            v_evs_store[pm].append((t, o, pt, ev))
            for pm in pron_modes:
                results[label(mtag(name, pm, False), snr, reset)][word] = {'keywords': n_kw, 'hours': hours, 'rows': rows_by_pm[pm]}
            if verifier is not None:
                for pm in pron_modes:
                    vt = v_tsets_by_pm[pm]
                    all_qs = [(int(e[0]), int(e[1]), vt) for _, _, _, ev in v_evs_store[pm] for e in ev]
                    v_map = score_trigger_queries(verifier, audio, all_qs, v_front=v_front, is_features=False, batch_size=64, device=a.device)
                    v_rows = []
                    for t, o, pt, ev in v_evs_store[pm]:
                        s1_th = pt['threshold']
                        ev_v = [(e, float(e[2]), v_map[(int(e[0]), int(e[1]), vt)]) for e in ev]
                        for vw in v_weights:
                            scored_ev = [(e, (1.0 - vw) * s1 + vw * vl) for e, s1, vl in ev_v]
                            for c_th in v_c_ths:
                                hit, fa = score([e for e, sc in scored_ev if sc >= c_th], times)
                                v_rows.append({'T': t, 'offset': o, 'mean_phone_probability': pt['mean_phone_probability'], 'max_seconds_per_phone': pt['max_seconds_per_phone'], 's1_th': s1_th, 'v_weight': vw, 'threshold': c_th, 'hits': hit, 'false_alarms': fa})
                    results[label(mtag(name, pm, True), snr, reset)][word] = {'keywords': n_kw, 'hours': hours, 'rows': v_rows}
            print(json.dumps({'word': word, 'model': name, 'reset': reset, 'pron_modes': pron_modes, 'verifier': bool(verifier), 'infer_and_sweep_s': round(time.monotonic() - t1)}), flush=True)
            del lp
        del audio
        (out / f'{a.tag}.json').write_text(json.dumps(results))
    summary = {}
    for name, by in results.items():
        sample_row = next((r for w in by.values() for r in w['rows']), {})
        scols = [c for c in ['T', 'offset', 'mean_phone_probability', 'max_seconds_per_phone', 's1_th', 'v_weight'] if c in sample_row]
        keys = sorted({tuple(r[c] for c in scols) for w in by.values() for r in w['rows']}); best = None
        for key in keys:
            miss = {}
            for word, w in by.items():
                allowed = w['hours'] / 10.
                ok = [r for r in w['rows'] if tuple(r[c] for c in scols) == key and r['false_alarms'] <= allowed]
                miss[word] = 1 - max([r['hits'] for r in ok], default=0) / w['keywords']
            avg = float(np.mean(list(miss.values())))
            if best is None or avg < best['average_miss_rate']: best = {'average_miss_rate': avg, 'settings': dict(zip(scols, key)), 'per_keyword': miss}
        per_word = {}
        for word, w in by.items():
            allowed = w['hours'] / 10.; per_word[word] = 1 - max([r['hits'] for r in w['rows'] if r['false_alarms'] <= allowed], default=0) / w['keywords']
        summary[name] = {'one_setting_threshold_per_word': best, 'every_setting_per_word_miss': per_word, 'every_setting_per_word_average_miss': float(np.mean(list(per_word.values())))}
    (out / f'{a.tag}-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
