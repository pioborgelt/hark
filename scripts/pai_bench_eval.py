"""Evaluate wakeword systems on PAI-Bench v1 (positives) and the closed 28.19 h negative streams (false alarms).

Positive conditions mirror the Hark development protocol: clean, music_10db, music_0db, room_low, context
(same music/speech pools and augmentation code via the R210 Study). A hit is an event between the start of the
utterance and the end of the call + 0.5 s (+0.64 s streaming allowance for sherpa). False alarms: events of the
language's call keywords on speech_de, speech_en and music (all 28.19 h, per language keyword set).
Systems: Hark checkpoints with explicit calibration + decoder parameters, sherpa-onnx KWS with keyword variants."""
import argparse, json, multiprocessing, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from hark.audio import read_audio, RATE

BENCH = Path('data/pai-bench-v1'); OUT = Path('reports/pai-bench-v1')
CONDITIONS = ['clean', 'music_10db', 'music_0db', 'room_low', 'context']
_E = None


def study():
    global _E
    if _E is None:
        from scripts.round210_development import Study
        plan = json.loads(Path('reports/round210/EVALUATION_PLAN.json').read_text())
        _E = Study(plan['reference']['checkpoint'])
    return _E


def positive_audio(e, row, cond):
    """Same augmentation code path as the development protocol, applied to a bench utterance."""
    rec = {'id': 'pai-' + row['id'], 'path': row['path'], 'language': row['language'], 'phone_end': row['call_end_upper_seconds']}
    e.positive_bundles.setdefault(rec['id'], e.positive_bundles[next(iter(e.positive_bundles))])
    return e.positive_audio(rec, cond)


def sherpa_job(args):
    variant, threshold, rows, neg = args
    from scripts.reference_sherpa_r210 import spotter, keyword_line, run_stream, CHUNK
    e = study(); kwdir = OUT / 'kw'; kwdir.mkdir(parents=True, exist_ok=True)
    texts = variant['keywords']; lines = [keyword_line(t) for t in texts]
    kf = kwdir / (variant['name'] + '.txt'); kf.write_text('\n'.join(lines) + '\n')
    kws = spotter(threshold, variant.get('score', 1.0), kf)
    if neg:
        rs = e.music if neg == 'music' else e.speech[neg.split('_')[1]]
        s = kws.create_stream(); n = 0; sec = 0.
        for r in rs:
            a = read_audio(r['path']).astype('float32'); sec += len(a) / RATE
            for i in range(0, len(a), CHUNK):
                s.accept_waveform(RATE, a[i:i + CHUNK])
                while kws.is_ready(s):
                    kws.decode_stream(s)
                    if kws.get_result(s): n += 1; kws.reset_stream(s)
        return {'stream': neg, 'events': n, 'seconds': sec}
    out = []
    for row in rows:
        for cond in CONDITIONS:
            audio, start, end = positive_audio(e, row, cond)
            ev = run_stream(kws, audio)
            out.append({'id': row['id'], 'language': row['language'], 'condition': cond,
                        'hit': any(start - .5 <= t <= end + .5 + .64 for t, _ in ev)})
    return out


def hark_system(spec, rows, threads, positives_only=False):
    import torch
    from hark.residual_acoustic_model import load_model
    from hark.fast_supported import decode
    from scripts.r211_fullscan import calibrate
    e = study(); m, ck = load_model(spec['checkpoint']); e.model = m.eval()
    torch.set_num_threads(threads)
    t, o = spec.get('T', 1.0), spec.get('offset', 0.)
    tok = {l: {c: np.asarray(e.tokens({'text': c, 'language': l, 'id': f'{l}:{c}'})) for c in ['Okay PAI', 'Hey PAI']} for l in ['de', 'en']}
    out = []
    for row in rows:
        l = row['language']
        for cond in CONDITIONS:
            audio, start, end = positive_audio(e, row, cond)
            lp = calibrate(e.infer(audio), t, o)
            ev = decode(lp, tok[l][row['call']], **spec['params'][l])
            out.append({'id': row['id'], 'language': l, 'condition': cond,
                        'hit': bool(len([x for x in ev if start - .5 <= x[0] * .02 + .025 <= end + .5]))})
    neg = {(n, k): 0 for n in ['speech_de', 'speech_en', 'music'] for k in ['de', 'en']} | {(n, 'seconds'): 1e-9 for n in ['speech_de', 'speech_en', 'music']}
    for name in ([] if positives_only else ['speech_de', 'speech_en', 'music']):
        rs = e.music if name == 'music' else e.speech[name.split('_')[1]]
        lp = calibrate(e.infer(np.concatenate([read_audio(r['path']) for r in rs])), t, o)
        for l in ['de', 'en']:
            neg[(name, l)] = sum(len(decode(lp, tok[l][c], **spec['params'][l])) for c in tok[l])
        neg[(name, 'seconds')] = sum(r['seconds'] for r in rs)
        del lp
    return out, neg


def summarize(name, pos, neg_by_lang, hours):
    res = {}
    for l in ['de', 'en']:
        p = [x for x in pos if x['language'] == l]
        res[l] = {'hits': sum(x['hit'] for x in p), 'cases': len(p),
                  'by_condition': {c: sum(x['hit'] for x in p if x['condition'] == c) for c in CONDITIONS},
                  'false_alarms': neg_by_lang[l], 'fa_per_hour': neg_by_lang[l] / hours}
    return {'system': name, **res}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--systems', required=True, help='JSON file with system specs')
    p.add_argument('--workers', type=int, default=6); p.add_argument('--tag', required=True)
    p.add_argument('--bench', default=str(BENCH)); p.add_argument('--positives-only', action='store_true'); a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(s) for s in (Path(a.bench) / 'manifest.jsonl').read_text().splitlines()]
    specs = json.loads(Path(a.systems).read_text()); results = []; t0 = time.monotonic()
    ctx = multiprocessing.get_context('spawn')
    for spec in specs:
        if spec['kind'] == 'sherpa':
            chunks = [rows[i::a.workers] for i in range(a.workers)]
            with ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
                pos = [x for c in pool.map(sherpa_job, [(spec, spec['threshold'], c, None) for c in chunks]) for x in c]
                negs = [] if a.positives_only else list(pool.map(sherpa_job, [(spec, spec['threshold'], None, n) for n in ['speech_de', 'speech_en', 'music']]))
            hours = max(sum(n['seconds'] for n in negs) / 3600, 1e-9); total = sum(n['events'] for n in negs)
            r = summarize(spec['name'], pos, {'de': total, 'en': total}, hours)
            r['note'] = 'sherpa is English-only: one keyword set for both languages; false alarms counted once over all streams'
        else:
            pos, neg = hark_system(spec, rows, a.workers, a.positives_only)
            hours = sum(neg[(n, 'seconds')] for n in ['speech_de', 'speech_en', 'music']) / 3600
            r = summarize(spec['name'], pos, {l: sum(neg[(n, l)] for n in ['speech_de', 'speech_en', 'music']) for l in ['de', 'en']}, hours)
        r['negative_hours'] = hours; r['spec'] = spec; results.append(r)
        print(json.dumps({k: (v if k != 'spec' else spec['name']) for k, v in r.items()}), flush=True)
        (OUT / f'{a.tag}.json').write_text(json.dumps({'results': results, 'bench': a.bench, 'positives_only': a.positives_only, 'utterances': len(rows),
                                                       'seconds': time.monotonic() - t0}, indent=1))


if __name__ == '__main__':
    main()
