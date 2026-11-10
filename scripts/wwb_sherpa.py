"""sherpa-onnx KWS (zipformer gigaspeech 3.3M, the reference PAI uses) on the Picovoice benchmark streams built by scripts.wwb_eval.

Same mixing, seed and hit rule as Hark (event inside [start, end + 0.5 s]) plus the 0.64 s streaming allowance used on PAI-Bench.
Each 24 h stream is built once, stored as int16 on disk and scanned by one process per (word, threshold)."""
import argparse, json, os, re, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import sherpa_onnx

from scripts.wwb_eval import KEYWORDS, build

M = Path('external/sherpa/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01')
RATE = 16000; CHUNK = 1600; LATENCY = .64


def keyword_line(text):
    words = ' '.join(re.sub(r"[^A-Z' ]+", ' ', text.upper()).split())
    toks = sherpa_onnx.text2token([words], tokens=str(M / 'tokens.txt'), tokens_type='bpe', bpe_model=str(M / 'bpe.model'))[0]
    return ' '.join(toks) + ' @' + words.replace(' ', '_')


def job(args):
    word, threshold, score, work = args
    kf = Path(work) / f'kw-{word.replace(" ", "_")}.txt'; kf.write_text(keyword_line(word) + '\n')
    kws = sherpa_onnx.KeywordSpotter(tokens=str(M / 'tokens.txt'), encoder=str(M / 'encoder-epoch-12-avg-2-chunk-16-left-64.onnx'),
                                     decoder=str(M / 'decoder-epoch-12-avg-2-chunk-16-left-64.onnx'), joiner=str(M / 'joiner-epoch-12-avg-2-chunk-16-left-64.onnx'),
                                     keywords_file=str(kf), num_threads=1, keywords_threshold=threshold, keywords_score=score)
    base = Path(work) / word.replace(' ', '_'); audio = np.load(str(base) + '.npy', mmap_mode='r'); times = json.loads(Path(str(base) + '.json').read_text())
    s = kws.create_stream(); events = []; t0 = time.monotonic()
    for i in range(0, len(audio), CHUNK):
        s.accept_waveform(RATE, (audio[i:i + CHUNK].astype(np.float32) / 32767.))
        while kws.is_ready(s):
            kws.decode_stream(s)
            if kws.get_result(s): events.append((i + CHUNK) / RATE); kws.reset_stream(s)
    seen = set(); fa = 0; j = 0
    for t in events:
        while j < len(times) and times[j][1] + LATENCY < t: j += 1
        if j < len(times) and times[j][0] <= t <= times[j][1] + LATENCY: seen.add(j)
        else: fa += 1
    return {'word': word, 'threshold': threshold, 'score': score, 'hits': len(seen), 'false_alarms': fa, 'events': len(events), 'seconds': round(time.monotonic() - t0)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', default='data/wwb'); p.add_argument('--hours', type=float, default=24.); p.add_argument('--keywords', default=','.join(KEYWORDS))
    p.add_argument('--thresholds', default='.5,.35,.25,.15,.08,.04,.02'); p.add_argument('--score', type=float, default=1.25); p.add_argument('--work', default='/content/wwb/streams')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1)); p.add_argument('--tag', default='sherpa'); a = p.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True); out = Path('reports/wwb'); out.mkdir(parents=True, exist_ok=True); meta = {}
    for word in a.keywords.split(','):
        t0 = time.monotonic(); audio, times, n_bg, n_kw = build(word, Path(a.data), a.hours); hours = audio.size / RATE / 3600
        np.save(work / (word.replace(' ', '_') + '.npy'), (audio * 32767).astype(np.int16)); (work / (word.replace(' ', '_') + '.json')).write_text(json.dumps(times)); del audio
        meta[word] = {'keywords': n_kw, 'hours': hours}; print(json.dumps({'word': word, 'hours': round(hours, 2), 'keywords': n_kw, 'build_s': round(time.monotonic() - t0)}), flush=True)
    jobs = [(w, float(t), a.score, str(work)) for w in meta for t in a.thresholds.split(',')]; rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(job, jobs):
            rows.append(r); print(json.dumps(r), flush=True); (out / f'{a.tag}.json').write_text(json.dumps({'meta': meta, 'rows': rows}))
    per = {}; by_t = {}
    for w, m in meta.items():
        ok = [r for r in rows if r['word'] == w and r['false_alarms'] <= m['hours'] / 10]
        per[w] = 1 - max([r['hits'] for r in ok], default=0) / m['keywords']
    for t in {r['threshold'] for r in rows}:
        by_t[t] = float(np.mean([1 - max([r['hits'] for r in rows if r['word'] == w and r['threshold'] == t and r['false_alarms'] <= m['hours'] / 10], default=0) / m['keywords'] for w, m in meta.items()]))
    best = min(by_t, key=by_t.get)
    summary = {'sherpa': {'one_threshold': {'threshold': best, 'average_miss_rate': by_t[best]}, 'per_word_threshold_miss': per, 'per_word_threshold_average_miss': float(np.mean(list(per.values())))}}
    (out / f'{a.tag}-summary.json').write_text(json.dumps(summary, indent=1)); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
