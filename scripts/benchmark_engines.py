"""Unified multi-engine wake-word benchmark runner on WWB (24h @ 10 dB / Clean) and PAI-Bench (EN / DE).

Supports both:
  1. Zero-shot / open-vocabulary text-enrolled engines:
     - Vosk (Kaldi TDNN-F grammar KWS, EN + DE)
     - PocketSphinx (CMUdict keyphrase spotter, EN)
     - sherpa-onnx (Zipformer GigaSpeech 3.3M, EN)
  2. Pre-trained fixed-keyword engines (on their supported WWB words):
     - openWakeWord (ONNX: official `alexa` + community `computer_v2`, `jarvis_v2`)
     - microWakeWord (ESPHome C++/TFLite: official `alexa`)
     - Picovoice Porcupine (`pvporcupine` on all 6 WWB words if `PICOVOICE_ACCESS_KEY` is set; plus published WWB reference)

Uses a two-pass deterministic streaming mixer (`iter_wwb_pcm16`) that reproduces `scripts.wwb_eval.build` bit-for-bit
in < 30 MB of RAM (zero multi-GB allocations or temporary disk files).
"""
import argparse, json, os, tempfile, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

RATE = 16000
FRAME = 512
KEYWORDS = ['alexa', 'computer', 'jarvis', 'smart mirror', 'snowboy', 'view glass']

# Official published 6-word average miss rates at 0.1 FA/h on 24h 10 dB WWB (Picovoice/wake-word-benchmark)
PUBLISHED_WWB_10DB = {
    'porcupine_published': {
        'engine': 'Picovoice Porcupine (v4, publiziert)',
        'category': 'fixed_pretrained',
        'open_vocabulary': False,
        'languages': ['en (feste Modelle)'],
        'fa_per_24h': 2.4,
        'average_miss_rate': 0.027,
        'average_hit_rate_pct': 97.3,
        'note': 'Closed-Source Einzelwort-Modelle (Picovoice WWB Referenz); kein Text-Enrollment ohne Console-Training + AccessKey.',
    },
    'snowboy_published': {
        'engine': 'Kitt.ai Snowboy (publiziert)',
        'category': 'fixed_pretrained',
        'open_vocabulary': False,
        'languages': ['en (Audio-Templates)'],
        'fa_per_24h': 2.4,
        'average_miss_rate': 0.319,
        'average_hit_rate_pct': 68.1,
        'note': 'Audio-Template Engine mit aktiviertem Audio-Frontend (Picovoice WWB Referenz).',
    },
    'pocketsphinx_published': {
        'engine': 'CMU PocketSphinx (publiziert)',
        'category': 'open_vocabulary',
        'open_vocabulary': True,
        'languages': ['en'],
        'fa_per_24h': 2.4,
        'average_miss_rate': 0.520,
        'average_hit_rate_pct': 48.0,
        'note': 'Klassisches Open-Vocabulary Keyphrase-Spotting via CMUdict (Picovoice WWB Referenz).',
    },
}


def _readable(path):
    try:
        return path.is_file() and sf.info(str(path)).frames > 0
    except Exception:
        return False


def _background_files(root, word):
    out = []
    w_up = word.upper()
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith('.trans.txt'):
                for line in open(Path(dirpath) / name):
                    stem, text = line.split(' ', 1)
                    if w_up not in text:
                        out.append(str(Path(dirpath) / f'{stem}.flac'))
    return sorted(out)


def _energy(x):
    n = x.size // FRAME
    return float((x[:n * FRAME].reshape(n, FRAME) ** 2).sum(1).max()) if n else 0.0


def build_wwb_plan(word, data_dir, hours=24.0):
    """Build lightweight index plan (< 1 MB RAM) matching `scripts.wwb_eval.build` RNG sequence 100%."""
    data = Path(data_dir)
    kw_paths = sorted(str(p) for p in (data / 'repo/audio' / word).iterdir() if _readable(p))
    # Verify readability with sf.read once on the ~350 short keyword clips
    valid_kw = []
    kw_sizes = []
    for p in kw_paths:
        try:
            x, r = sf.read(p, dtype='float32')
            if x.size > 0 and r == RATE:
                valid_kw.append(p)
                kw_sizes.append(x.size)
        except Exception:
            pass
    bg_paths = _background_files(data / 'LibriSpeech/test-clean', word)
    bg_sizes = [sf.info(p).frames for p in bg_paths]
    noise_paths = sorted(str(p / 'ch01.wav') for p in (data / 'demand').iterdir())
    noise_audio = []
    for p in noise_paths:
        nx, nr = sf.read(p, dtype='float32')
        assert nr == RATE
        noise_audio.append(nx / max(float(np.abs(nx).max()), 1e-9))

    rng = np.random.RandomState(778)
    bg_rng = np.random.RandomState(778)
    noise_rng = np.random.RandomState(778)

    each = int(hours * 3600 * RATE) // (len(valid_kw) + 1)

    def part_list():
        parts = []
        total = 0
        while total < each:
            bi = bg_rng.randint(low=0, high=len(bg_paths))
            sz = bg_sizes[bi]
            total += sz
            if rng.uniform() < 0.2:
                parts.append(('bg', bi, sz))
            else:
                parts.append(('sil', -1, sz))
        return parts

    parts = part_list()
    times = []
    position = sum(p[2] for p in parts)
    for ki in rng.permutation(np.arange(len(valid_kw))):
        ki = int(ki)
        sz = kw_sizes[ki]
        times.append((position / RATE, (position + sz) / RATE + 0.5))
        position += sz
        parts.append(('kw', ki, sz))
        more = part_list()
        parts.extend(more)
        position += sum(p[2] for p in more)

    noise_seq = []
    ntot = 0
    while ntot < position:
        ni = int(noise_rng.randint(low=0, high=len(noise_audio)))
        noise_seq.append(ni)
        ntot += noise_audio[ni].size

    return {
        'word': word,
        'position': position,
        'hours': position / RATE / 3600.0,
        'n_kw': len(valid_kw),
        'n_bg': len(bg_paths),
        'times': times,
        'parts': parts,
        'kw_paths': valid_kw,
        'bg_paths': bg_paths,
        'noise_audio': noise_audio,
        'noise_seq': noise_seq,
    }


def _iter_raw_blocks(plan, snr=10.0):
    """Yield mixed un-normalized float32 segments per part, identical to `wwb_eval.build`."""
    noise_audio = plan['noise_audio']
    noise_seq = plan['noise_seq']
    ni_ptr = 0
    n_off = 0

    def take_noise(length):
        nonlocal ni_ptr, n_off
        if snr >= 99.0:
            return np.zeros(length, dtype=np.float32)
        out = np.empty(length, dtype=np.float32)
        filled = 0
        while filled < length:
            cur = noise_audio[noise_seq[ni_ptr]]
            avail = cur.size - n_off
            take = min(length - filled, avail)
            out[filled:filled + take] = cur[n_off:n_off + take]
            filled += take
            n_off += take
            if n_off >= cur.size:
                ni_ptr += 1
                n_off = 0
        return out

    for kind, idx, sz in plan['parts']:
        seg = take_noise(sz)
        if kind != 'sil':
            p_path = plan['kw_paths'][idx] if kind == 'kw' else plan['bg_paths'][idx]
            p, _ = sf.read(p_path, dtype='float32')
            p = p / max(float(np.abs(p).max()), 1e-9)
            sp = _energy(p)
            if snr >= 99.0:
                seg += p * 0.3
            elif sp > 0.0:
                seg += p * np.sqrt(_energy(seg) * (10.0 ** (snr / 10.0)) / sp)
        yield seg


def iter_wwb_pcm16(plan, snr=10.0, chunk_samples=1280 * 25):
    """Two-pass exact int16 chunk iterator (< 30 MB RAM) matching `(wwb_eval.build * 32767).astype(np.int16)`."""
    peak = 0.0
    for seg in _iter_raw_blocks(plan, snr=snr):
        if seg.size:
            m = float(np.abs(seg).max())
            if m > peak:
                peak = m
    peak = max(peak, 1e-9)

    buf = np.empty(0, dtype=np.int16)
    for seg in _iter_raw_blocks(plan, snr=snr):
        pcm = ((seg / peak) * 32767.0).astype(np.int16)
        if buf.size:
            pcm = np.concatenate([buf, pcm])
            buf = np.empty(0, dtype=np.int16)
        n_full = (pcm.size // chunk_samples) * chunk_samples
        for i in range(0, n_full, chunk_samples):
            yield pcm[i:i + chunk_samples]
        if n_full < pcm.size:
            buf = pcm[n_full:]
    if buf.size:
        yield buf


def score_events(event_times, kw_times, latency_allowance=0.64, refractory_sec=1.5):
    """Apply refractory deduplication and score against WWB `[start, end + 0.5 + latency]` intervals."""
    dedup = []
    for t in event_times:
        if not dedup or t - dedup[-1] >= refractory_sec:
            dedup.append(t)
    seen = set()
    fa = 0
    j = 0
    for t in dedup:
        while j < len(kw_times) and kw_times[j][1] + latency_allowance < t:
            j += 1
        if j < len(kw_times) and kw_times[j][0] <= t <= kw_times[j][1] + latency_allowance:
            seen.add(j)
        else:
            fa += 1
    return len(seen), fa


def eval_openwakeword_wwb_job(args):
    word, model_spec, data_dir, hours, snr = args
    from openwakeword.model import Model as OWWModel
    t0 = time.monotonic()
    plan = build_wwb_plan(word, data_dir, hours=hours)
    oww = OWWModel(wakeword_models=[model_spec], inference_framework='onnx')
    key = list(oww.models.keys())[0]

    # Collect (time_sec, score) for frames with score >= 0.005 (tiny list in RAM!)
    triggers = []
    sample_pos = 0
    for block in iter_wwb_pcm16(plan, snr=snr, chunk_samples=1280 * 50):
        n = (block.size // 1280) * 1280
        for i in range(0, n, 1280):
            d = oww.predict(block[i:i + 1280])
            sc = float(d[key])
            sample_pos += 1280
            if sc >= 0.005:
                triggers.append((sample_pos / RATE, sc))

    thresholds = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35, 0.42, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    rows = []
    for th in thresholds:
        ev_t = [t for t, s in triggers if s >= th]
        hits, fa = score_events(ev_t, plan['times'], latency_allowance=0.64, refractory_sec=1.5)
        rows.append({'threshold': th, 'hits': hits, 'false_alarms': fa, 'hit_rate_pct': round(100.0 * hits / plan['n_kw'], 2)})
    max_fa = 0.1 * plan['hours']
    ok = [r for r in rows if r['false_alarms'] <= max_fa]
    best = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else {'hits': 0, 'false_alarms': None, 'hit_rate_pct': 0.0}
    return {
        'engine': 'openWakeWord',
        'word': word,
        'model': model_spec,
        'snr_db': snr,
        'hours': round(plan['hours'], 3),
        'keywords': plan['n_kw'],
        'best_at_0.1fa_per_h': best,
        'miss_rate': round(1.0 - best['hits'] / plan['n_kw'], 4),
        'seconds': round(time.monotonic() - t0, 1),
        'rows': rows,
    }


def eval_microwakeword_wwb_job(args):
    word, data_dir, hours, snr = args
    import pymicro_wakeword as mww
    t0 = time.monotonic()
    plan = build_wwb_plan(word, data_dir, hours=hours)
    mw = mww.MicroWakeWord.from_builtin(mww.Model.ALEXA)
    mw_feat = mww.MicroWakeWordFeatures()

    triggers = []
    frame_idx = 0  # 10 ms per feature step (stride=1 -> 10ms, or check mw.stride)
    sample_pos = 0
    for block in iter_wwb_pcm16(plan, snr=snr, chunk_samples=16000 * 10):
        # Process in 1600-sample (0.1s) chunks so timestamps are exact to 0.1s
        for i in range(0, block.size, 1600):
            sub = block[i:i + 1600]
            sample_pos += sub.size
            max_p = 0.0
            for feat in mw_feat.process_streaming(sub.tobytes()):
                p = float(mw.process_streaming_prob(feat))
                if p > max_p:
                    max_p = p
            if max_p >= 0.01:
                triggers.append((sample_pos / RATE, max_p))

    thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.94, 0.97, 0.99, 0.995]
    rows = []
    for th in thresholds:
        ev_t = [t for t, s in triggers if s >= th]
        hits, fa = score_events(ev_t, plan['times'], latency_allowance=0.64, refractory_sec=1.5)
        rows.append({'threshold': th, 'hits': hits, 'false_alarms': fa, 'hit_rate_pct': round(100.0 * hits / plan['n_kw'], 2)})
    max_fa = 0.1 * plan['hours']
    ok = [r for r in rows if r['false_alarms'] <= max_fa]
    best = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else {'hits': 0, 'false_alarms': None, 'hit_rate_pct': 0.0}
    return {
        'engine': 'microWakeWord',
        'word': word,
        'model': 'builtin:alexa',
        'snr_db': snr,
        'hours': round(plan['hours'], 3),
        'keywords': plan['n_kw'],
        'best_at_0.1fa_per_h': best,
        'miss_rate': round(1.0 - best['hits'] / plan['n_kw'], 4),
        'seconds': round(time.monotonic() - t0, 1),
        'rows': rows,
    }


def _vosk_phrase_detections(rec, pcm_bytes, offset_sec, target_phrases):
    """Feed PCM bytes to Vosk KaldiRecognizer and return `(end_sec, min_word_conf)` for matching target phrases."""
    import json as _json
    hits = []

    def check_res(res_str):
        d = _json.loads(res_str)
        words = d.get('result') or []
        if not words:
            return
        w_texts = [w['word'] for w in words]
        for tp in target_phrases:
            tp_words = tp.split()
            L = len(tp_words)
            for k in range(len(w_texts) - L + 1):
                if w_texts[k:k + L] == tp_words:
                    conf = min(float(words[k + j].get('conf', 1.0)) for j in range(L))
                    end_t = offset_sec + float(words[k + L - 1].get('end', 0.0))
                    hits.append((end_t, conf))

    step = 8000 * 2  # 0.5s int16 bytes
    for i in range(0, len(pcm_bytes), step):
        if rec.AcceptWaveform(pcm_bytes[i:i + step]):
            check_res(rec.Result())
    check_res(rec.FinalResult())
    return hits


def eval_vosk_pai_bench(language, bench_dir, model_dir, workers=6):
    """Evaluate Vosk open-vocabulary grammar KWS on PAI-Bench (EN v1 or DE v2-de) + 28.19h negative streams."""
    import vosk
    from hark.audio import read_audio
    from scripts.pai_bench_eval import study, positive_audio, CONDITIONS

    vosk.SetLogLevel(-1)
    t0 = time.monotonic()
    e = study()
    rows = [r for r in (json.loads(s) for s in (Path(bench_dir) / 'manifest.jsonl').read_text().splitlines()) if r['language'] == language]

    # Grammar phrases: in EN, "PAI" is pronounced /paɪ/ ("pie"); in DE, Vosk DE vocabulary uses known words/subwords
    vm = vosk.Model(str(model_dir))
    if language == 'en':
        call_map = {
            'Okay PAI': ['okay pie', 'ok pie'],
            'Hey PAI': ['hey pie'],
        }
    else:
        call_map = {
            'Okay PAI': ['okay kai', 'okay bei', 'okay mai', 'ok kai', 'okay frei'],
            'Hey PAI': ['hey kai', 'hey bei', 'hey mai'],
        }
        # Check which exact words exist in vosk-model-small-de-0.15 graph/words.txt!
        words_txt = Path(model_dir) / 'graph/words.txt'
        if words_txt.exists():
            vocab_words = {ln.split()[0] for ln in words_txt.read_text(errors='ignore').splitlines() if ln.strip()}
            ok_variants = [f'{p} {s}' for p in ('okay', 'ok') for s in ('pai', 'pei', 'pie', 'kai', 'bai') if p in vocab_words and s in vocab_words]
            hey_variants = [f'hey {s}' for s in ('pai', 'pei', 'pie', 'kai', 'bai') if 'hey' in vocab_words and s in vocab_words]
            if ok_variants:
                call_map['Okay PAI'] = ok_variants[:4]
            if hey_variants:
                call_map['Hey PAI'] = hey_variants[:3]

    all_phrases = [p for ps in call_map.values() for p in ps]
    grammar = json.dumps(all_phrases + ['[unk]'])

    # 1. Positives (with confidence per case)
    case_confs = []
    for r in rows:
        t_phrases = call_map[r['call']]
        for cond in CONDITIONS:
            audio, start, end = positive_audio(e, r, cond)
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            rec = vosk.KaldiRecognizer(vm, RATE, grammar)
            rec.SetWords(True)
            evs = _vosk_phrase_detections(rec, pcm16.tobytes(), 0.0, t_phrases)
            valid = [c for t, c in evs if start - 0.5 <= t <= end + 0.5 + 0.64]
            case_confs.append((cond, max(valid) if valid else -1.0))

    # 2. Negatives (28.19h: music, speech_de, speech_en) in parallel
    neg_jobs = [(str(model_dir), grammar, all_phrases, [r['path'] for r in (e.music if sname == 'music' else e.speech[sname.split('_')[1]])])
                for sname in ['music', 'speech_de', 'speech_en']]
    neg_confs = []
    neg_hours = 0.0
    with ProcessPoolExecutor(3) as pool:
        for confs_part, hrs_part in pool.map(_vosk_neg_stream_job, neg_jobs):
            neg_confs.extend(confs_part)
            neg_hours += hrs_part

    thresholds = sorted({0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0} | {round(c, 4) for c in neg_confs})
    curve = []
    for th in thresholds:
        fa = sum(1 for c in neg_confs if c >= th)
        hits = sum(1 for _, c in case_confs if c >= th)
        by_c = {cond: sum(1 for cd, c in case_confs if cd == cond and c >= th) for cond in CONDITIONS}
        curve.append({'threshold': th, 'hits': hits, 'hit_rate_pct': round(100.0 * hits / len(case_confs), 2), 'false_alarms': fa, 'by_condition': by_c})

    frontier = {}
    for target_fa in [0, 1, 2, 5, 10]:
        ok = [r for r in curve if r['false_alarms'] <= target_fa]
        frontier[str(target_fa)] = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else None

    return {
        'engine': f'Vosk 0.3.45 ({Path(model_dir).name})',
        'language': language,
        'grammar': call_map,
        'cases': len(case_confs),
        'negative_hours': round(neg_hours, 2),
        'seconds': round(time.monotonic() - t0, 1),
        'frontier': frontier,
        'top_hits': max(r['hits'] for r in curve),
        'curve': curve,
    }


def _vosk_neg_stream_job(args):
    model_dir, grammar, all_phrases, paths = args
    import vosk
    from hark.audio import read_audio
    vosk.SetLogLevel(-1)
    vm = vosk.Model(model_dir)
    rec = vosk.KaldiRecognizer(vm, RATE, grammar)
    rec.SetWords(True)
    confs = []
    offset_s = 0.0
    for p in paths:
        a = read_audio(p).astype(np.float32)
        pcm16 = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16)
        step = 8000 * 2
        raw = pcm16.tobytes()
        for i in range(0, len(raw), step):
            if rec.AcceptWaveform(raw[i:i + step]):
                d = json.loads(rec.Result())
                words = d.get('result') or []
                w_texts = [w['word'] for w in words]
                for tp in all_phrases:
                    tpw = tp.split(); L = len(tpw)
                    for k in range(len(w_texts) - L + 1):
                        if w_texts[k:k + L] == tpw:
                            confs.append(min(float(words[k + j].get('conf', 1.0)) for j in range(L)))
        offset_s += len(a) / RATE
    d_fin = json.loads(rec.FinalResult())
    words = d_fin.get('result') or []
    w_texts = [w['word'] for w in words]
    for tp in all_phrases:
        tpw = tp.split(); L = len(tpw)
        for k in range(len(w_texts) - L + 1):
            if w_texts[k:k + L] == tpw:
                confs.append(min(float(words[k + j].get('conf', 1.0)) for j in range(L)))
    return confs, offset_s / 3600.0


def _ps_neg_stream_job(paths):
    from pocketsphinx import Decoder
    from hark.audio import read_audio
    with tempfile.NamedTemporaryFile('w', suffix='.kws') as tf:
        tf.write('okay pie /1e-40/\nhey pie /1e-40/\n')
        tf.flush()
        dec = Decoder(kws=tf.name, loglevel='FATAL')
    events = []  # (global_t_sec, prob)
    offset_s = 0.0
    for p in paths:
        a = read_audio(p).astype(np.float32)
        pcm = (np.clip(a, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        dec.start_utt()
        dec.process_raw(pcm, False, True)
        dec.end_utt()
        for s in (dec.seg() or []):
            events.append((offset_s + float(s.end_frame) * 0.01, float(s.prob)))
        offset_s += len(a) / RATE
    return events, offset_s / 3600.0


def eval_pocketsphinx_pai_bench(bench_dir='data/pai-bench-v1'):
    """Evaluate CMU PocketSphinx 5.1 keyphrase spotter on PAI-Bench EN + 28.19h negative streams."""
    from pocketsphinx import Decoder
    from scripts.pai_bench_eval import study, positive_audio, CONDITIONS

    t0 = time.monotonic()
    e = study()
    rows = [r for r in (json.loads(s) for s in (Path(bench_dir) / 'manifest.jsonl').read_text().splitlines()) if r['language'] == 'en']
    call_map = {'Okay PAI': 'okay pie', 'Hey PAI': 'hey pie'}
    with tempfile.NamedTemporaryFile('w', suffix='.kws') as tf:
        tf.write('okay pie /1e-40/\nhey pie /1e-40/\n')
        tf.flush()
        dec = Decoder(kws=tf.name, loglevel='FATAL')

    case_confs = []
    for r in rows:
        target = call_map[r['call']]
        for cond in CONDITIONS:
            audio, start, end = positive_audio(e, r, cond)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            dec.start_utt()
            dec.process_raw(pcm, False, True)
            dec.end_utt()
            valid = [
                float(s.prob) for s in (dec.seg() or [])
                if s.word.strip() == target and start - 0.5 <= float(s.end_frame) * 0.01 <= end + 0.5 + 0.64
            ]
            case_confs.append((cond, max(valid) if valid else -1.0))

    neg_jobs = [[r['path'] for r in (e.music if sname == 'music' else e.speech[sname.split('_')[1]])]
                for sname in ['music', 'speech_de', 'speech_en']]
    stream_events = []
    neg_hours = 0.0
    with ProcessPoolExecutor(3) as pool:
        for evs, hrs in pool.map(_ps_neg_stream_job, neg_jobs):
            stream_events.append(evs)
            neg_hours += hrs

    all_probs = [p for evs in stream_events for _, p in evs] + [c for _, c in case_confs if c >= 0]
    if all_probs:
        qs = np.quantile(all_probs, np.linspace(0.5, 1.0, 120)).tolist()
        top_sorted = sorted((p for evs in stream_events for _, p in evs), reverse=True)[:40]
        thresholds = sorted({round(x, 6) for x in ([0.0] + qs + top_sorted + [x + 1e-6 for x in top_sorted[:15]])})
    else:
        thresholds = [0.0, 0.5, 0.8, 0.9, 1.0]

    def count_fa(th):
        total = 0
        for evs in stream_events:
            last_t = -1e9
            for t, p in evs:
                if p >= th and t - last_t >= 1.0:
                    total += 1
                    last_t = t
        return total

    curve = []
    for th in thresholds:
        fa = count_fa(th)
        hits = sum(1 for _, c in case_confs if c >= th)
        by_c = {cond: sum(1 for cd, c in case_confs if cd == cond and c >= th) for cond in CONDITIONS}
        curve.append({'threshold': th, 'hits': hits, 'hit_rate_pct': round(100.0 * hits / len(case_confs), 2), 'false_alarms': fa, 'by_condition': by_c})

    frontier = {}
    for target_fa in [0, 1, 2, 5, 10]:
        ok = [r for r in curve if r['false_alarms'] <= target_fa]
        frontier[str(target_fa)] = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else None

    return {
        'engine': 'CMU PocketSphinx 5.1.1',
        'language': 'en',
        'grammar': call_map,
        'cases': len(case_confs),
        'negative_hours': round(neg_hours, 2),
        'seconds': round(time.monotonic() - t0, 1),
        'frontier': frontier,
        'top_hits': max(r['hits'] for r in curve),
        'curve': curve,
    }


def eval_vosk_wwb_job(args):
    word, model_dir, data_dir, hours, snr = args
    import vosk
    vosk.SetLogLevel(-1)
    t0 = time.monotonic()
    plan = build_wwb_plan(word, data_dir, hours=hours)
    vm = vosk.Model(str(model_dir))
    phrases = ['snow boy'] if word == 'snowboy' else [word]
    grammar = json.dumps(phrases + ['[unk]'])
    rec = vosk.KaldiRecognizer(vm, RATE, grammar)
    rec.SetWords(True)

    triggers = []  # (end_sec, conf)
    for block in iter_wwb_pcm16(plan, snr=snr, chunk_samples=8000 * 20):
        raw = block.tobytes()
        step = 8000 * 2
        for i in range(0, len(raw), step):
            if rec.AcceptWaveform(raw[i:i + step]):
                d = json.loads(rec.Result())
                words = d.get('result') or []
                w_texts = [w['word'] for w in words]
                for tp in phrases:
                    tpw = tp.split(); L = len(tpw)
                    for k in range(len(w_texts) - L + 1):
                        if w_texts[k:k + L] == tpw:
                            conf = min(float(words[k + j].get('conf', 1.0)) for j in range(L))
                            triggers.append((float(words[k + L - 1]['end']), conf))
    d_fin = json.loads(rec.FinalResult())
    words = d_fin.get('result') or []
    w_texts = [w['word'] for w in words]
    for tp in phrases:
        tpw = tp.split(); L = len(tpw)
        for k in range(len(w_texts) - L + 1):
            if w_texts[k:k + L] == tpw:
                conf = min(float(words[k + j].get('conf', 1.0)) for j in range(L))
                triggers.append((float(words[k + L - 1]['end']), conf))

    thresholds = [0.0, 0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97, 0.985, 0.995, 1.0]
    rows = []
    for th in thresholds:
        ev_t = [t for t, c in triggers if c >= th]
        hits, fa = score_events(ev_t, plan['times'], latency_allowance=0.80, refractory_sec=1.5)
        rows.append({'threshold': th, 'hits': hits, 'false_alarms': fa, 'hit_rate_pct': round(100.0 * hits / plan['n_kw'], 2)})
    max_fa = 0.1 * plan['hours']
    ok = [r for r in rows if r['false_alarms'] <= max_fa]
    best = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else {'hits': 0, 'false_alarms': None, 'hit_rate_pct': 0.0}
    return {
        'engine': 'Vosk (small-en-us-0.15)',
        'word': word,
        'snr_db': snr,
        'hours': round(plan['hours'], 3),
        'keywords': plan['n_kw'],
        'best_at_0.1fa_per_h': best,
        'miss_rate': round(1.0 - best['hits'] / plan['n_kw'], 4),
        'seconds': round(time.monotonic() - t0, 1),
        'rows': rows,
    }


def eval_porcupine_wwb_job(args):
    word, access_key, data_dir, hours, snr = args
    import pvporcupine
    t0 = time.monotonic()
    plan = build_wwb_plan(word, data_dir, hours=hours)
    sensitivities = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
    # Run all sensitivities in parallel inside a single audio stream pass!
    detectors = [pvporcupine.create(access_key=access_key, keywords=[word.lower()], sensitivities=[s]) for s in sensitivities]
    ev_by_s = {s: [] for s in sensitivities}
    sample_pos = 0
    try:
        for block in iter_wwb_pcm16(plan, snr=snr, chunk_samples=512 * 100):
            n = (block.size // 512) * 512
            for i in range(0, n, 512):
                frame = block[i:i + 512]
                sample_pos += 512
                t_sec = sample_pos / RATE
                for s, det in zip(sensitivities, detectors):
                    if det.process(frame) == 0:
                        ev_by_s[s].append(t_sec)
    finally:
        for det in detectors:
            det.delete()

    rows = []
    for s in sensitivities:
        hits, fa = score_events(ev_by_s[s], plan['times'], latency_allowance=0.64, refractory_sec=1.0)
        rows.append({'sensitivity': s, 'hits': hits, 'false_alarms': fa, 'hit_rate_pct': round(100.0 * hits / plan['n_kw'], 2)})
    max_fa = 0.1 * plan['hours']
    ok = [r for r in rows if r['false_alarms'] <= max_fa]
    best = max(ok, key=lambda r: (r['hits'], -r['false_alarms'])) if ok else {'hits': 0, 'false_alarms': None, 'hit_rate_pct': 0.0}
    return {
        'engine': 'Picovoice Porcupine (v4 live)',
        'word': word,
        'snr_db': snr,
        'hours': round(plan['hours'], 3),
        'keywords': plan['n_kw'],
        'best_at_0.1fa_per_h': best,
        'miss_rate': round(1.0 - best['hits'] / plan['n_kw'], 4),
        'seconds': round(time.monotonic() - t0, 1),
        'rows': rows,
    }


def _load_env_key(name):
    if os.environ.get(name):
        return os.environ[name]
    env_p = Path('.env')
    if env_p.exists():
        for ln in env_p.read_text().splitlines():
            if ln.startswith(name + '='):
                return ln.split('=', 1)[1].strip().strip('"').strip("'")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='all', choices=['all', 'wwb-fast', 'pai', 'wwb-vosk', 'porcupine'])
    p.add_argument('--hours', type=float, default=24.0)
    p.add_argument('--snr', type=float, default=10.0)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--porcupine-key', default=None)
    p.add_argument('--out', default='reports/wwb/engines_comparison.json')
    a = p.parse_args()

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(out_path.read_text()) if out_path.exists() else {'published': PUBLISHED_WWB_10DB, 'wwb_measured': {}, 'pai_measured': {}}

    def save():
        out_path.write_text(json.dumps(report, indent=2))

    save()

    if a.suite in ('all', 'pai'):
        res_ps = eval_pocketsphinx_pai_bench('data/pai-bench-v1')
        report['pai_measured']['pocketsphinx_en'] = res_ps
        save()
        print(json.dumps({'done': 'pai_pocketsphinx_en', 'fa0_hits': res_ps['frontier']['0'], 'top_hits': res_ps['top_hits'], 'sec': res_ps['seconds']}), flush=True)

        for lang, bench, mdir in [
            ('en', 'data/pai-bench-v1', 'external/vosk/vosk-model-small-en-us-0.15'),
            ('de', 'data/pai-bench-v2-de', 'external/vosk/vosk-model-small-de-0.15'),
        ]:
            res = eval_vosk_pai_bench(lang, bench, mdir)
            report['pai_measured'][f'vosk_{lang}'] = res
            save()
            print(json.dumps({'done': f'pai_vosk_{lang}', 'fa0_hits': res['frontier']['0'], 'top_hits': res['top_hits'], 'sec': res['seconds']}), flush=True)

    if a.suite in ('all', 'wwb-fast'):
        # 1. microWakeWord (official builtin alexa)
        r_mww = eval_microwakeword_wwb_job(('alexa', 'data/wwb', a.hours, a.snr))
        report['wwb_measured'][f'microwakeword_alexa@{int(a.snr)}dB'] = r_mww
        save()
        print(json.dumps({'done': 'microwakeword_alexa', 'miss': r_mww['miss_rate'], 'hit_pct': r_mww['best_at_0.1fa_per_h']['hit_rate_pct'], 'sec': r_mww['seconds']}), flush=True)

        # 2. openWakeWord (official alexa + community computer_v2 + jarvis_v2) in parallel
        oww_jobs = [
            ('alexa', 'alexa', 'data/wwb', a.hours, a.snr),
            ('computer', 'external/openwakeword/computer_v2.onnx', 'data/wwb', a.hours, a.snr),
            ('jarvis', 'external/openwakeword/jarvis_v2.onnx', 'data/wwb', a.hours, a.snr),
        ]
        with ProcessPoolExecutor(min(a.workers, 3)) as pool:
            for r_oww in pool.map(eval_openwakeword_wwb_job, oww_jobs):
                report['wwb_measured'][f"openwakeword_{r_oww['word']}@{int(a.snr)}dB"] = r_oww
                save()
                print(json.dumps({'done': f"openwakeword_{r_oww['word']}", 'miss': r_oww['miss_rate'], 'hit_pct': r_oww['best_at_0.1fa_per_h']['hit_rate_pct'], 'sec': r_oww['seconds']}), flush=True)

    if a.suite in ('all', 'wwb-vosk'):
        vosk_jobs = [(w, 'external/vosk/vosk-model-small-en-us-0.15', 'data/wwb', a.hours, a.snr) for w in KEYWORDS]
        with ProcessPoolExecutor(min(a.workers, 6)) as pool:
            for r_v in pool.map(eval_vosk_wwb_job, vosk_jobs):
                report['wwb_measured'][f"vosk_{r_v['word']}@{int(a.snr)}dB"] = r_v
                save()
                print(json.dumps({'done': f"vosk_{r_v['word']}", 'miss': r_v['miss_rate'], 'hit_pct': r_v['best_at_0.1fa_per_h']['hit_rate_pct'], 'sec': r_v['seconds']}), flush=True)

    pkey = a.porcupine_key or _load_env_key('PICOVOICE_ACCESS_KEY')
    if a.suite == 'porcupine' or (a.suite == 'all' and pkey):
        if not pkey:
            raise RuntimeError('Porcupine live run requires PICOVOICE_ACCESS_KEY in .env or --porcupine-key')
        for w in KEYWORDS:
            r_p = eval_porcupine_wwb_job((w, pkey, 'data/wwb', a.hours, a.snr))
            report['wwb_measured'][f"porcupine_{w}@{int(a.snr)}dB"] = r_p
            save()
            print(json.dumps({'done': f'porcupine_{w}', 'miss': r_p['miss_rate'], 'hit_pct': r_p['best_at_0.1fa_per_h']['hit_rate_pct'], 'sec': r_p['seconds']}), flush=True)


if __name__ == '__main__':
    main()
