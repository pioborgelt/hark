"""Tune one Hark checkpoint for the actual call phrases ('Okay PAI', 'Hey PAI') on PAI-Bench (EN v1 / DE v2-de).

Same treatment sherpa got (its score/threshold were chosen on this bench), so the sealed final set stays the only
unbiased comparison. Posteriors are computed once; T/offset applied post hoc. For each calibration, points are visited in
descending hit order and scanned on the 28.19 h negative streams (music first, abort once events exceed --max-fa), bisecting towards each false-alarm target. Reports the best hits at each false-alarm level."""
import argparse, itertools, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from hark.audio import FeatureStream, LogMel, RATE, read_audio
from hark.fast_supported import decode
from hark.residual_acoustic_model import load_model
from hark.text import encode_multi
from hark.verifier import load_verifier, score_trigger_queries
from scripts.pai_bench_eval import study, positive_audio, CONDITIONS
from scripts.r211_fullscan import calibrate
from scripts.wwb_eval import decode_sets


def _stream_feats_f16(audio, bins):
    fs = FeatureStream(bins)
    parts = []
    for i in range(0, len(audio), RATE * 16):
        f = fs.push(audio[i:i + RATE * 16])
        if len(f):
            parts.append(f.numpy().astype(np.float16))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, bins), dtype=np.float16)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--language', required=True); p.add_argument('--bench', required=True); p.add_argument('--threads', type=int, default=8)
    p.add_argument('--temps', default='1,1.25,1.5,2'); p.add_argument('--offsets', default='0,.3,.6,1'); p.add_argument('--max-fa', type=int, default=2); p.add_argument('--limit', type=int, default=40); p.add_argument('--reset', default='0', help='warm:hop seconds periodic state reset; 0 = continuous')
    p.add_argument('--pron-mode', default='strict', choices=['strict', 'multi'])
    p.add_argument('--verifier', default='', help='optional path to Stage-2 BiPhonemeVerifier checkpoint')
    p.add_argument('--s1-max-fa', type=int, default=45, help='max Stage-1 FAs over 28.19h when collecting candidates for Stage-2')
    p.add_argument('--verifier-device', default='cpu')
    a = p.parse_args()
    out = Path('reports/pai-sweep'); out.mkdir(parents=True, exist_ok=True); t0 = time.monotonic()
    e = study(); m, ck = load_model(a.checkpoint); e.model = m.eval(); l = a.language
    verifier, vck = (load_verifier(a.verifier, device=a.verifier_device) if a.verifier else (None, None))
    v_bins = vck.get('bins', 64) if vck else 64
    v_front = LogMel(v_bins) if verifier is not None else None
    if a.reset != '0':
        from scripts.wwb_eval import infer_blocked
        w, h = map(float, a.reset.split(':')); e.infer = lambda audio: infer_blocked(e.model, ck, np.asarray(audio, dtype=np.float32), w, h)
    rows = [r for r in (json.loads(s) for s in (Path(a.bench) / 'manifest.jsonl').read_text().splitlines()) if r['language'] == l]
    if a.pron_mode == 'multi':
        tok = {c: [np.asarray(t) for t in encode_multi(c, l, ck['vocab'], ck.get('text_version'), 'espeak-v1')] for c in ['Okay PAI', 'Hey PAI']}
    else:
        tok = {c: [np.asarray(e.tokens({'text': c, 'language': l, 'id': f'{l}:{c}'}))] for c in ['Okay PAI', 'Hey PAI']}
    v_tok = {}
    if verifier is not None:
        for c in ['Okay PAI', 'Hey PAI']:
            if a.pron_mode == 'multi':
                seqs = encode_multi(c, l, vck['vocab'], vck.get('text_version'), 'espeak-v1')
            else:
                seqs = [e.tokens({'text': c, 'language': l, 'id': f'{l}:{c}'})] if vck['vocab'] == ck['vocab'] else encode_multi(c, l, vck['vocab'], vck.get('text_version'), 'espeak-v1')[:1]
            v_tok[c] = tuple(tuple(int(x) for x in s) for s in seqs)

    cases = []
    pos_feats = []
    for r in rows:
        for cond in CONDITIONS:
            audio, start, end = positive_audio(e, r, cond)
            cases.append((tok[r['call']], start, end, cond, e.infer(audio), r['call']))
            if verifier is not None:
                with torch.no_grad():
                    pos_feats.append(v_front(torch.as_tensor(audio, dtype=torch.float32))[0::2].numpy().astype(np.float16))

    streams = {}
    stream_feats = {}
    for n in ['music', 'speech_de', 'speech_en']:
        cat_a = np.concatenate([read_audio(r['path']) for r in (e.music if n == 'music' else e.speech[n.split('_')[1]])])
        streams[n] = e.infer(cat_a)
        if verifier is not None:
            stream_feats[n] = _stream_feats_f16(cat_a, v_bins)
        del cat_a
    print(json.dumps({'ready_seconds': time.monotonic() - t0, 'cases': len(cases), 'verifier': bool(verifier)}), flush=True)

    base = {'min_blank_probability': .1, 'min_frame_probability': .1, 'min_phone_probability': .003, 'phone_margin': .1}
    th_low = -2.1 if verifier is not None else -1.5
    grid = [{**base, 'threshold': float(th), 'mean_phone_probability': mp, 'max_seconds_per_phone': ms}
            for th, mp, ms in itertools.product(np.arange(th_low, -.25, .1).round(3), [.1, .14, .18, .225, .28], [.16, .2, .26])]
    pool = ThreadPoolExecutor(a.threads); s1_res = []
    scan_cap = max(a.max_fa, a.s1_max_fa) if verifier is not None else a.max_fa
    probe_targets = (0, a.max_fa, 10, 22, scan_cap) if verifier is not None else (0, a.max_fa)

    for t, o in itertools.product([float(x) for x in a.temps.split(',')], [float(x) for x in a.offsets.split(',')]):
        lps = [calibrate(c[4], t, o) for c in cases]
        def hits(pt):
            n = 0; by = dict.fromkeys(CONDITIONS, 0)
            for c, lp in zip(cases, lps):
                ev = decode_sets(lp, c[0], **pt)
                if any(c[1] - .5 <= x[0] * .02 + .025 <= c[2] + .5 for x in ev): n += 1; by[c[3]] += 1
            return n, by
        pos = list(pool.map(hits, grid)); cal = {k: calibrate(v, t, o) for k, v in streams.items()}
        def fa(pt):
            total = 0
            for name in ['music', 'speech_de', 'speech_en']:
                total += sum(len(decode_sets(cal[name], k, **pt)) for k in tok.values())
                if total > scan_cap: return None
            return total
        order = sorted(range(len(grid)), key=lambda i: (-pos[i][0], -grid[i]['threshold']))
        fas = {}
        def probe(k):
            i = order[k]
            if i not in fas: fas[i] = fa(grid[i])
            return fas[i]
        for target in probe_targets:
            lo, hi = 0, len(order) - 1
            while lo < hi:
                mid = (lo + hi) // 2; v = probe(mid)
                if v is not None and v <= target: hi = mid
                else: lo = mid + 1
            for k in range(max(0, lo - 4), min(len(order), lo + 3)): probe(k)
        for i in list(fas):
            s1_res.append({'T': t, 'offset': o, 'grid_idx': i, 'parameters': grid[i], 'hits': pos[i][0], 'by_condition': pos[i][1], 'false_alarms': fas[i]})
        ok = [r for r in s1_res if r['T'] == t and r['offset'] == o and r['false_alarms'] is not None]
        print(json.dumps({'T': t, 'offset': o, 'top_hits': max(x[0] for x in pos), 'scanned': len(fas),
                          'best_fa0': max([r['hits'] for r in ok if r['false_alarms'] == 0], default=None),
                          'best_fa_s1_cap': max([r['hits'] for r in ok], default=None), 'seconds': time.monotonic() - t0}), flush=True)

    if verifier is None:
        res = [{k: v for k, v in r.items() if k != 'grid_idx'} for r in s1_res]
    else:
        # Select Pareto-optimal + top-recall Stage-1 candidate configs across FA buckets [0..scan_cap]
        valid_s1 = [r for r in s1_res if r['false_alarms'] is not None and r['false_alarms'] <= scan_cap]
        selected = []
        seen_cfg = set()
        for bucket_hi in [0, 2, 5, 10, 18, 28, scan_cap]:
            cand = [r for r in valid_s1 if r['false_alarms'] <= bucket_hi]
            cand.sort(key=lambda r: (-r['hits'], r['false_alarms']))
            for r in cand[:3]:
                cid = (r['T'], r['offset'], r['grid_idx'])
                if cid not in seen_cfg:
                    seen_cfg.add(cid)
                    selected.append(r)

        # Collect Stage-1 events for selected configs and batch-score unique windows per case/stream
        pos_ev_by_cfg = {}  # cid -> list over cases of list of (s1_score, q_tuple)
        neg_ev_by_cfg = {}  # cid -> list of (stream_name, s1_score, q_tuple)
        pos_queries_by_case = [[] for _ in range(len(cases))]
        neg_queries_by_stream = {n: [] for n in ['music', 'speech_de', 'speech_en']}

        by_to = {}
        for r in selected:
            by_to.setdefault((r['T'], r['offset']), []).append(r)

        for (t, o), cfg_list in by_to.items():
            lps = [calibrate(c[4], t, o) for c in cases]
            cal = {k: calibrate(v, t, o) for k, v in streams.items()}
            for r in cfg_list:
                cid = (t, o, r['grid_idx'])
                pt = r['parameters']
                p_list = []
                for ci, (c, lp) in enumerate(zip(cases, lps)):
                    ev = decode_sets(lp, c[0], **pt)
                    matched = []
                    vt = v_tok[c[5]]
                    for x in ev:
                        if c[1] - .5 <= x[0] * .02 + .025 <= c[2] + .5:
                            q = (int(x[0]), int(x[1]), vt)
                            matched.append((float(x[2]), q))
                            pos_queries_by_case[ci].append(q)
                    p_list.append(matched)
                pos_ev_by_cfg[cid] = p_list

                n_list = []
                for sname in ['music', 'speech_de', 'speech_en']:
                    for call_name, tset in tok.items():
                        vt = v_tok[call_name]
                        for x in decode_sets(cal[sname], tset, **pt):
                            q = (int(x[0]), int(x[1]), vt)
                            n_list.append((sname, float(x[2]), q))
                            neg_queries_by_stream[sname].append(q)
                neg_ev_by_cfg[cid] = n_list

        t_v0 = time.monotonic()
        pos_vscores = [
            score_trigger_queries(verifier, pos_feats[ci], qs, is_features=True, device=a.verifier_device) if qs else {}
            for ci, qs in enumerate(pos_queries_by_case)
        ]
        neg_vscores = {
            sname: (score_trigger_queries(verifier, stream_feats[sname], qs, is_features=True, device=a.verifier_device) if qs else {})
            for sname, qs in neg_queries_by_stream.items()
        }
        print(json.dumps({
            'verifier_scored_s': round(time.monotonic() - t_v0, 2),
            'selected_s1_configs': len(selected),
            'unique_neg_triggers': sum(len(v) for v in neg_vscores.values()),
        }), flush=True)

        res = []
        for r in selected:
            cid = (r['T'], r['offset'], r['grid_idx'])
            p_list = pos_ev_by_cfg[cid]
            n_list = neg_ev_by_cfg[cid]
            for vw in [0.0, 0.15, 0.30, 0.50, 0.75, 1.0]:
                neg_comb = sorted(((1.0 - vw) * s1 + vw * neg_vscores[sn][q] for sn, s1, q in n_list), reverse=True)
                # Candidate thresholds achieving exact 0..max_fa false alarms
                cand_ths = [-1e9]
                for k_fa in range(min(len(neg_comb), a.max_fa + 3)):
                    cand_ths.append(neg_comb[k_fa] + 1e-6)
                for c_th in sorted(set(cand_ths)):
                    fa_cnt = sum(1 for sc in neg_comb if sc >= c_th)
                    if fa_cnt > a.max_fa:
                        continue
                    hits_cnt = 0
                    by_c = dict.fromkeys(CONDITIONS, 0)
                    for ci, matched in enumerate(p_list):
                        if matched and any((1.0 - vw) * s1 + vw * pos_vscores[ci][q] >= c_th for s1, q in matched):
                            hits_cnt += 1
                            by_c[cases[ci][3]] += 1
                    res.append({
                        'T': r['T'],
                        'offset': r['offset'],
                        'parameters': {**r['parameters'], 'v_weight': vw, 'combined_threshold': round(float(c_th), 5)},
                        's1_false_alarms': r['false_alarms'],
                        'hits': hits_cnt,
                        'by_condition': by_c,
                        'false_alarms': fa_cnt,
                    })

    frontier = {str(n): max([r for r in res if r['false_alarms'] is not None and r['false_alarms'] <= n], key=lambda r: r['hits'], default=None) for n in range(a.max_fa + 1)}
    (out / f'{a.tag}.json').write_text(json.dumps({'checkpoint': a.checkpoint, 'verifier': a.verifier or None, 'pron_mode': a.pron_mode, 'language': l, 'bench': a.bench, 'cases': len(cases), 'frontier': frontier, 'rows': res}))
    print(json.dumps({'frontier': {k: v and (v['hits'], v['false_alarms'], v['T'], v['offset'], v['by_condition']) for k, v in frontier.items()}}), flush=True)


if __name__ == '__main__':
    main()
