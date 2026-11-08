"""R211: everything that must never enter scaled training.

Collects, from every local manifest/music file, all speakers and origin IDs of non-train material
(dev/test/final/negative streams), plus the R208 reserved future-final VoxPopuli speaker pool, plus every
MUSAN origin used as evaluation music. Output: reports/round211/EXCLUSIONS.json (small, text only)."""
import ast, glob, json, re
from collections import defaultdict
from pathlib import Path
from scripts.evaluate_round15 import sha, save


def norm_text(t):
    return ' '.join(re.sub(r"[^\w']+", ' ', t.lower()).split())


def speaker_key(r):
    src = r.get('source', '')
    raw = r.get('raw_vox_speaker_id') or r.get('speaker_id') or r.get('speaker')
    if raw is None: return None
    raw = str(raw)
    for p in ('vox-de:', 'vox-en:', 'vox:', 'mls-de:', 'mls-en:', 'ls-en:'):
        if raw.startswith(p): raw = raw[len(p):]
    return src, raw


def main():
    speakers = defaultdict(set); origins = defaultdict(set); music = set(); files = {}; texts = set()
    for path in sorted(glob.glob('data/*/protocol.json')) + sorted(glob.glob('data/*/queries.json')):
        d = json.loads(Path(path).read_text()); qs = d.get('queries', []) if isinstance(d, dict) else d
        for q in qs:
            if isinstance(q, dict) and q.get('text'): texts.add(norm_text(q['text']))
        files[path] = sha(path)
    for path in sorted(glob.glob('data/pai-bench-*/manifest.jsonl')):
        for s_ in Path(path).read_text().splitlines():
            r = json.loads(s_); texts.add(norm_text(r['text']))
            for part in re.split(r'[,.?!]', r['text']):
                if norm_text(part): texts.add(norm_text(part))
        files[path] = sha(path)
    paths = sorted(glob.glob('data/*/manifest.jsonl')) + sorted(glob.glob('data/*/music.json'))
    for path in paths:
        files[path] = sha(path)
        if path.endswith('.jsonl'):
            rows = []
            for s in Path(path).read_text().splitlines():
                try: rows.append(json.loads(s))
                except json.JSONDecodeError: pass
        else:
            d = json.loads(Path(path).read_text()); rows = d.get('records', []) if isinstance(d, dict) else d
        for r in rows:
            if not isinstance(r, dict): continue
            split = r.get('split'); role = r.get('role')
            evaluation = split not in (None, 'train') or role in ('negative',) and split != 'train'
            if not evaluation: continue
            src = r.get('source', '')
            if 'musan' in src.lower() or str(r.get('origin_id', '')).startswith('music/'):
                music.add(r.get('origin_id')); continue
            k = speaker_key(r)
            if k: speakers[k[0]].add(k[1])
            if r.get('text'): texts.add(norm_text(r['text']))
            if r.get('origin_id'): origins[src].add(str(r['origin_id']))
    pool = json.loads(Path('reports/round208/POOL_REGISTRATION.json').read_text())
    future = pool['future_unscored_raw_speaker_ids']
    future = ast.literal_eval(future) if isinstance(future, str) else future
    prior = pool['excluded_all_previously_used_raw_ids']
    prior = ast.literal_eval(prior) if isinstance(prior, str) else prior
    vox_future = sorted({str(x) for v in future.values() for x in v})
    result = {
        'version': 'r211-exclusions-v1',
        'speakers_by_source': {k: sorted(v) for k, v in speakers.items()},
        'origin_ids_by_source': {k: sorted(v) for k, v in origins.items()},
        'evaluation_music_origin_ids': sorted(x for x in music if x),
        'voxpopuli_reserved_future_final_speakers': vox_future,
        'voxpopuli_all_previously_used_raw_ids': sorted(str(x) for x in prior),
        'evaluation_texts_normalized': sorted(t for t in texts if t),
        'rule': ('Training rows are dropped if their speaker is listed for the same corpus, if the VoxPopuli speaker is in '
                 'the reserved future-final pool, or if a MUSAN path matches an evaluation music origin. MLS/LibriSpeech '
                 'official train splits are speaker-disjoint from their dev/test splits; the explicit lists are a second guard.'),
        'source_files': files,
    }
    save('reports/round212/EXCLUSIONS.json', result)
    print(json.dumps({k: len(v) for k, v in result['speakers_by_source'].items()} | {
        'music': len(result['evaluation_music_origin_ids']), 'vox_future': len(vox_future), 'texts': len(result['evaluation_texts_normalized'])}))


if __name__ == '__main__':
    main()
