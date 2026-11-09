"""R211 scaled acoustic training (runs on a Colab VM; also smoke-testable locally).

Stage `prepare`: download pinned parquet shards (MLS-DE, LibriSpeech-960, part of VoxPopuli DE/EN), drop
excluded speakers/origins, phonemize transcripts with the unchanged Hark espeak front end, and write a small
text-only index per shard. Stage `train`: stream audio from the shards, augment on the fly (real/synthetic
RIR, MUSAN music/noise without evaluation origins, own training music), compute the unchanged
logmel64-power-v1 features at 50 fps on the GPU, and train several Hark AcousticModel sizes on identical
batches with plain CTC. No audio or feature copies are written; checkpoints are small.
"""
import argparse, hashlib, io, json, math, os, random, time, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly

from hark.audio import LogMel, RATE, WINDOW, HOP
from hark.model import AcousticModel
from hark.text import encode

TEXT_VERSION = 'espeak-case-v2'
FEATURE_VERSION = 'logmel64-power-v1'
LS_REV = '71cacbfb7e2354c4226d01e70d77d5fca3d04ba1'
MLS_REV = '2e83e61823b4c47dcbcb1980bb88601274127609'
VOX_REV = '42f01879c780b4a2e90ec0b4f616c2ece526e4f1'
MUSAN_REPO = 'corypaik/musan'
MUSAN_REV = '76f9882cfa4475efe11508ac9aa32722f84ca5b7'


CV_DE_REPO = 'q1805/common_voice_26_0_de'
CV_DE_REV = '5ad4b18b75971253d982ba4119eb575f306a3e90'
CV_EN_REPO = 'fixie-ai/common_voice_17_0'  # Common Voice 17 English, CC0, crowd-sourced microphones and accents
CV_EN_REV = '34f78a43893414e7b6e271ba94c1d5e05f18b239'


def norm_text(t):
    import re
    return ' '.join(re.sub(r"[^\w']+", ' ', t.lower()).split())


def source_specs(vox_de_shards, vox_en_shards, cv_de_shards=0, cv_en_shards=0):
    from huggingface_hub import list_repo_files
    ls = sorted(f for f in list_repo_files('openslr/librispeech_asr', repo_type='dataset', revision=LS_REV)
                if f.startswith(('all/train.clean.100/', 'all/train.clean.360/', 'all/train.other.500/')) and f.endswith('.parquet'))
    specs = [dict(name='mls_de', repo='facebook/multilingual_librispeech', rev=MLS_REV, file=f'german/train-{i:05d}-of-00063.parquet',
                  lang='de', text='transcript', speaker='speaker_id', origin='id', exclusion_source='facebook/multilingual_librispeech')
             for i in range(63)]
    specs += [dict(name='ls_en', repo='openslr/librispeech_asr', rev=LS_REV, file=f, lang='en', text='text', speaker='speaker_id',
                   origin='id', exclusion_source='openslr/librispeech_asr') for f in ls]
    specs += [dict(name='vox_de', repo='facebook/voxpopuli', rev=VOX_REV, file=f'de/train-{i:05d}-of-00016.parquet', lang='de',
                   text='normalized_text', speaker='speaker_id', origin='audio_id', exclusion_source='facebook/voxpopuli')
              for i in range(vox_de_shards)]
    specs += [dict(name='vox_en', repo='facebook/voxpopuli', rev=VOX_REV, file=f'en/train-{i:05d}-of-00030.parquet', lang='en',
                   text='normalized_text', speaker='speaker_id', origin='audio_id', exclusion_source='facebook/voxpopuli')
              for i in range(vox_en_shards)]
    specs += [dict(name='cv_de', repo=CV_DE_REPO, rev=CV_DE_REV, file=f'data/audio/validated-{i:05d}-of-00048.parquet', lang='de',
                   text='sentence', speaker='client_id', origin='path', exclusion_source=['pklumpp/CommonPhoneDataset', 'CommonPhone'])
              for i in range(cv_de_shards)]
    specs += [dict(name='cv_en', repo=CV_EN_REPO, rev=CV_EN_REV, file=f'en/train-{i:05d}-of-00090.parquet', lang='en',
                   text='sentence', speaker='client_id', origin='path', exclusion_source=['pklumpp/CommonPhoneDataset', 'CommonPhone'])
              for i in range(cv_en_shards)]
    return specs


def file_sha(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def holdout(speaker):
    return int(hashlib.sha256(('r211-holdout:' + speaker).encode()).hexdigest()[:8], 16) % 100 == 0


def download(repo, filename, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    delay = 30
    for attempt in range(12):
        try:
            return hf_hub_download(repo, filename, repo_type='dataset', revision=revision, cache_dir=cache_dir)
        except Exception as e:
            msg = str(e)
            print(json.dumps({'download_retry': filename, 'attempt': attempt, 'error': msg[:200]}), flush=True)
            time.sleep(delay + random.random() * 10); delay = min(delay * 2, 600)
    raise RuntimeError('download failed repeatedly: ' + filename)


def index_shard(args):
    spec, root, exclusions_path, vocab = args
    from huggingface_hub import hf_hub_download
    out = Path(root) / 'index' / (spec['file'].replace('/', '__') + '.json')
    if out.exists(): return json.loads(out.read_text())['summary']
    path = spec['local_path'] if spec.get('local_path') else download(spec['repo'], spec['file'], spec['rev'], str(Path(root) / 'hf'))
    ex = json.loads(Path(exclusions_path).read_text())
    sources = spec['exclusion_source'] if isinstance(spec['exclusion_source'], list) else [spec['exclusion_source']]
    bad_speakers = set().union(*[set(ex['speakers_by_source'].get(x, [])) for x in sources])
    bad_texts = set(ex.get('evaluation_texts_normalized', []))
    if spec['repo'] == 'facebook/voxpopuli':
        bad_speakers |= set(ex['voxpopuli_reserved_future_final_speakers'])
    bad_origins = {o.split(':', 1)[-1] for x in sources for o in ex['origin_ids_by_source'].get(x, [])}
    table = pq.read_table(path, columns=[spec['text'], spec['speaker'], spec['origin']])
    texts, speakers, origins = (table.column(i).to_pylist() for i in range(3))
    rows = []; counts = dict(excluded_speaker=0, excluded_origin=0, excluded_text=0, unpronounceable=0, empty=0)
    for i, (t, s, o) in enumerate(zip(texts, speakers, origins)):
        s = str(s); o = str(o)
        if s in bad_speakers: counts['excluded_speaker'] += 1; continue
        if o in bad_origins or ('ls-' + o) in bad_origins or ('mls-' + o) in bad_origins: counts['excluded_origin'] += 1; continue
        if not t or not t.strip(): counts['empty'] += 1; continue
        if norm_text(t) in bad_texts: counts['excluded_text'] += 1; continue
        pron = 'de-lexicon-v1' if spec['lang'] == 'de' else 'espeak-v1'
        try: ids = encode(t, spec['lang'], vocab, TEXT_VERSION, pron)
        except ValueError: counts['unpronounceable'] += 1; continue
        rows.append([i, s, ids, holdout(sources[0] + ':' + s)])
    summary = dict(file=spec['file'], name=spec['name'], lang=spec['lang'], path=path, sha256=file_sha(path), rows=len(texts),
                   kept=len(rows), **counts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'rows': rows}))
    return summary


def prepare_augmentation(root, exclusions_path, music_dir):
    """MUSAN music+noise (minus every evaluation origin), own training music and real RIRs.
    Stored as one concatenated int16 array + offsets per kind so dataloader workers share it via mmap."""
    from huggingface_hub import hf_hub_download, list_repo_files
    out = Path(root) / 'aug'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'done.json').exists(): return json.loads((out / 'done.json').read_text())
    ex = set(json.loads(Path(exclusions_path).read_text())['evaluation_music_origin_ids'])
    ex_names = {e.rsplit('/', 1)[-1] for e in ex}
    files = [f for f in list_repo_files(MUSAN_REPO, repo_type='dataset', revision=MUSAN_REV) if f.endswith('.parquet') and f.startswith(('music/', 'noise/'))]
    clips = {'music': [], 'noise': []}; skipped = 0; used = []
    for f in sorted(files):
        key = f.split('/', 1)[0]
        p = download(MUSAN_REPO, f, MUSAN_REV, str(Path(root) / 'hf'))
        t = pq.read_table(p, columns=['audio', 'path'])
        for a, path in zip(t.column('audio').to_pylist(), t.column('path').to_pylist()):
            name = (path or a.get('path') or '').rsplit('/', 1)[-1]
            if path in ex or name in ex_names: skipped += 1; continue
            x, r = sf.read(io.BytesIO(a['bytes']), dtype='float32', always_2d=True); x = x.mean(1)
            if r != RATE: g = math.gcd(r, RATE); x = resample_poly(x, RATE // g, r // g)
            clips[key].append((np.clip(x, -1, 1) * 32767).astype(np.int16)); used.append(path)
        Path(p).unlink()
    for w in sorted(Path(music_dir).glob('*.wav')):
        x, r = sf.read(w, dtype='float32', always_2d=True); clips['music'].append((x.mean(1) * 32767).astype(np.int16))
    import urllib.request
    z = Path(root) / 'rirs_noises.zip'
    if not z.exists(): urllib.request.urlretrieve('https://www.openslr.org/resources/28/rirs_noises.zip', z)
    rirs = []
    with zipfile.ZipFile(z) as arc:
        for n in arc.namelist():
            real = 'real_rirs_isotropic_noises' in n and '_rir_' in n
            sim = 'simulated_rirs' in n and random.Random(n).random() < .15
            if n.endswith('.wav') and (real or sim):
                x, r = sf.read(io.BytesIO(arc.read(n)), dtype='float32', always_2d=True); x = x[:, 0]
                if r != RATE: g = math.gcd(r, RATE); x = resample_poly(x, RATE // g, r // g)
                x = x[np.argmax(np.abs(x)):][:RATE]
                rirs.append(np.pad(x / (np.abs(x).max() + 1e-9), (0, RATE - len(x))).astype(np.float32))
    z.unlink()
    for k, v in clips.items():
        np.save(out / f'{k}.npy', np.concatenate(v)); np.save(out / f'{k}_offsets.npy', np.cumsum([0] + [len(x) for x in v]))
    np.save(out / 'rirs.npy', np.stack(rirs))
    done = dict(music=len(clips['music']), noise=len(clips['noise']), rirs=len(rirs), skipped_evaluation_music=skipped,
                music_hours=sum(len(x) for x in clips['music']) / RATE / 3600, noise_hours=sum(len(x) for x in clips['noise']) / RATE / 3600,
                musan_revision=MUSAN_REV)
    (out / 'done.json').write_text(json.dumps(done)); (out / 'used_musan_paths.json').write_text(json.dumps(used)); return done


class Pool:
    def __init__(self, aug, kind):
        self.ok = (aug / f'{kind}.npy').exists()
        if self.ok: self.data = np.load(aug / f'{kind}.npy', mmap_mode='r'); self.off = np.load(aug / f'{kind}_offsets.npy')
    def __len__(self): return len(self.off) - 1 if self.ok else 0
    def __getitem__(self, i): return self.data[self.off[i]:self.off[i + 1]]


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, root, max_seconds, seed, holdout_only=False, limit_files=None, negative_prob=0., aug=(.5, .6, 0., 25.), mix=None, mix_active=4, stream_seconds=0., cold_start_prob=0.):
        self.root = Path(root); self.max_seconds = max_seconds; self.seed = seed; self.holdout_only = holdout_only
        self.negative_prob = negative_prob; self.aug = aug; self.mix = mix; self.mix_active = mix_active
        self.stream_seconds = stream_seconds; self.cold_start_prob = cold_start_prob
        self.files = sorted((self.root / 'index').glob('*.json'))[:limit_files]
        # Holdout PER stays comparable between runs: synthetic speech never enters it.
        if holdout_only: self.files = [f for f in self.files if not f.name.startswith('gemini__')]

    def _unit_rows(self, unit, cache, rng, nrng, music, noise, rirs, dry=False):
        import pyarrow as pa
        f, g, offset = unit
        if f not in cache:
            d = json.loads(f.read_text()); cache[f] = (d['summary'], {r[0]: r for r in d['rows'] if not r[3]})
            while len(cache) > 4: cache.pop(next(iter(cache)))
        s, rows = cache[f]
        if not rows: return
        pf = pq.ParquetFile(s['path']); cap = min(self.max_seconds, self.stream_seconds - .6) if dry and self.stream_seconds > 0 else self.max_seconds
        sub_off = offset
        for batch in pf.iter_batches(batch_size=512, row_groups=[g], columns=['audio']):
            col = batch.column(0).to_pylist(); del batch
            pa.default_memory_pool().release_unused()
            order = list(range(len(col))); rng.shuffle(order)
            for j in order:
                r = rows.get(sub_off + j); item = col[j]; col[j] = None
                if r is None: continue
                try: x, rate = sf.read(io.BytesIO(item['bytes']), dtype='float32', always_2d=True)
                except Exception: continue
                x = x.mean(1)
                if rate != RATE: gg = math.gcd(rate, RATE); x = resample_poly(x, RATE // gg, rate // gg).astype(np.float32)
                if len(x) < RATE * .8 or len(x) > RATE * cap: continue
                if dry:
                    yield x.astype(np.float32), np.array(r[2], np.int64), s['lang']
                    continue
                x = self.augment(x, rng, nrng, music, noise, rirs, self.aug)
                yield x.astype(np.float32), np.array(r[2], np.int64), s['lang']
                if (len(music) or len(noise)) and rng.random() < self.negative_prob:
                    far = self.aug[4] if len(self.aug) > 4 else 0.
                    yield self.negative(rng, nrng, music, noise, rirs, far), np.zeros(0, np.int64), 'neg'
            sub_off += len(col); del col

    @staticmethod
    def _room_speech(x, rng, nrng, rirs, far):
        u = rng.random()
        if len(rirs) and rng.random() < far: x = Stream.far_room(x, rng, rirs)
        elif u < .25 and len(rirs):
            h = rirs[rng.randrange(len(rirs))]; x = fftconvolve(x, h)[:len(x)].astype(np.float32)
        elif u < .4:
            t = np.arange(int(RATE * rng.uniform(.2, .6))) / RATE
            h = nrng.normal(size=len(t)) * .03 * np.exp(-t / rng.uniform(.05, .2)); h[0] = 1.
            x = fftconvolve(x, h)[:len(x)].astype(np.float32)
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-9))
        if rms < .025: x = x * (.06 / rms)
        elif rms > .28: x = x * (.18 / rms)
        return x.astype(np.float32)

    @staticmethod
    def _bed(W, rng, nrng, music, noise, rirs, music_share):
        use_synth = rng.random() < .15 or (not len(music) and not len(noise))
        if use_synth:
            w = nrng.normal(size=W).astype(np.float32); spec = np.fft.rfft(w); f = np.fft.rfftfreq(W, 1 / RATE)
            bg = np.fft.irfft(spec / (np.maximum(f, 40.) ** rng.uniform(0., 1.2)), W).astype(np.float32)
        else:
            pool = music if (rng.random() < music_share and len(music)) else (noise if len(noise) else music)
            bg = pool[rng.randrange(len(pool))].astype(np.float32) / 32767
            if len(bg) < W: bg = np.tile(bg, int(np.ceil(W / max(len(bg), 1))))
            o = rng.randrange(0, len(bg) - W + 1); bg = bg[o:o + W].copy()
        if rng.random() < .25 and len(rirs): bg = fftconvolve(bg, rirs[rng.randrange(len(rirs))])[:W].astype(np.float32)
        return bg

    def _pack_streams(self, raw_iter, rng, nrng, music, noise, rirs):
        """Pack 1-3 utterances (or 0 for negatives) at random offsets onto an uninterrupted W-sample noise/music/clean bed."""
        W = int(round(RATE * self.stream_seconds))
        bg_prob, music_share, snr_low, snr_high = self.aug[:4]
        far = self.aug[4] if len(self.aug) > 4 else 0.
        buf = None
        while True:
            if rng.random() < self.negative_prob:
                bg = self._bed(W, rng, nrng, music, noise, rirs, music_share)
                bg = bg / (float(np.sqrt(np.mean(bg ** 2))) + 1e-6) * rng.uniform(.015, .25)
                if rng.random() < far: bg = Stream.device_chain(bg, rng)
                peak = np.abs(bg).max()
                yield (bg / peak if peak > 1 else bg).astype(np.float32), np.zeros(0, np.int64), 'neg'
                continue
            first = buf if buf is not None else next(raw_iter, None); buf = None
            if first is None: return
            c0 = self._room_speech(first[0], rng, nrng, rirs, far); ref_rms = float(np.sqrt(np.mean(c0 ** 2) + 1e-9))
            chosen = [(c0, first[1], first[2])]; used = len(c0)
            k = rng.choices([1, 2, 3], [.45, .4, .15])[0]
            min_gap = int(RATE * .35)
            while len(chosen) < k:
                nxt = next(raw_iter, None)
                if nxt is None: break
                if used + len(nxt[0]) + min_gap * (len(chosen) + 1) <= W:
                    ci = self._room_speech(nxt[0], rng, nrng, rirs, far)
                    ci = ci * (ref_rms / float(np.sqrt(np.mean(ci ** 2) + 1e-9))) * 10 ** (rng.uniform(-3.5, 3.5) / 20)
                    chosen.append((ci.astype(np.float32), nxt[1], nxt[2])); used += len(ci)
                else:
                    buf = nxt; break
            m = len(chosen); inner = min_gap * (m - 1); slack = max(0, W - used - inner)
            is_cold = self.cold_start_prob > 0 and rng.random() < self.cold_start_prob
            if is_cold:
                lead = min(slack, int(RATE * rng.uniform(0., .25)))
                rem = slack - lead
                if m == 1:
                    gaps = [lead, rem]
                else:
                    weights = [rng.expovariate(1.) for _ in range(m)]; ws = sum(weights) + 1e-9
                    tail = [int(rem * w / ws) for w in weights]
                    tail[-1] += rem - sum(tail)
                    gaps = [lead] + tail
            else:
                weights = [rng.expovariate(1.) for _ in range(m + 1)]; ws = sum(weights) + 1e-9
                gaps = [int(slack * w / ws) for w in weights]
                gaps[-1] += slack - sum(gaps)
            for idx in range(1, m): gaps[idx] += min_gap
            speech = np.zeros(W, np.float32); pos = gaps[0]
            for idx, (ci, _, _) in enumerate(chosen):
                end = min(W, pos + len(ci)); speech[pos:end] += ci[:end - pos]
                pos = end + (gaps[idx + 1] if idx + 1 < len(gaps) else 0)
            ps = float(np.mean(np.concatenate([c for c, _, _ in chosen]) ** 2)) + 1e-9
            if rng.random() < bg_prob:
                bg = self._bed(W, rng, nrng, music, noise, rirs, music_share)
                snr = rng.uniform(snr_low, snr_high); pb = float(np.mean(bg ** 2)) + 1e-9
                y = speech + bg * math.sqrt(ps / (pb * 10 ** (snr / 10)))
            else:
                floor = nrng.normal(size=W).astype(np.float32) * math.sqrt(ps) * 10 ** (rng.uniform(-60, -35) / 20)
                y = speech + floor
            if rng.random() < far: y = Stream.device_chain(y, rng)
            y = y * 10 ** (rng.uniform(-18, 6) / 20)
            peak = np.abs(y).max()
            if peak > 1: y = y / peak
            tag = ('cold_' + chosen[0][2]) if is_cold else chosen[0][2]
            yield y.astype(np.float32), np.concatenate([t for _, t, _ in chosen]), tag

    def _mixed(self, wid, n):
        """Several row groups read at once, each refilled from a pool chosen by the target mix, so batches blend sources and languages."""
        rng = random.Random(self.seed * 1000 + wid); nrng = np.random.default_rng(self.seed * 1000 + wid)
        aug = self.root / 'aug'; music = Pool(aug, 'music'); noise = Pool(aug, 'noise')
        rirs = np.load(aug / 'rirs.npy', mmap_mode='r') if (aug / 'rirs.npy').exists() else []
        pools = {k: [] for k in self.mix}; cache = {}
        split = random.Random(self.seed)
        for f in self.files:
            d = json.loads(f.read_text()); s = d['summary']; key = ('gem_' + s['lang']) if s['name'].startswith('gemini') else (s['name'] if s['name'] in pools else s['lang'])
            if key not in pools: continue
            pf = pq.ParquetFile(s['path']); off = 0
            for g in range(pf.num_row_groups):
                pools[key].append((f, g, off)); off += pf.metadata.row_group(g).num_rows
        for k in list(pools):
            split.shuffle(pools[k]); share = pools[k][wid::n] or pools[k]
            pools[k] = list(share); rng.shuffle(pools[k])
        keys = [k for k in self.mix if pools[k]]; weights = [self.mix[k] for k in keys]; pos = {k: 0 for k in keys}
        if not keys: return
        dry = self.stream_seconds > 0

        def fresh():
            k = rng.choices(keys, weights)[0]
            if pos[k] >= len(pools[k]): rng.shuffle(pools[k]); pos[k] = 0
            unit = pools[k][pos[k]]; pos[k] += 1
            return self._unit_rows(unit, cache, rng, nrng, music, noise, rirs, dry=dry)

        def raw():
            active = [fresh() for _ in range(self.mix_active)]
            while True:
                i = rng.randrange(len(active)); item = next(active[i], None)
                if item is None: active[i] = fresh(); continue
                yield item

        if dry:
            yield from self._pack_streams(raw(), rng, nrng, music, noise, rirs)
        else:
            yield from raw()

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); wid = info.id if info else 0; n = info.num_workers if info else 1
        if self.mix and not self.holdout_only:
            yield from self._mixed(wid, n); return
        rng = random.Random(self.seed * 1000 + wid); nrng = np.random.default_rng(self.seed * 1000 + wid)
        aug = self.root / 'aug'
        music = Pool(aug, 'music'); noise = Pool(aug, 'noise')
        rirs = np.load(aug / 'rirs.npy') if (aug / 'rirs.npy').exists() else []
        units = []
        for f in self.files:
            d = json.loads(f.read_text())
            pf = pq.ParquetFile(d['summary']['path']); off = 0
            for g in range(pf.num_row_groups):
                units.append((f, g, off)); off += pf.metadata.row_group(g).num_rows
        units = units[wid::n]
        cache = {}
        epoch = 0
        while True:
            rng.shuffle(units)
            for f, g, offset in units:
                if f not in cache:
                    d = json.loads(f.read_text())
                    cache[f] = (d['summary'], {r[0]: r for r in d['rows'] if bool(r[3]) == self.holdout_only})
                    while len(cache) > 4: cache.pop(next(iter(cache)))
                s, rows = cache[f]
                if not rows: continue
                pf = pq.ParquetFile(s['path']); sub_off = offset
                for batch in pf.iter_batches(batch_size=512, row_groups=[g], columns=['audio']):
                    col = batch.column(0).to_pylist(); del batch
                    order = list(range(len(col))); rng.shuffle(order)
                    for j in order:
                        r = rows.get(sub_off + j)
                        if r is None: continue
                        try:
                            x, rate = sf.read(io.BytesIO(col[j]['bytes']), dtype='float32', always_2d=True)
                        except Exception:
                            continue
                        x = x.mean(1)
                        if rate != RATE: gg = math.gcd(rate, RATE); x = resample_poly(x, RATE // gg, rate // gg).astype(np.float32)
                        if len(x) < RATE * .8 or len(x) > RATE * self.max_seconds: continue
                        if not self.holdout_only: x = self.augment(x, rng, nrng, music, noise, rirs, self.aug)
                        yield x.astype(np.float32), np.array(r[2], np.int64), s['lang']
                        if not self.holdout_only and (len(music) or len(noise)) and rng.random() < self.negative_prob:
                            far = self.aug[4] if len(self.aug) > 4 else 0.
                            yield self.negative(rng, nrng, music, noise, rirs, far), np.zeros(0, np.int64), 'neg'
                    sub_off += len(col); del col
            if not units: return
            epoch += 1
            if self.holdout_only: return

    @staticmethod
    def far_room(x, rng, rirs):
        """Speaker far from the microphone: a real room response with its direct sound attenuated, plus air absorption."""
        h = np.array(rirs[rng.randrange(len(rirs))], np.float32)
        p = int(np.abs(h).argmax()); w = int(RATE * .0025)
        h[max(0, p - w):p + w] *= 10 ** (-rng.uniform(3, 15) / 20)
        y = fftconvolve(x, h)[:len(x)]
        spec = np.fft.rfft(y); f = np.fft.rfftfreq(len(y), 1 / RATE)
        spec *= 1 / np.sqrt(1 + (f / rng.uniform(2500, 7500)) ** 4)
        y = np.fft.irfft(spec, len(y)); gain = np.sqrt((np.mean(x ** 2) + 1e-9) / (np.mean(y ** 2) + 1e-9))
        return (y * gain).astype(np.float32)

    @staticmethod
    def device_chain(x, rng):
        """Consumer capture path: random microphone response, band limits, limiter or level control, and coarse companding."""
        n = len(x)
        if n < 64: return x
        spec = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / RATE)
        points = np.geomspace(60, RATE / 2, 8); response = np.array([rng.uniform(-6, 6) for _ in points])
        gain = 10 ** (np.interp(np.log(np.maximum(f, 1.)), np.log(points), response) / 20)
        gain *= 1 / np.sqrt(1 + (rng.uniform(60, 300) / np.maximum(f, 1.)) ** 4)
        if rng.random() < .5: gain *= 1 / np.sqrt(1 + (f / rng.uniform(3400, 7800)) ** 8)
        y = np.fft.irfft(spec * gain, n)
        u = rng.random()
        if u < .2:
            drive = 10 ** (rng.uniform(3, 15) / 20); peak = np.abs(y).max() + 1e-9
            y = np.tanh(y / peak * drive) * peak / np.tanh(drive)
        elif u < .3:
            peak = np.abs(y).max() + 1e-9; z = y / peak; mu = 255.
            c = np.round((np.sign(z) * np.log1p(mu * np.abs(z)) / np.log1p(mu)) * 127) / 127
            y = np.sign(c) * (np.expm1(np.abs(c) * np.log1p(mu)) / mu) * peak
        return y.astype(np.float32)

    @staticmethod
    def negative(rng, nrng, music, noise, rirs, far=0.):
        n = int(RATE * rng.uniform(2, 12))
        pool = music if (rng.random() < .65 and len(music)) else noise
        bg = pool[rng.randrange(len(pool))].astype(np.float32) / 32767
        if len(bg) < n: bg = np.tile(bg, int(np.ceil(n / max(len(bg), 1))))
        o = rng.randrange(0, len(bg) - n + 1); x = bg[o:o + n].copy()
        if rng.random() < .3 and len(rirs): x = fftconvolve(x, rirs[rng.randrange(len(rirs))])[:n].astype(np.float32)
        if rng.random() < far: x = Stream.device_chain(x, rng)
        x = x * 10 ** (rng.uniform(-25, 3) / 20)
        peak = np.abs(x).max()
        return (x / peak if peak > 1 else x).astype(np.float32)

    @staticmethod
    def augment(x, rng, nrng, music, noise, rirs, aug=(.5, .6, 0., 25.)):
        bg_prob, music_share, snr_low, snr_high = aug[:4]
        far = aug[4] if len(aug) > 4 else 0.
        x = np.pad(x, (int(RATE * rng.uniform(0, .3)), int(RATE * rng.uniform(0, .3))))
        u = rng.random()
        if len(rirs) and rng.random() < far:
            x = Stream.far_room(x, rng, rirs)
        elif u < .25 and len(rirs):
            h = rirs[rng.randrange(len(rirs))]; x = fftconvolve(x, h)[:len(x)].astype(np.float32)
        elif u < .4:
            t = np.arange(int(RATE * rng.uniform(.2, .6))) / RATE
            h = nrng.normal(size=len(t)) * .03 * np.exp(-t / rng.uniform(.05, .2)); h[0] = 1.
            x = fftconvolve(x, h)[:len(x)].astype(np.float32)
        if rng.random() < bg_prob:
            pool = music if (rng.random() < music_share and len(music)) else noise
            if len(pool):
                bg = pool[rng.randrange(len(pool))].astype(np.float32) / 32767
                if len(bg) < len(x): bg = np.tile(bg, int(np.ceil(len(x) / max(len(bg), 1))))
                o = rng.randrange(0, len(bg) - len(x) + 1); bg = bg[o:o + len(x)]
                snr = rng.uniform(snr_low, snr_high); ps = np.mean(x ** 2) + 1e-9; pb = np.mean(bg ** 2) + 1e-9
                x = x + bg * np.sqrt(ps / (pb * 10 ** (snr / 10)))
        if rng.random() < far: x = Stream.device_chain(x, rng)
        x = x * 10 ** (rng.uniform(-20, 6) / 20)
        peak = np.abs(x).max()
        if peak > 1: x = x / peak
        return x


def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'): return round(int(line.split()[1]) / 1e6, 1)
    return None


def parse_mix(text):
    if not text: return None
    pairs = [p.split('=') for p in text.split(',')]; total = sum(float(v) for _, v in pairs)
    return {k: float(v) / total for k, v in pairs}


PAD_NOISE = [0.]


def collate(batch):
    n = max(len(b[0]) for b in batch)
    audio = torch.zeros(len(batch), n)
    for i, b in enumerate(batch):
        audio[i, :len(b[0])] = torch.from_numpy(b[0])
        if PAD_NOISE[0] and len(b[0]) < n and random.random() < PAD_NOISE[0]:
            audio[i, len(b[0]):] = torch.randn(n - len(b[0])) * 10 ** (random.uniform(-85, -45) / 20)
    lengths = torch.tensor([len(b[0]) for b in batch])
    targets = torch.cat([torch.from_numpy(b[1]) for b in batch]); tl = torch.tensor([len(b[1]) for b in batch])
    return audio, lengths, targets, tl, [b[2] for b in batch]


def frames(samples):
    n = torch.clamp((samples - WINDOW) // HOP + 1, min=0)
    return (n + 1) // 2


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def greedy_per(lp, lengths, targets, tl):
    err = tot = 0; off = 0; best = lp.argmax(-1).cpu()
    for i in range(len(lengths)):
        seq = best[i, :lengths[i]].tolist(); hyp = [s for k, s in enumerate(seq) if s and (k == 0 or s != seq[k - 1])]
        ref = targets[off:off + tl[i]].tolist(); off += tl[i]
        err += levenshtein(hyp, ref); tot += len(ref)
    return err, tot


def spec_augment_feats(feats, fl, prob, rng):
    if prob <= 0.: return feats
    out = feats.clone()
    B, T, F_bins = out.shape
    for i in range(B):
        if rng.random() >= prob: continue
        ti = int(fl[i].item())
        if ti < 10: continue
        fill = float(out[i, :ti].mean().item())
        for _ in range(1 if rng.random() < .5 else 2):
            fw = rng.randint(1, min(8, F_bins // 4))
            f0 = rng.randint(0, F_bins - fw)
            out[i, :ti, f0:f0 + fw] = fill
        for _ in range(1 if rng.random() < .5 else 2):
            tw = rng.randint(1, min(5, max(1, ti // 8)))
            t0 = rng.randint(0, ti - tw)
            out[i, t0:t0 + tw, :] = fill
    return out


def make_lr_lambda(warm, restart, steps, cosine_restart=False):
    if cosine_restart and restart > 0:
        span = max(1, steps - restart)
        rw = min(warm, max(1, span // 5))
        return lambda s: 0.0 if s < restart else min(1., (s - restart + 1) / rw) * max(.05, .5 * (1. + math.cos(math.pi * min(1., (s - restart) / span))))
    return lambda s: min(1., (s + 1) / warm) * min(1., (s - restart + 1) / 300 if s >= restart else 1.) * max(.02, .5 * (1 + math.cos(math.pi * min(1., s / steps))))


def train(a):
    torch.backends.cudnn.benchmark = True
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(a.vocab).read_text()); classes = len(vocab) + 1
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    front = LogMel(64).to(dev)
    sizes = [int(h) for h in a.hidden.split(',')]
    models, opts, scheds = {}, {}, {}
    for h in sizes:
        torch.manual_seed(a.seed + h)
        m = AcousticModel(classes, h, 3, a.dropout, 64).to(dev); models[h] = m
        opts[h] = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=.01, betas=(.9, .999))
        scheds[h] = torch.optim.lr_scheduler.LambdaLR(opts[h], make_lr_lambda(a.warmup, a.init_step, a.steps, getattr(a, 'cosine_restart', False)))
    start_step = 0
    resume = out / 'resume.pt'
    if a.init_step and not resume.exists():
        for h in sizes:
            ck = torch.load(Path(a.init_dir) / f'h{h}-step{a.init_step:06d}.pt', map_location=dev, weights_only=False)
            models[h].load_state_dict(ck['model'])
            for _ in range(a.init_step): scheds[h].step()
        start_step = a.init_step
        print(json.dumps({'initialized_from_step': a.init_step, 'optimizer_state': 'fresh (not recoverable)', 'cosine_restart': getattr(a, 'cosine_restart', False)}), flush=True)
    if resume.exists():
        st = torch.load(resume, map_location=dev, weights_only=False)
        for h in sizes:
            models[h].load_state_dict(st['models'][str(h)]); opts[h].load_state_dict(st['opts'][str(h)]); scheds[h].load_state_dict(st['scheds'][str(h)])
        start_step = st['step']
    PAD_NOISE[0] = a.pad_noise
    cold_prob = getattr(a, 'cold_start_prob', 0.)
    spec_aug = getattr(a, 'spec_aug', 0.)
    data = torch.utils.data.DataLoader(Stream(root, a.max_seconds, a.seed + start_step, limit_files=a.limit_files, negative_prob=a.negative_prob,
                                       aug=(a.aug_bg_prob, a.aug_music_share, a.aug_snr_low, a.aug_snr_high, a.aug_far),
                                       mix=parse_mix(a.mix), mix_active=a.mix_active, stream_seconds=a.stream_seconds, cold_start_prob=cold_prob), batch_size=a.batch,
                                       num_workers=a.workers, collate_fn=collate, prefetch_factor=4 if a.workers else None,
                                       persistent_workers=bool(a.workers))
    ctc = torch.nn.CTCLoss(zero_infinity=True); skipped = {h: 0 for h in sizes}
    log = (out / 'train.jsonl').open('a'); t0 = time.time(); seen = 0.; it = iter(data); step = start_step
    carry = {h: None for h in sizes}; crng = torch.Generator().manual_seed(a.seed + start_step); srng = random.Random(a.seed + start_step)
    while step < a.steps:
        audio, lengths, targets, tl, langs = next(it)
        audio = audio.to(dev, non_blocking=True)
        fl = frames(lengths).to(dev); seen += lengths.sum().item() / RATE
        with torch.no_grad():
            feats = front(audio)[:, 0::2]
            if spec_aug > 0.:
                feats = spec_augment_feats(feats, fl, spec_aug, srng)
        cold_mask = None
        if any(isinstance(l, str) and l.startswith('cold_') for l in langs):
            cold_mask = torch.tensor([0. if (isinstance(l, str) and l.startswith('cold_')) else 1. for l in langs], device=dev, dtype=feats.dtype)
        losses = {}
        for h, m in models.items():
            m.train()
            state = None
            if a.carry_prob > 0 and carry[h] is not None and carry[h]['gru'].shape[1] == feats.shape[0]:
                # Stateful long-stream training: each batch slot keeps (detached) the state left by its previous utterance and the
                # padding after it, unless marked as an explicit cold-start utterance.
                keep = (torch.rand(feats.shape[0], generator=crng) < a.carry_prob).to(dev, feats.dtype)
                if cold_mask is not None: keep = keep * cold_mask
                state = {'gru': carry[h]['gru'] * keep[None, :, None], 'conv': [c * keep[:, None, None] for c in carry[h]['conv']]}
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda' and a.amp):
                lp, new_state = m(feats, state)
            if a.carry_prob > 0: carry[h] = {'gru': new_state['gru'].detach().float(), 'conv': [c.detach().float() for c in new_state['conv']]}
            loss = ctc(lp.float().transpose(0, 1), targets.to(dev), fl, tl.to(dev))
            opts[h].zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); gn = torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
                if torch.isfinite(gn): opts[h].step()
                else: skipped[h] += 1
            else: skipped[h] += 1
            scheds[h].step(); losses[h] = loss.item()
        step += 1
        if step % 50 == 0:
            row = dict(step=step, hours_seen=seen / 3600, seconds=time.time() - t0, lr=scheds[sizes[0]].get_last_lr()[0],
                       loss={str(h): v for h, v in losses.items()}, skipped={str(h): v for h, v in skipped.items()}, audio_hours_per_wall_hour=seen / max(time.time() - t0, 1e-9), mem_avail_gb=mem_available_gb())
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({'step': step, 'models': {str(h): m.state_dict() for h, m in models.items()},
                        'opts': {str(h): o.state_dict() for h, o in opts.items()},
                        'scheds': {str(h): s.state_dict() for h, s in scheds.items()}}, out / 'resume.tmp')
            os.replace(out / 'resume.tmp', resume)
            per = evaluate_holdout(models, root, front, dev, a) if step % a.eval_every == 0 or step == a.steps else None
            for h, m in models.items():
                save_checkpoint(m, vocab, h, a, step, seen, per and per[str(h)], out / f'h{h}-step{step:06d}.pt')
            if per: log.write(json.dumps(dict(step=step, holdout_per=per)) + '\n'); log.flush(); print(json.dumps(dict(step=step, holdout_per=per)), flush=True)
    (out / 'DONE.json').write_text(json.dumps(dict(step=step, hours_seen=seen / 3600)))


@torch.no_grad()
def evaluate_holdout(models, root, front, dev, a):
    data = torch.utils.data.DataLoader(Stream(root, a.max_seconds, 0, holdout_only=True, limit_files=a.limit_files), batch_size=32,
                                       num_workers=0, collate_fn=collate)
    res = {str(h): {'de': [0, 0], 'en': [0, 0]} for h in models}; n = 0
    for audio, lengths, targets, tl, langs in data:
        audio = audio.to(dev); feats = front(audio)[:, 0::2]; fl = frames(lengths)
        for h, m in models.items():
            m.eval(); lp, _ = m(feats)
            off = 0
            for i, lang in enumerate(langs):
                e, t = greedy_per(lp[i:i + 1], fl[i:i + 1], targets[off:off + tl[i]], tl[i:i + 1]); off += tl[i]
                res[str(h)][lang][0] += e; res[str(h)][lang][1] += t
        n += len(langs)
        if n >= a.holdout_utterances: break
    return {h: {l: (v[0] / max(v[1], 1)) for l, v in r.items()} | {'utterances': n} for h, r in res.items()}


def save_checkpoint(m, vocab, h, a, step, seen, per, path):
    sd = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    torch.save({'model': sd, 'vocab': vocab, 'hidden': h, 'bins': 64, 'causal_layers': 3, 'dropout': a.dropout, 'step': step,
                'epoch': 0, 'best': None, 'feature_version': FEATURE_VERSION, 'text_version': TEXT_VERSION,
                'pronunciation_label_versions': ['de-lexicon-v1', 'espeak-v1'], 'initialization': 'random seed %d; no external weights' % (a.seed + h),
                'parameters': sum(v.numel() for v in m.parameters()), 'args': vars(a), 'training_audio_hours_seen': seen / 3600,
                'holdout_phone_error_rate': per, 'study': 'r211-scaled-ctc', 'posterior_temperature': 1.0}, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'train'])
    p.add_argument('--root', default='/content/r211'); p.add_argument('--out', default='/content/r211/out')
    p.add_argument('--exclusions', default='reports/round212/EXCLUSIONS.json'); p.add_argument('--cv-de-shards', type=int, default=48); p.add_argument('--cv-en-shards', type=int, default=0); p.add_argument('--vocab', default='reports/round211/vocab.json')
    p.add_argument('--music-dir', default='data/round14-music/wav'); p.add_argument('--vox-de-shards', type=int, default=6)
    p.add_argument('--vox-en-shards', type=int, default=6); p.add_argument('--procs', type=int, default=os.cpu_count())
    p.add_argument('--only', default=None, help='comma list of shard names for smoke tests')
    p.add_argument('--hidden', default='384,512'); p.add_argument('--steps', type=int, default=60000); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-3); p.add_argument('--warmup', type=int, default=1500); p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--max-seconds', type=float, default=17.); p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--seed', type=int, default=2110); p.add_argument('--save-every', type=int, default=2000); p.add_argument('--eval-every', type=int, default=4000)
    p.add_argument('--holdout-utterances', type=int, default=1500); p.add_argument('--limit-files', type=int, default=None)
    p.add_argument('--skip-aug', action='store_true')
    p.add_argument('--amp', action='store_true'); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--aug-far', type=float, default=0.)
    p.add_argument('--mix', default='', help='target clip shares per pool, e.g. de=.4,en=.4,gem_de=.09,gem_en=.11; empty keeps file-proportional sampling')
    p.add_argument('--mix-active', type=int, default=4); p.add_argument('--gemini-dir', default='')
    p.add_argument('--negative-prob', type=float, default=.12); p.add_argument('--aug-bg-prob', type=float, default=.5)
    p.add_argument('--aug-music-share', type=float, default=.6); p.add_argument('--aug-snr-low', type=float, default=0.); p.add_argument('--aug-snr-high', type=float, default=25.)
    p.add_argument('--carry-prob', type=float, default=0., help='per slot chance to start from the state its previous utterance left')
    p.add_argument('--pad-noise', type=float, default=0., help='share of padded slots whose padding is faint noise instead of zeros')
    p.add_argument('--stream-seconds', type=float, default=0., help='fixed stream window in seconds with 1-3 utterances over a continuous noise/music/clean bed')
    p.add_argument('--cold-start-prob', type=float, default=0., help='share of stream windows starting speech within 0-250ms from fresh zero state')
    p.add_argument('--spec-aug', type=float, default=0., help='per-sample SpecAugment probability on 50Hz log-mel features')
    p.add_argument('--cosine-restart', action='store_true', help='run a fresh warmup + cosine decay across [init_step, steps]')
    p.add_argument('--init-dir', default='/content/r211/init'); p.add_argument('--init-step', type=int, default=0)
    a = p.parse_args()
    if a.stage == 'prepare':
        vocab = json.loads(Path(a.vocab).read_text())
        specs = source_specs(a.vox_de_shards, a.vox_en_shards, a.cv_de_shards, a.cv_en_shards)
        if a.gemini_dir:
            specs += [dict(name=f'gemini_{lang}', repo=None, rev=None, file=f'gemini/{lang}/{f.name}', lang=lang, text='text', speaker='speaker', origin='origin',
                           exclusion_source='gemini', local_path=str(f)) for lang in ('de', 'en') for f in sorted((Path(a.gemini_dir) / lang).glob('*.parquet'))]
        if a.only: specs = [s for s in specs if s['file'] in a.only.split(',')]
        Path(a.root).mkdir(parents=True, exist_ok=True)
        if not a.skip_aug: print(json.dumps({'augmentation': prepare_augmentation(a.root, a.exclusions, a.music_dir)}), flush=True)
        summaries = []
        with ProcessPoolExecutor(min(a.procs, 6)) as pool:
            for s in pool.map(index_shard, [(s, a.root, a.exclusions, vocab) for s in specs]):
                summaries.append(s); print(json.dumps({'indexed': s['file'], 'kept': s['kept'], 'rows': s['rows']}), flush=True)
        (Path(a.root) / 'PREPARED.json').write_text(json.dumps(summaries, indent=1))
    else:
        train(a)


if __name__ == '__main__':
    main()
