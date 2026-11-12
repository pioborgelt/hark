"""R218 training & holdout evaluation for BiPhonemeVerifier (open-vocabulary Stage-2 verifier).

Reuses the exact exclusion-checked shard index and on-the-fly RIR/MUSAN/music augmentation from
`hark.r211_remote`, training a bidirectional phoneme CTC + cross-attention span verifier on general
DE/EN speech and music/noise negatives without any hardcoded wake words or benchmark audio.
"""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
from numba import njit
import torch
import torch.nn.functional as F

from hark.audio import LogMel, RATE, WINDOW, HOP
from hark.r211_remote import (
    FEATURE_VERSION, PAD_NOISE, TEXT_VERSION, Stream, collate, frames, greedy_per,
    make_lr_lambda, mem_available_gb, parse_mix, spec_augment_feats
)
from hark.verifier import VERIFIER_KIND, BiPhonemeVerifier

CONSONANT_CHARS = tuple('bcdfhjklmnpstvwxzðŋɕɡʃʒθ')


@njit(cache=False, nogil=True)
def viterbi_phoneme_spans(lp, targets):
    T, V = lp.shape
    M = len(targets)
    p_start = np.zeros(M, np.int32)
    p_end = np.zeros(M, np.int32)
    if M == 0:
        return False, p_start, p_end
    repeats = 0
    for i in range(1, M):
        if targets[i] == targets[i - 1]:
            repeats += 1
    if T < M + repeats:
        return False, p_start, p_end
    S = 2 * M + 1
    ext = np.zeros(S, np.int64)
    for i in range(M):
        ext[2 * i + 1] = targets[i]
    neg_inf = np.float32(-1e30)
    dp = np.full((T, S), neg_inf, np.float32)
    ptr = np.zeros((T, S), np.int16)
    dp[0, 0] = lp[0, 0]
    if S > 1:
        dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best = dp[t - 1, s]
            prev = s
            if s >= 1 and dp[t - 1, s - 1] > best:
                best = dp[t - 1, s - 1]
                prev = s - 1
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best = dp[t - 1, s - 2]
                prev = s - 2
            if best > -1e29:
                dp[t, s] = best + lp[t, ext[s]]
            ptr[t, s] = prev
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= -1e28:
        return False, p_start, p_end
    p_start[:] = T
    p_end[:] = 0
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            m = s // 2
            if t < p_start[m]:
                p_start[m] = t
            if t + 1 > p_end[m]:
                p_end[m] = t + 1
        s = ptr[t, s]
    for m in range(M):
        if p_end[m] <= p_start[m]:
            return False, p_start, p_end
    return True, p_start, p_end


def build_consonant_pools(vocab):
    cons = sorted(vocab[c] for c in CONSONANT_CHARS if c in vocab)
    inserts = [vocab[c] for c in ('t', 'k', 'p', 's', 'f', 'm', 'n', 'l') if c in vocab]
    # Map each consonant to non-confusable substitutes (never swapping t<->d or rhotics/vowels)
    close_pairs = {
        ('p', 'b'), ('t', 'd'), ('k', 'ɡ'), ('f', 'v'), ('s', 'z'), ('ʃ', 'ʒ'), ('θ', 'ð'), ('m', 'n'), ('n', 'ŋ')
    }
    inv = {v: k for k, v in vocab.items()}
    subs = {}
    for cid in cons:
        ch = inv[cid]
        alts = [x for x in cons if x != cid and (ch, inv[x]) not in close_pairs and (inv[x], ch) not in close_pairs]
        subs[cid] = alts or [x for x in cons if x != cid]
    return set(cons), subs, inserts


def corrupt_consonants(seq, cons_set, subs, inserts, rng):
    out = list(seq)
    n_edits = 1 if rng.random() < 0.65 else 2
    for _ in range(n_edits):
        c_pos = [i for i, t in enumerate(out) if t in cons_set]
        op = rng.random()
        if c_pos and op < 0.60:
            idx = rng.choice(c_pos)
            alts = subs.get(out[idx], inserts)
            if alts:
                out[idx] = rng.choice(alts)
        elif op < 0.85 and len(out) < 16 and inserts:
            idx = rng.randint(0, len(out))
            out.insert(idx, rng.choice(inserts))
        elif len(c_pos) >= 2 and len(out) >= 6:
            idx = rng.choice(c_pos)
            out.pop(idx)
        elif inserts:
            idx = rng.randint(0, len(out))
            out.insert(idx, rng.choice(inserts))
    if out == list(seq) and inserts:
        out.insert(rng.randint(0, len(out)), rng.choice(inserts))
    return out


def corrupt_edge(seq, cons_set, subs, inserts, all_ids, rng):
    out = list(seq)
    k = 2 if len(out) >= 6 else 1
    if rng.random() < 0.5:
        for i in range(k):
            out[i] = rng.choice(subs.get(out[i], inserts or all_ids))
            if out[i] == seq[i]:
                out[i] = rng.choice(all_ids)
    else:
        for i in range(1, k + 1):
            out[-i] = rng.choice(subs.get(out[-i], inserts or all_ids))
            if out[-i] == seq[-i]:
                out[-i] = rng.choice(all_ids)
    return out


def corrupt_minimal_rhyme(seq, cons_set, subs, rng, flap_pair=None):
    """Single-consonant minimal-pair rhyme confuser preserving 100% of vowels while respecting close_pairs."""
    out = list(seq)
    c_pos = [i for i, t in enumerate(out) if t in cons_set]
    if not c_pos:
        return None
    idx = rng.choice(c_pos)
    cur = out[idx]
    forbid = {cur}
    if flap_pair and 0 < idx < len(out) - 1 and cur in flap_pair:
        forbid |= flap_pair
    cands = [c for c in subs.get(cur, cons_set) if c not in forbid]
    if not cands:
        return None
    out[idx] = rng.choice(cands)
    return out


def is_subseq(sub, full):
    n, m = len(sub), len(full)
    if n > m:
        return False
    for i in range(m - n + 1):
        if full[i:i + n] == sub:
            return True
    return False


def build_verification_pairs(feats_cpu, fl_cpu, targets_cpu, tl_cpu, langs, bi_lp_cpu, cons_info, all_ids, rng, hard_rhyme=False, flap_pair=None):
    cons_set, subs, inserts = cons_info
    B = feats_cpu.shape[0]
    extracted = []
    uniq_wins = []
    neg_utts = []
    off = 0
    for b in range(B):
        T_b = int(fl_cpu[b].item())
        M_b = int(tl_cpu[b].item())
        utt_toks = targets_cpu[off:off + M_b].tolist()
        off += M_b
        if langs[b] == 'neg' or M_b == 0:
            if T_b >= 40:
                neg_utts.append((feats_cpu[b, :T_b], T_b))
            continue
        if M_b < 6 or T_b < 24:
            continue
        ok, ps, pe = viterbi_phoneme_spans(bi_lp_cpu[b, :T_b], np.asarray(utt_toks, dtype=np.int64))
        if not ok:
            continue
        n_spans = 2 if M_b >= 16 and rng.random() < 0.5 else 1
        for _ in range(n_spans):
            L = rng.randint(5, min(12, M_b))
            a = rng.randint(0, M_b - L)
            s0_raw = int(ps[a])
            s1_raw = int(pe[a + L - 1])
            span_len = s1_raw - s0_raw
            if span_len < max(5, L - 1) or span_len > 115:
                continue
            j0 = (-3, 3)
            j1 = (-2, 4)
            s0 = max(0, min(T_b - 4, s0_raw + rng.randint(*j0)))
            s1 = max(s0 + 4, min(T_b, s1_raw + rng.randint(*j1)))
            left_ctx = rng.randint(12, 24)
            right_ctx = rng.randint(8, 18)
            w0 = max(0, s0 - left_ctx)
            w1 = min(T_b, s1 + right_ctx)
            if w1 - w0 < 16 or w1 - w0 > 160:
                continue
            u_id = len(uniq_wins)
            uniq_wins.append(feats_cpu[b, w0:w1])
            extracted.append({
                'u_id': u_id,
                'bounds': (s0 - w0, s1 - w0),
                'toks': utt_toks[a:a + L],
                'after_toks': utt_toks[a + L:a + L + 4],
                'before_toks': utt_toks[max(0, a - 4):a],
                'utt_toks': utt_toks,
                'tok_set': set(utt_toks[a:a + L]),
            })

    if not extracted:
        return None

    items = []  # (u_id, (rs0, rs1), tok_list, label)
    for idx, ex in enumerate(extracted):
        u_id, bnd, toks = ex['u_id'], ex['bounds'], ex['toks']
        # 1. Positive
        items.append((u_id, bnd, toks, 1.0))
        # 2. Hard Negative A: consonant skeleton corruption on exact same audio
        items.append((u_id, bnd, corrupt_consonants(toks, cons_set, subs, inserts, rng), 0.0))
        # 2b. Hard Negative F: minimal-pair single-consonant rhyme confuser on exact same audio
        if hard_rhyme:
            rhyme_neg = corrupt_minimal_rhyme(toks, cons_set, subs, rng, flap_pair=flap_pair)
            if rhyme_neg is not None and rhyme_neg != toks:
                items.append((u_id, bnd, rhyme_neg, 0.0))
        # 3. Hard Negative B: prefix/suffix partial match on exact same audio
        items.append((u_id, bnd, corrupt_edge(toks, cons_set, subs, inserts, all_ids, rng), 0.0))
        # 4. Hard Negative C: prefix/suffix audio truncation (audio has toks, query demands extra missing tail/head)
        if len(toks) <= 10 and rng.random() < 0.65:
            if rng.random() < 0.6:
                forbid = set(ex['after_toks'])
                cands = [c for c in cons_set if c not in forbid]
                if len(cands) >= 2:
                    items.append((u_id, bnd, list(toks) + [rng.choice(cands), rng.choice(cands)], 0.0))
            else:
                forbid = set(ex['before_toks'])
                cands = [c for c in cons_set if c not in forbid]
                if len(cands) >= 2:
                    items.append((u_id, bnd, [rng.choice(cands), rng.choice(cands)] + list(toks), 0.0))
        # 5. Hard Negative D: acoustically closest distractor speech window in batch
        best_j = -1
        best_sim = -1.0
        tset = ex['tok_set']
        slen = bnd[1] - bnd[0]
        for j, other in enumerate(extracted):
            if j == idx or is_subseq(toks, other['utt_toks']):
                continue
            ov = len(tset & other['tok_set']) / max(1, len(tset | other['tok_set']))
            olen = other['bounds'][1] - other['bounds'][0]
            sim = ov - 0.008 * abs(slen - olen)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0:
            other = extracted[best_j]
            items.append((other['u_id'], other['bounds'], toks, 0.0))

    # 6. Hard Negative E: pure music / singing / noise peaks
    for n_feat, T_n in neg_utts:
        for _ in range(min(3, len(extracted))):
            ref = rng.choice(extracted)
            slen = ref['bounds'][1] - ref['bounds'][0]
            left_ctx = rng.randint(12, 22)
            right_ctx = rng.randint(8, 16)
            need = min(T_n, left_ctx + slen + right_ctx)
            if need < 16:
                continue
            cands = [rng.randint(0, T_n - need) for _ in range(4)]
            w0 = max(cands, key=lambda c: float(n_feat[c:c + need].mean().item()))
            rs0 = min(need - 4, left_ctx)
            rs1 = min(need, rs0 + slen)
            u_id = len(uniq_wins)
            uniq_wins.append(n_feat[w0:w0 + need])
            items.append((u_id, (rs0, max(rs0 + 4, rs1)), ref['toks'], 0.0))

    if len(items) > 256:
        pos = [x for x in items if x[3] > 0.5]
        neg = [x for x in items if x[3] < 0.5]
        rng.shuffle(neg)
        items = pos + neg[:max(len(pos), 256 - len(pos))]

    # Compact uniq_wins to only referenced windows
    used_old = sorted({x[0] for x in items})
    remap = {old: new for new, old in enumerate(used_old)}
    comp_wins = [uniq_wins[old] for old in used_old]

    U = len(comp_wins)
    N = len(items)
    W_max = max(w.shape[0] for w in comp_wins)
    L_max = max(len(x[2]) for x in items)
    win_feats = torch.zeros((U, W_max, feats_cpu.shape[2]), dtype=torch.float32)
    win_lens = torch.zeros(U, dtype=torch.long)
    for u, w in enumerate(comp_wins):
        wi = w.shape[0]
        win_feats[u, :wi] = w
        win_lens[u] = wi

    win_idx = torch.zeros(N, dtype=torch.long)
    span_bounds = torch.zeros((N, 2), dtype=torch.long)
    q_tokens = torch.zeros((N, L_max), dtype=torch.long)
    q_lens = torch.zeros(N, dtype=torch.long)
    labels = torch.zeros(N, dtype=torch.float32)
    for i, (old_u, (rs0, rs1), toks, y) in enumerate(items):
        li = len(toks)
        win_idx[i] = remap[old_u]
        span_bounds[i, 0] = rs0
        span_bounds[i, 1] = rs1
        q_tokens[i, :li] = torch.tensor(toks, dtype=torch.long)
        q_lens[i] = li
        labels[i] = y
    return win_feats, win_lens, win_idx, span_bounds, q_tokens, q_lens, labels


def roc_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {'auc': 0.5, 'tpr_fpr1e2': 0.0, 'tpr_fpr2e3': 0.0, 'pos': int(len(pos)), 'neg': int(len(neg))}
    neg_sorted = np.sort(neg)
    ranks = np.searchsorted(neg_sorted, pos, side='right') + 0.5 * (
        np.searchsorted(neg_sorted, pos, side='right') - np.searchsorted(neg_sorted, pos, side='left')
    )
    auc = float(np.mean(ranks) / len(neg))
    th_1e2 = float(neg_sorted[min(len(neg) - 1, max(0, int(math.floor(0.99 * len(neg)))))])
    th_2e3 = float(neg_sorted[min(len(neg) - 1, max(0, int(math.floor(0.998 * len(neg)))))])
    return {
        'auc': round(auc, 5),
        'tpr_fpr1e2': round(float(np.mean(pos > th_1e2)), 4),
        'tpr_fpr2e3': round(float(np.mean(pos > th_2e3)), 4),
        'pos': int(len(pos)),
        'neg': int(len(neg)),
    }


@torch.no_grad()
def evaluate_holdout(model, root, front, dev, cons_info, all_ids, a):
    model.eval()
    data = torch.utils.data.DataLoader(
        Stream(root, min(a.max_seconds, 12.0), 0, holdout_only=True, limit_files=a.limit_files, negative_prob=0.15,
               aug=(0.6, 0.4, 0.0, 20.0, 0.25)),
        batch_size=32, num_workers=0, collate_fn=collate
    )
    res_per = {'de': [0, 0], 'en': [0, 0]}
    all_scores, all_labels = [], []
    rng = random.Random(2180)
    n_utt = 0
    use_amp = (dev == 'cuda' and a.amp)
    for audio, lengths, targets, tl, langs in data:
        audio_dev = audio.to(dev)
        feats = front(audio_dev)[:, 0::2]
        fl = frames(lengths).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            _, bi_lp = model.encode_audio(feats, fl)
        off = 0
        for i, lang in enumerate(langs):
            li = int(tl[i].item())
            if lang in res_per and li > 0:
                e, t = greedy_per(bi_lp[i:i + 1].float(), fl[i:i + 1], targets[off:off + li], tl[i:i + 1])
                res_per[lang][0] += e
                res_per[lang][1] += t
                n_utt += 1
            off += li
        vb = build_verification_pairs(
            feats.cpu(), fl.cpu(), targets.cpu(), tl.cpu(), langs, bi_lp.float().cpu().numpy(), cons_info, all_ids, rng
        )
        if vb is not None:
            wf, wl, wi, sb, qt, ql, lb = vb
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model.verify_windows(wf.to(dev), wl.to(dev), sb.to(dev), qt.to(dev), ql.to(dev), win_idx=wi.to(dev))
            all_scores.extend(logits.cpu().tolist())
            all_labels.extend(lb.cpu().tolist())
        if n_utt >= a.holdout_utterances:
            break
    per_out = {l: round(v[0] / max(v[1], 1), 5) for l, v in res_per.items()} | {'utterances': n_utt}
    ver_out = roc_metrics(all_scores, all_labels)
    return {'384': per_out | {'verifier': ver_out}}


def save_verifier(model, vocab, h, a, step, seen, metrics, path):
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({
        'model': sd,
        'model_kind': VERIFIER_KIND,
        'vocab': vocab,
        'hidden': h,
        'gru_hidden': 256,
        'gru_layers': 3,
        'attn_dim': 256,
        'attn_heads': 4,
        'bins': 64,
        'dropout': a.dropout,
        'step': step,
        'feature_version': FEATURE_VERSION,
        'text_version': TEXT_VERSION,
        'pronunciation_label_versions': ['de-lexicon-v1', 'espeak-v1'],
        'parameters': sum(v.numel() for v in model.parameters()),
        'args': vars(a),
        'training_audio_hours_seen': seen / 3600,
        'holdout_metrics': metrics,
        'study': 'r218-biphoneme-verifier',
    }, path)


def train(a):
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(a.vocab).read_text())
    cons_info = build_consonant_pools(vocab)
    flap_pair = {vocab[c] for c in ('t', 'd') if c in vocab}
    all_ids = sorted(vocab.values())
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    h = int(a.hidden.split(',')[0])
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed)
    front = LogMel(64).to(dev)
    model = BiPhonemeVerifier(len(vocab) + 1, hidden=h, dropout=a.dropout, bins=64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(a.warmup, a.init_step, a.steps, getattr(a, 'cosine_restart', False))
    )
    start_step = 0
    resume = out / 'resume.pt'
    if a.init_step and not resume.exists():
        ck_path = Path(a.init_dir) / f'v{h}-step{a.init_step:06d}.pt'
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        for _ in range(a.init_step):
            sched.step()
        start_step = a.init_step
        print(json.dumps({'initialized_verifier_from_step': a.init_step, 'cosine_restart': getattr(a, 'cosine_restart', False)}), flush=True)
    if resume.exists():
        st = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start_step = st['step']

    # Warmup Numba JIT before data workers spin
    _dummy_lp = np.zeros((16, len(vocab) + 1), dtype=np.float32)
    _dummy_t = np.array([1, 2, 3], dtype=np.int64)
    viterbi_phoneme_spans(_dummy_lp, _dummy_t)

    PAD_NOISE[0] = a.pad_noise
    spec_aug = getattr(a, 'spec_aug', 0.0)
    max_sec = min(a.max_seconds, 12.0)
    n_workers = min(a.workers, 4)
    data = torch.utils.data.DataLoader(
        Stream(root, max_sec, a.seed + start_step, limit_files=a.limit_files, negative_prob=a.negative_prob,
               aug=(a.aug_bg_prob, a.aug_music_share, a.aug_snr_low, a.aug_snr_high, a.aug_far),
               mix=parse_mix(a.mix), mix_active=a.mix_active, stream_seconds=0.0),
        batch_size=a.batch, num_workers=n_workers, collate_fn=collate,
        prefetch_factor=2 if n_workers else None, persistent_workers=bool(n_workers)
    )
    ctc = torch.nn.CTCLoss(zero_infinity=True)
    rng = random.Random(a.seed + start_step)
    skipped = 0
    log = (out / 'train.jsonl').open('a')
    t0 = time.time(); seen = 0.0; it = iter(data); step = start_step
    while step < a.steps:
        audio, lengths, targets, tl, langs = next(it)
        audio = audio.to(dev, non_blocking=True)
        fl = frames(lengths).to(dev)
        seen += lengths.sum().item() / RATE
        with torch.no_grad():
            clean_feats = front(audio)[:, 0::2]
            feats = spec_augment_feats(clean_feats, fl, spec_aug, rng) if spec_aug > 0.0 else clean_feats

        model.train()
        opt.zero_grad(set_to_none=True)
        use_amp = (dev == 'cuda' and a.amp)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
            _, bi_lp = model.encode_audio(feats, fl)
        speech_mask = tl > 0
        if speech_mask.any():
            loss_ctc = ctc(bi_lp[speech_mask].float().transpose(0, 1), targets.to(dev), fl[speech_mask], tl[speech_mask].to(dev))
        else:
            loss_ctc = bi_lp.sum() * 0.0

        w_ver = min(1.0, max(0.0, (step - 150) / 450.0)) if a.steps >= 1000 else 1.0
        loss_ver = loss_ctc.new_tensor(0.0)
        n_pairs = 0
        if w_ver > 0.0:
            vb = build_verification_pairs(
                feats.detach().cpu(), fl.detach().cpu(), targets.cpu(), tl.cpu(), langs,
                bi_lp.detach().float().cpu().numpy(), cons_info, all_ids, rng,
                hard_rhyme=True, flap_pair=flap_pair
            )
            if vb is not None:
                wf, wl, wi, sb, qt, ql, lb = vb
                n_pairs = int(lb.shape[0])
                n_pos = float((lb > 0.5).sum().item())
                n_neg = float((lb < 0.5).sum().item())
                pw = torch.tensor(max(1.0, min(5.0, n_neg / max(n_pos, 1.0))), device=dev, dtype=torch.float32)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=use_amp):
                    logits, _ = model.verify_windows(wf.to(dev), wl.to(dev), sb.to(dev), qt.to(dev), ql.to(dev), win_idx=wi.to(dev))
                loss_ver = F.binary_cross_entropy_with_logits(logits.float(), lb.to(dev), pos_weight=pw)

        loss = loss_ctc + w_ver * loss_ver

        if torch.isfinite(loss):
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if torch.isfinite(gn):
                opt.step()
            else:
                skipped += 1
        else:
            skipped += 1
        sched.step()
        step += 1

        if step % 50 == 0:
            row = dict(
                step=step, hours_seen=seen / 3600, seconds=time.time() - t0, lr=sched.get_last_lr()[0],
                loss={str(h): round(loss.item(), 4)}, ctc_loss=round(loss_ctc.item(), 4),
                ver_loss=round(loss_ver.item(), 4), pairs=n_pairs, skipped={str(h): skipped},
                audio_hours_per_wall_hour=seen / max(time.time() - t0, 1e-9), mem_avail_gb=mem_available_gb()
            )
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)

        if step % 1000 == 0 or step % a.save_every == 0 or step == a.steps:
            torch.save({'step': step, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict()}, out / 'resume.tmp')
            os.replace(out / 'resume.tmp', resume)

        if step % a.save_every == 0 or step == a.steps:
            metrics = evaluate_holdout(model, root, front, dev, cons_info, all_ids, a) if (step % a.eval_every == 0 or step == a.steps) else None
            save_verifier(model, vocab, h, a, step, seen, metrics and metrics[str(h)], out / f'v{h}-step{step:06d}.pt')
            if metrics:
                log.write(json.dumps(dict(step=step, holdout_per=metrics)) + '\n'); log.flush()
                print(json.dumps(dict(step=step, holdout_per=metrics)), flush=True)

    (out / 'DONE.json').write_text(json.dumps(dict(step=step, hours_seen=seen / 3600)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train'])
    p.add_argument('--root', default='/content/r211'); p.add_argument('--out', default='/content/r211/out')
    p.add_argument('--vocab', default='reports/round211/vocab.json')
    p.add_argument('--hidden', default='384'); p.add_argument('--steps', type=int, default=20000); p.add_argument('--batch', type=int, default=48)
    p.add_argument('--lr', type=float, default=1.2e-3); p.add_argument('--warmup', type=int, default=1000); p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max-seconds', type=float, default=12.0); p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--seed', type=int, default=2180); p.add_argument('--save-every', type=int, default=2000); p.add_argument('--eval-every', type=int, default=4000)
    p.add_argument('--holdout-utterances', type=int, default=1200); p.add_argument('--limit-files', type=int, default=None)
    p.add_argument('--amp', action='store_true', default=True); p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--aug-far', type=float, default=0.35); p.add_argument('--mix', default=''); p.add_argument('--mix-active', type=int, default=4)
    p.add_argument('--negative-prob', type=float, default=0.16); p.add_argument('--aug-bg-prob', type=float, default=0.65)
    p.add_argument('--aug-music-share', type=float, default=0.45); p.add_argument('--aug-snr-low', type=float, default=-3.0); p.add_argument('--aug-snr-high', type=float, default=22.0)
    p.add_argument('--carry-prob', type=float, default=0.0); p.add_argument('--pad-noise', type=float, default=0.0); p.add_argument('--stream-seconds', type=float, default=0.0)
    p.add_argument('--cold-start-prob', type=float, default=0.0); p.add_argument('--spec-aug', type=float, default=0.0); p.add_argument('--cosine-restart', action='store_true')
    p.add_argument('--init-dir', default='/content/r211/init'); p.add_argument('--init-step', type=int, default=0)
    a = p.parse_args()
    train(a)


if __name__ == '__main__':
    main()
