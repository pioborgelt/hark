"""Open-vocabulary text-enrolled bidirectional second-stage wake-word verifier (R218).

Operates strictly on short candidate windows (~1.5-2.8 s) around Stage-1 streaming triggers and the
deterministic phoneme token sequence from `hark.text.encode` (zero keyword audio enrollment).
Combines a 3-layer bidirectional dilated-conv + BiGRU phoneme encoder (trained jointly with CTC)
and a 2-layer phoneme-to-frame cross-attention discriminator with min/mean phone pooling and
bidirectional CTC span alignment features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

VERIFIER_KIND = 'own-biphoneme-verifier-v1'


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, kv_pad_mask):
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        out, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_pad_mask, need_weights=False)
        q = q + out
        q = q + self.ffn(self.ffn_norm(q))
        return q


class BiPhonemeVerifier(nn.Module):
    def __init__(self, classes, hidden=384, gru_hidden=256, gru_layers=3, attn_dim=256, attn_heads=4, dropout=0.1, bins=64):
        super().__init__()
        self.classes = classes
        self.hidden = hidden
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.attn_dim = attn_dim
        self.attn_heads = attn_heads
        self.bins = bins

        self.in_proj = nn.Sequential(nn.Linear(bins, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.convs = nn.ModuleList([
            nn.ModuleDict({
                'dw': nn.Conv1d(hidden, hidden, 5, padding=2 * (2 ** i), dilation=2 ** i, groups=hidden),
                'pw': nn.Conv1d(hidden, hidden, 1),
                'norm': nn.LayerNorm(hidden),
            })
            for i in range(3)
        ])
        self.drop = nn.Dropout(dropout)
        self.bigru = nn.GRU(
            hidden, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0
        )
        enc_dim = gru_hidden * 2
        self.ctc_head = nn.Linear(enc_dim, classes)

        self.kv_proj = nn.Sequential(
            nn.Linear(enc_dim + classes + 4, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.SiLU(),
        )
        self.tok_emb = nn.Embedding(classes, attn_dim - 1, padding_idx=0)
        self.tok_gru = nn.GRU(attn_dim, attn_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.xattn = nn.ModuleList([CrossAttnBlock(attn_dim, attn_heads, dropout) for _ in range(2)])
        self.phone_score = nn.Linear(attn_dim, 1)
        self.feat_drop = nn.Dropout(0.15)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim * 2 + 7, attn_dim),
            nn.LayerNorm(attn_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def encode_audio(self, features, lengths):
        B, T, _ = features.shape
        idx = torch.arange(T, device=features.device)[None, :]
        valid = (idx < lengths[:, None]).unsqueeze(-1).to(features.dtype)
        x = self.in_proj(features) * valid
        for blk in self.convs:
            r = blk['pw'](F.silu(blk['dw'](x.transpose(1, 2)))).transpose(1, 2)
            x = (x + self.drop(F.silu(blk['norm'](r)))) * valid
        packed = nn.utils.rnn.pack_padded_sequence(
            self.drop(x), lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.bigru(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        bi_lp = self.ctc_head(self.drop(enc)).float().log_softmax(-1)
        return enc, bi_lp

    def _bi_ctc_features(self, bi_lp, win_lens, span_bounds, tokens, token_lens):
        with torch.amp.autocast(bi_lp.device.type, enabled=False):
            bi_lp = bi_lp.float()
            N, W, C = bi_lp.shape
            dev = bi_lp.device
            s0 = span_bounds[:, 0].clamp(0, W - 1)
            s1 = torch.maximum(s0 + 1, torch.minimum(span_bounds[:, 1], win_lens.clamp_min(1)))
            span_lens = s1 - s0
            max_S = int(span_lens.max().item())

            span_lp = bi_lp.new_zeros((max_S, N, C))
            flat_targets = []
            for i in range(N):
                L_i = int(token_lens[i].item())
                S_i = int(span_lens[i].item())
                span_lp[:S_i, i, :] = bi_lp[i, s0[i]:s1[i], :]
                flat_targets.append(tokens[i, :L_i])
            cat_targets = torch.cat(flat_targets, dim=0)
            ctc_nll = F.ctc_loss(
                span_lp, cat_targets, span_lens.cpu(), token_lens.cpu(), reduction='none', zero_infinity=True
            )
            ctc_score = (-ctc_nll / token_lens.to(bi_lp.dtype).clamp_min(1.0)).clamp(-8.0, 0.0)
            ctc_score = torch.where(span_lens >= token_lens, ctc_score, ctc_score.new_full((N,), -8.0))

            frame_top = bi_lp.max(dim=-1).values
            t_idx = torch.arange(W, device=dev)[None, :]
            in_ext = (t_idx >= (s0 - 2).clamp_min(0)[:, None]) & (t_idx < torch.minimum(win_lens, s1 + 2)[:, None])
            peak_means = bi_lp.new_zeros(N)
            peak_mins = bi_lp.new_zeros(N)
            left_b = bi_lp.new_zeros(N)
            right_b = bi_lp.new_zeros(N)
            for i in range(N):
                L_i = int(token_lens[i].item())
                toks_i = tokens[i, :L_i]
                mask_i = in_ext[i]
                if mask_i.any():
                    sub_lp = bi_lp[i, mask_i][:, toks_i] - frame_top[i, mask_i][:, None]
                    pks = sub_lp.max(dim=0).values.clamp(-8.0, 0.0)
                    peak_means[i] = pks.mean()
                    peak_mins[i] = pks.min()
                else:
                    peak_means[i] = -8.0
                    peak_mins[i] = -8.0
                a0 = max(0, int(s0[i].item()) - 4)
                a1 = min(int(win_lens[i].item()), int(s0[i].item()) + 1)
                left_b[i] = bi_lp[i, a0:max(a0 + 1, a1), 0].max().clamp(-6.0, 0.0)
                b0 = max(0, int(s1[i].item()) - 1)
                b1 = min(int(win_lens[i].item()), int(s1[i].item()) + 4)
                right_b[i] = bi_lp[i, b0:max(b0 + 1, b1), 0].max().clamp(-6.0, 0.0)
            return torch.stack([ctc_score, peak_means, peak_mins, left_b, right_b], dim=-1)

    def verify_windows(self, win_feats, win_lens, span_bounds, tokens, token_lens, win_idx=None):
        u_enc, u_bi_lp = self.encode_audio(win_feats, win_lens)
        if win_idx is not None:
            enc = u_enc[win_idx]
            bi_lp = u_bi_lp[win_idx]
            pair_lens = win_lens[win_idx]
        else:
            enc = u_enc
            bi_lp = u_bi_lp
            pair_lens = win_lens
        N, W, _ = enc.shape
        L = tokens.shape[1]
        dev = enc.device
        dtype = enc.dtype

        t_idx = torch.arange(W, device=dev, dtype=dtype)[None, :]
        s0 = span_bounds[:, 0:1].to(dtype)
        s1 = torch.maximum(s0 + 1.0, span_bounds[:, 1:2].to(dtype))
        in_span = ((t_idx >= s0) & (t_idx < s1)).to(dtype)
        rel_pos = ((t_idx - s0) / (s1 - s0).clamp_min(1.0)).clamp(-1.5, 2.5)
        probs = bi_lp.exp().to(dtype)
        blank_p = probs[:, :, 0]
        speech_p = 1.0 - blank_p
        span_feats = torch.stack([in_span, rel_pos, blank_p, speech_p], dim=-1)

        kv = self.kv_proj(torch.cat([enc, probs, span_feats], dim=-1))
        kv_pad = torch.arange(W, device=dev)[None, :] >= pair_lens.clamp_min(1)[:, None]

        p_idx = torch.arange(L, device=dev, dtype=dtype)[None, :] / (token_lens[:, None].to(dtype) - 1.0).clamp_min(1.0)
        p_idx = p_idx.clamp(0.0, 1.0).unsqueeze(-1)
        q_in = torch.cat([self.tok_emb(tokens).to(dtype), p_idx], dim=-1)
        q_pack = nn.utils.rnn.pack_padded_sequence(
            q_in, token_lens.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        q_out_pack, _ = self.tok_gru(q_pack)
        q, _ = nn.utils.rnn.pad_packed_sequence(q_out_pack, batch_first=True, total_length=L)

        for blk in self.xattn:
            q = blk(q, kv, kv_pad)

        tok_valid = torch.arange(L, device=dev)[None, :] < token_lens[:, None]
        tok_mask = tok_valid.unsqueeze(-1).to(q.dtype)
        q_mean = (q * tok_mask).sum(dim=1) / token_lens[:, None].to(q.dtype).clamp_min(1.0)
        q_masked_for_min = q.masked_fill(~tok_valid.unsqueeze(-1), 1e4)
        q_min = q_masked_for_min.min(dim=1).values

        ps = self.phone_score(q).squeeze(-1).float()
        ps_mean = (ps * tok_valid.float()).sum(dim=1, keepdim=True) / token_lens[:, None].float().clamp_min(1.0)
        ps_min = ps.masked_fill(~tok_valid, 1e4).min(dim=1, keepdim=True).values

        bi_feats = self._bi_ctc_features(bi_lp.detach() if self.training else bi_lp, pair_lens, span_bounds, tokens, token_lens)
        if self.training:
            bi_feats = self.feat_drop(bi_feats)

        head_in = torch.cat([q_mean.float(), q_min.float(), ps_mean, ps_min, bi_feats], dim=-1).to(dtype)
        logits = self.classifier(head_in).squeeze(-1).float()
        return logits, bi_lp


def load_verifier(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('model_kind') != VERIFIER_KIND:
        raise ValueError(f"Expected {VERIFIER_KIND}, got {ck.get('model_kind')}")
    m = BiPhonemeVerifier(
        classes=len(ck['vocab']) + 1,
        hidden=ck.get('hidden', 384),
        gru_hidden=ck.get('gru_hidden', 256),
        gru_layers=ck.get('gru_layers', 3),
        attn_dim=ck.get('attn_dim', 256),
        attn_heads=ck.get('attn_heads', 4),
        dropout=ck.get('dropout', 0.1),
        bins=ck.get('bins', 64),
    ).to(device)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, ck


@torch.no_grad()
def score_trigger_queries(
    verifier,
    source,
    queries,
    v_front=None,
    is_features=False,
    start_shift=-4,
    end_shift=0,
    span_shifts=((-3, -1), (-4, 1)),
    left_ctx=16,
    right_ctx=14,
    batch_size=64,
    device='cpu',
):
    """Score Stage-1 candidate triggers `(end_frame, start_frame, token_sets_tuple)` in batched windows.

    Args:
        verifier: `BiPhonemeVerifier` in eval mode.
        source: 1D float32 16 kHz audio (`is_features=False`) or 2D `(T, bins)` 50 Hz features (`is_features=True`).
        queries: iterable of `(end_frame, start_frame, token_sets_tuple)` where `token_sets_tuple` is a tuple of
                 1+ phoneme-ID tuples `((t1, t2, ...), ...)`.
        v_front: `LogMel(bins)` on `device` (required when `is_features=False`).
    Returns:
        dict mapping each unique query tuple to its float Stage-2 verifier logit (max over pronunciation variants).
    """
    uniq_queries = list(dict.fromkeys(queries))
    if not uniq_queries:
        return {}

    if is_features:
        total_frames = int(source.shape[0])
    else:
        n_samp = len(source)
        if n_samp < 400:
            return {q: -20.0 for q in uniq_queries}
        n_100 = 1 + (n_samp - 400) // 160
        total_frames = (n_100 + 1) // 2

    if total_frames < 6:
        return {q: -20.0 for q in uniq_queries}

    active_shifts = span_shifts if span_shifts is not None else ((start_shift, end_shift),)

    # Deduplicate audio windows `(w0, w1)` across all queries
    win_map = {}
    win_list = []
    win_to_items = {}  # u_id -> list of (q_idx, rs0, rs1, tok_tuple)
    for q_idx, (end_f, start_f, tsets) in enumerate(uniq_queries):
        s0_min = max(0, min(total_frames - 2, int(start_f) + min(sh[0] for sh in active_shifts)))
        s1_max = max(s0_min + 2, min(total_frames, int(end_f) + 1 + max(sh[1] for sh in active_shifts)))
        w0 = max(0, s0_min - left_ctx)
        w1 = min(total_frames, max(w0 + 6, s1_max + right_ctx))
        w_key = (w0, w1)
        u_id = win_map.get(w_key)
        if u_id is None:
            u_id = len(win_list)
            win_map[w_key] = u_id
            win_list.append(w_key)
            win_to_items[u_id] = []
        for st_sh, en_sh in active_shifts:
            s0 = max(w0, min(w1 - 2, int(start_f) + st_sh))
            s1 = max(s0 + 2, min(w1, int(end_f) + 1 + en_sh))
            rs0, rs1 = s0 - w0, s1 - w0
            for tok_tup in tsets:
                if len(tok_tup) > 0:
                    win_to_items[u_id].append((q_idx, rs0, rs1, tok_tup))

    bins = verifier.bins
    q_scores = [-20.0] * len(uniq_queries)
    use_amp = (str(device).startswith('cuda'))

    for b0 in range(0, len(win_list), batch_size):
        chunk_wins = win_list[b0:b0 + batch_size]
        U = len(chunk_wins)
        extracted_feats = []
        for w0, w1 in chunk_wins:
            if is_features:
                f = torch.as_tensor(source[w0:w1], dtype=torch.float32)
            else:
                seg = source[w0 * 320:(w1 - 1) * 320 + 400]
                a_t = torch.as_tensor(seg, dtype=torch.float32, device=device)
                f = v_front(a_t)[0::2]
            extracted_feats.append(f)

        W_max = max(f.shape[0] for f in extracted_feats)
        win_feats = torch.zeros((U, W_max, bins), dtype=torch.float32, device=device)
        win_lens = torch.zeros(U, dtype=torch.long, device=device)
        for local_u, f in enumerate(extracted_feats):
            wi = f.shape[0]
            win_feats[local_u, :wi] = f.to(device)
            win_lens[local_u] = wi

        chunk_items = []
        for local_u in range(U):
            global_u = b0 + local_u
            for q_idx, rs0, rs1, tok_tup in win_to_items[global_u]:
                chunk_items.append((local_u, q_idx, rs0, rs1, tok_tup))

        if not chunk_items:
            continue

        N = len(chunk_items)
        L_max = max(len(x[4]) for x in chunk_items)
        win_idx = torch.zeros(N, dtype=torch.long, device=device)
        span_bounds = torch.zeros((N, 2), dtype=torch.long, device=device)
        q_tokens = torch.zeros((N, L_max), dtype=torch.long, device=device)
        q_lens = torch.zeros(N, dtype=torch.long, device=device)
        for i, (local_u, _, rs0, rs1, tok_tup) in enumerate(chunk_items):
            li = len(tok_tup)
            win_idx[i] = local_u
            span_bounds[i, 0] = rs0
            span_bounds[i, 1] = rs1
            q_tokens[i, :li] = torch.tensor(tok_tup, dtype=torch.long, device=device)
            q_lens[i] = li

        with torch.autocast('cuda' if str(device).startswith('cuda') else 'cpu', dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = verifier.verify_windows(win_feats, win_lens, span_bounds, q_tokens, q_lens, win_idx=win_idx)
        logits_list = logits.float().cpu().tolist()
        for (_, q_idx, _, _, _), lg in zip(chunk_items, logits_list):
            if q_scores[q_idx] == -20.0 or lg > q_scores[q_idx]:
                q_scores[q_idx] = float(lg)

    return {uniq_queries[i]: q_scores[i] for i in range(len(uniq_queries))}

