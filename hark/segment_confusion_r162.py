"""Generic short aligned-phone alternatives, derived only from original labels.

No query-conditioned loss or teacher. Mismatches, skipped alignment positions,
long gaps and infeasible CTC windows are excluded. Original whole-utterance CTC
and timed targets remain the trainer's responsibility and are never replaced.
"""
import math
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .contrastive_ctc_r110 import _contains


def aligned_windows(record, vocab, frames, lengths=(4, 8, 12), forbidden=()):
    """Exact contiguous matches under the original alignment's edit traceback.

    Windows start every four target codepoints. Frame centers match alignment.py;
    no added future padding can introduce an unlabelled neighbouring phone.
    Returned timing metadata stays in RAM, and contains no audio or model output.
    """
    if frames <= 0 or not lengths or any(n < 2 for n in lengths):
        raise ValueError('Need positive frames and segment lengths >=2')
    target = list(record['phones'])
    source, spans = [], []
    previous_end = 0.
    for label in record.get('labels', []):
        start, end = label['start'], label['end']
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError('Invalid annotation interval')
        if start < previous_end-1e-6:
            raise ValueError('Overlapping or unordered annotations')
        previous_end = end
        if label['symbol'] == '(...)':
            continue
        chars = [c for c in unicodedata.normalize('NFD', label['symbol'])
                 if c not in ' ˈˌː' and not c.isspace()]
        for i, char in enumerate(chars):
            source.append(char)
            spans.append((start+(end-start)*i/len(chars), start+(end-start)*(i+1)/len(chars)))
    if not source or not target:
        return []
    n, m = len(source), len(target)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    dp[:, 0] = np.arange(n+1); dp[0] = np.arange(m+1)
    for i, char in enumerate(source, 1):
        for j, expected in enumerate(target, 1):
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1,
                           dp[i-1, j-1]+(char != expected))
    if dp[n, m]/max(n, m) > .5:
        return []
    pairs = {}; i, j = n, m
    while i and j:
        if dp[i, j] == dp[i-1, j-1]+(source[i-1] != target[j-1]):
            if source[i-1] == target[j-1]:
                pairs[j-1] = i-1
            i -= 1; j -= 1
        elif dp[i, j] == dp[i-1, j]+1:
            i -= 1
        else:
            j -= 1
    windows = []
    for length in lengths:
        for offset in range(0, m-length+1, 4):
            positions = [pairs.get(k, -1) for k in range(offset, offset+length)]
            if positions[0] < 0 or positions != list(range(positions[0], positions[0]+length)):
                continue
            if any(spans[b][0]-spans[a][1] > .2 for a, b in zip(positions, positions[1:])):
                continue
            truth_chars = target[offset:offset+length]
            if any(c not in vocab for c in truth_chars):
                continue
            truth = tuple(vocab[c] for c in truth_chars)
            if _contains(truth, forbidden):
                continue
            begin = max(0, math.ceil((spans[positions[0]][0]-.0125)/.02))
            end = min(frames, math.ceil((spans[positions[-1]][1]-.0125)/.02))
            required = len(truth)+sum(a == b for a, b in zip(truth, truth[1:]))
            if end-begin >= required:
                windows.append((begin, end, truth))
    return windows


def edited_segment(truth, available, neighbors, rng, forbidden=()):
    """One feasible insertion, deletion or neighbour substitution; private RNG."""
    if not truth or any(v <= 0 for v in truth):
        raise ValueError('Need nonblank segment ground truth')
    modes = ['substitute', 'insert', 'delete']; rng.shuffle(modes)
    positions = list(range(len(truth))); rng.shuffle(positions)
    for mode in modes:
        for position in positions:
            options = [None] if mode == 'delete' else list(neighbors[truth[position]])
            rng.shuffle(options)
            for phone in options:
                value = list(truth)
                if mode == 'delete':
                    del value[position]
                else:
                    if phone <= 0 or phone == truth[position]:
                        raise ValueError('Expected a different nonblank neighbour')
                    if mode == 'substitute':
                        value[position] = phone
                    else:
                        value.insert(position, phone)
                required = len(value)+sum(a == b for a, b in zip(value, value[1:]))
                if value and required <= available and not _contains(tuple(value), forbidden):
                    return tuple(value), mode
    return None


def segment_loss(ctc, lp, windows, neighbors, rng, forbidden=(), margin=1.):
    """At most one window per example, with zero contribution for missing labels."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid CTC margin')
    if ctc.reduction != 'none' or ctc.blank != 0 or ctc.zero_infinity:
        raise ValueError('Need unreduced finite CTC with blank0')
    if len(windows) != len(lp):
        raise ValueError('One window list per batch example required')
    indices, slices, truths, decoys, sizes = [], [], [], [], []
    for index, choices in enumerate(windows):
        if not choices:
            continue
        begin, end, truth = rng.choice(choices)
        if not 0 <= begin < end <= lp.shape[1]:
            raise ValueError('Segment outside actual model frames')
        if _contains(tuple(truth), forbidden):
            raise ValueError('Protected full call in segment truth')
        changed = edited_segment(truth, end-begin, neighbors, rng, forbidden)
        if changed is None:
            continue
        decoy, _ = changed
        indices.append(index); slices.append(lp[index, begin:end])
        truths.append(truth); decoys.append(decoy); sizes.append(end-begin)
    # Retain a graph with exactly zero derivative for all empty/music examples.
    output = lp.sum(dim=(1, 2))*0
    if not slices:
        return output
    packed = pad_sequence(slices, batch_first=False)
    il = torch.tensor(sizes, dtype=torch.long)
    tl = torch.tensor([len(t) for t in truths], dtype=torch.long)
    dl = torch.tensor([len(t) for t in decoys], dtype=torch.long)
    truth_tensor = torch.tensor([p for t in truths for p in t], dtype=torch.long, device=lp.device)
    decoy_tensor = torch.tensor([p for t in decoys for p in t], dtype=torch.long, device=lp.device)
    true_cost = ctc(packed, truth_tensor, il, tl)
    wrong_cost = ctc(packed, decoy_tensor, il, dl)
    if not torch.isfinite(true_cost).all() or not torch.isfinite(wrong_cost).all():
        raise ValueError('Nonfinite or infeasible segment CTC')
    value = F.softplus(margin+true_cost-wrong_cost)/tl.to(lp.device)
    return output.index_add(0, torch.tensor(indices, device=lp.device), value)
