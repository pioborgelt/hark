"""Training-only near-phone CTC alternatives from own acoustic head weights.

Alternative sequences are decoys, never replacement transcripts. Generation
uses a separate RNG and excludes protected full-call sequences. The loss has
no inference-time query rules and does not apply to empty music targets.
"""
import math
import torch
from torch.nn import functional as F


def head_neighbors(head, count=3):
    if head.ndim != 2 or len(head) < 3 or not torch.isfinite(head).all():
        raise ValueError('Need finite own blank-plus-phone head weights')
    if not 1 <= count <= len(head)-2:
        raise ValueError('Invalid phone-neighbor count')
    value = F.normalize(head.detach().cpu()[1:], dim=1)
    similarity = value @ value.T
    similarity.fill_diagonal_(-float('inf'))
    order = similarity.argsort(dim=1, descending=True, stable=True)[:, :count]+1
    return {i+1: row.tolist() for i, row in enumerate(order)}


def _contains(sequence, forbidden):
    return any(sequence[i:i+len(part)] == part
               for part in forbidden for i in range(len(sequence)-len(part)+1))


def alternatives(targets, target_lengths, input_lengths, neighbors, rng, forbidden=()):
    """Return one feasible single-phone decoy per nonempty target when possible."""
    values = targets.detach().cpu().tolist()
    lengths = target_lengths.detach().cpu().tolist()
    frames = input_lengths.detach().cpu().tolist()
    if len(lengths) != len(frames) or sum(lengths) != len(values):
        raise ValueError('Target/input lengths do not match the batch')
    if any(n < 0 for n in lengths) or any(n <= 0 for n in frames) or any(v <= 0 for v in values):
        raise ValueError('Invalid CTC target or input length')
    forbidden = tuple(tuple(part) for part in forbidden)
    if any(not part or any(v <= 0 for v in part) for part in forbidden):
        raise ValueError('Protected targets must be nonempty phone sequences')
    out = []; active = []; offset = 0
    for length, available in zip(lengths, frames):
        source = values[offset:offset+length]; offset += length
        changed = None
        positions = list(range(length)); rng.shuffle(positions)
        for position in positions:
            options = list(neighbors[source[position]]); rng.shuffle(options)
            for phone in options:
                if phone <= 0 or phone == source[position]:
                    raise ValueError('A neighbor must be a different nonblank phone')
                candidate = source.copy(); candidate[position] = phone
                required = len(candidate)+sum(a == b for a, b in zip(candidate, candidate[1:]))
                if required > available or _contains(tuple(candidate), forbidden):
                    continue
                changed = candidate; break
            if changed is not None: break
        out.extend(source if changed is None else changed); active.append(changed is not None)
    return (torch.tensor(out, dtype=targets.dtype, device=targets.device),
            torch.tensor(active, dtype=torch.bool, device=targets.device))


def confusion_loss(ctc, lp, targets, decoys, input_lengths, target_lengths, active,
                   margin=1., true_losses=None):
    """Per-example soft margin between the true and one edited CTC sequence.

The margin measures whole-sequence log probability; division by target length
matches the ordinary per-phone CTC objective. Empty/unsampleable rows contribute
exactly zero, including their gradients. Both paths use the actual own logits.
"""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid CTC log-probability margin')
    if ctc.reduction != 'none' or ctc.blank != 0 or ctc.zero_infinity:
        raise ValueError('Need unreduced CTC with blank0 and zero_infinity=False')
    if targets.shape != decoys.shape or active.shape != target_lengths.shape:
        raise ValueError('Decoy/mask shape mismatch')
    if true_losses is None:
        true_losses = ctc(lp.transpose(0, 1), targets, input_lengths, target_lengths)
    if true_losses.shape != target_lengths.shape:
        raise ValueError('True CTC losses must be one value per example')
    wrong_losses = ctc(lp.transpose(0, 1), decoys, input_lengths, target_lengths)
    if not torch.isfinite(true_losses).all() or not torch.isfinite(wrong_losses).all():
        raise ValueError('Infeasible or nonfinite CTC alternatives')
    scale = target_lengths.to(lp.device).clamp_min(1)
    mask = active.to(lp.device) & (target_lengths.to(lp.device) > 0)
    return torch.where(mask, F.softplus(margin+true_losses-wrong_losses)/scale,
                       torch.zeros_like(true_losses))
