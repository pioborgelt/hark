"""Own-student greedy alternatives for existing exact training-label windows.

Ground-truth targets are never replaced. The detached current prediction is
only a candidate wrong-sequence competitor, with at most two ordinary phone
edits and the existing protected-call exclusions. No teacher or query scores.
"""
import torch
from .contrastive_ctc_r110 import _contains


def greedy_alternative(log_probs, truth, forbidden=(), maximum_edits=2):
    if (log_probs.ndim!=2 or not len(log_probs) or log_probs.shape[1]<2
            or not truth or any(not isinstance(v,int) or not 0<v<log_probs.shape[1] for v in truth)
            or maximum_edits not in [1,2]):
        raise ValueError('Finite frame/class posteriors, nonblank truth and one/two edits required')
    detached=log_probs.detach()
    if not torch.isfinite(detached).all():raise ValueError('Nonfinite own posterior')
    path=detached.argmax(-1).tolist();candidate=[];previous=-1
    for value in path:
        if value!=previous and value: candidate.append(value)
        previous=value
    value=tuple(candidate)
    if not value or value==tuple(truth) or abs(len(value)-len(truth))>maximum_edits or _contains(value,forbidden):
        return None
    previous=list(range(len(value)+1))
    for i,phone in enumerate(truth,1):
        current=[i]
        for j,other in enumerate(value,1):
            current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(phone!=other)))
        previous=current
    if not 1<=previous[-1]<=maximum_edits:return None
    required=len(value)+sum(a==b for a,b in zip(value,value[1:]))
    return value if required<=len(log_probs) else None


import math
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .segment_confusion_r162 import edited_segment


def segment_loss(ctc, lp, windows, neighbors, rng, forbidden=(), margin=1., counters=None):
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
        mined=greedy_alternative(lp[index,begin:end],tuple(truth),forbidden)
        if mined is not None:
            decoy=mined;route="mined"
        elif changed is not None:
            decoy,_=changed;route="fallback"
        else:
            continue
        if counters is not None:counters[route]=counters.get(route,0)+1
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
