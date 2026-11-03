"""Ordinary whole-utterance training plus a short-segment contrastive term."""
import math

import torch

from .train_music_context_r77 import microbatch_backward as ordinary_backward
from .train_music_context_r77 import normalized_ctc_loss
from .timed_conditional_phone_loss import conditional_timed_phone_loss
from .timed_phone_loss import timed_phone_loss
from .music_peak_loss import music_peak_blank_loss
from .mined_segment_r202 import segment_loss


def microbatch_backward(model, ctc, x, y, input_lengths, target_lengths, labels,
                        aux_weight, blank_weight, microbatch,
                        alignment_mode='unconditional', music_peak_weight=0.,
                        segment_weight=0., neighbors=None, rng=None,
                        forbidden=(), windows=None, mining_counts=None):
    if not math.isfinite(segment_weight) or segment_weight < 0:
        raise ValueError('Invalid segment loss weight')
    if not segment_weight:
        return (*ordinary_backward(model, ctc, x, y, input_lengths, target_lengths,
                    labels, aux_weight, blank_weight, microbatch,
                    alignment_mode, music_peak_weight), 0.)
    if alignment_mode not in ['conditional', 'unconditional']:
        raise ValueError('Unknown alignment objective')
    if microbatch < 1 or windows is None or len(windows) != len(x):
        raise ValueError('Invalid microbatch or window count')
    for choices, length, phones in zip(windows, input_lengths, target_lengths):
        if not int(phones) and choices:
            raise ValueError('Empty/music targets cannot receive segment targets')
        if any(not 0 <= begin < end <= int(length) for begin, end, _ in choices):
            raise ValueError('Window outside actual input length')
    offsets = torch.cat([target_lengths.new_zeros(1), target_lengths.cumsum(0)])
    totals = [0., 0., 0., 0.]
    for begin in range(0, len(x), microbatch):
        end = min(begin+microbatch, len(x)); fraction = (end-begin)/len(x)
        lp, _ = model(x[begin:end])
        il, tl = input_lengths[begin:end], target_lengths[begin:end]
        truth = y[int(offsets[begin]):int(offsets[end])]
        c = normalized_ctc_loss(ctc, lp, truth, il, tl, blank_weight)
        objective = conditional_timed_phone_loss if alignment_mode == 'conditional' else timed_phone_loss
        aux = objective(lp, labels[begin:end], il).mean() if aux_weight else lp.new_zeros(())
        peak = music_peak_blank_loss(lp, il, tl).mean() if music_peak_weight else lp.new_zeros(())
        extra = segment_loss(ctc, lp, windows[begin:end], neighbors, rng, forbidden, counters=mining_counts).mean()
        loss = (c+aux_weight*aux+music_peak_weight*peak+segment_weight*extra)*fraction
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite short-segment microbatch loss')
        loss.backward()
        for index, value in enumerate((c, aux, peak, extra)):
            totals[index] += float(value.detach())*fraction
    return tuple(totals)
