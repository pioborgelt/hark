"""Matched .05-loss modes; retain the frozen edited-control implementation."""
from .segment_training_r168 import microbatch_backward as edited_backward
from .segment_training_r202 import microbatch_backward as mined_backward


def backward(*args, alternative_mode, mining_counts=None, **kwargs):
    if alternative_mode=='edited':
        return edited_backward(*args,**kwargs)
    if alternative_mode=='mined':
        return mined_backward(*args,mining_counts=mining_counts,**kwargs)
    raise ValueError('Expected edited or mined training alternatives')
