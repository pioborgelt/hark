"""Train phone discrimination without directly pushing down the blank logit."""
import torch
from .timed_phone_loss import timed_phone_loss

def conditional_timed_phone_loss(log_probs,targets,lengths,lookahead=3):
    if log_probs.ndim!=3 or log_probs.shape[-1]<2:
        raise ValueError('Need batch, time and blank plus phone classes')
    # For every nonblank target this equals z_phone-logsumexp(z_phones).
    # The common full-softmax denominator cancels, leaving zero derivative
    # with respect to the blank logit. CTC still supervises blank probability.
    conditional=log_probs-torch.logsumexp(log_probs[...,1:],dim=-1,keepdim=True)
    return timed_phone_loss(conditional,targets,lengths,lookahead=lookahead)
