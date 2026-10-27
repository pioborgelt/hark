"""Training-only supervision from existing phonetic time annotations."""
import torch
import torch.nn.functional as F

def timed_phone_loss(log_probs,targets,lengths,lookahead=3):
    """Match annotated phones within the next 0..60 ms; ignore blanks/unknowns.

    Future frames are used only in the training objective. Inference remains
    the original causal encoder. Padded frames never provide phone evidence.
    """
    if log_probs.ndim!=3 or targets.shape!=log_probs.shape[:2]:raise ValueError('Timed phone shape mismatch')
    if not isinstance(lookahead,int) or isinstance(lookahead,bool) or lookahead<0 or lookahead>10:raise ValueError('Invalid timing tolerance')
    lengths=torch.as_tensor(lengths,device=log_probs.device)
    if lengths.shape!=(len(log_probs),) or torch.any(lengths<=0) or torch.any(lengths>log_probs.shape[1]):raise ValueError('Invalid lengths')
    if targets.dtype!=torch.long or targets.device!=log_probs.device:raise ValueError('Integer targets on model device required')
    if torch.any((targets<0)&(targets!=-100)) or torch.any(targets>=log_probs.shape[2]):raise ValueError('Invalid phone targets')
    frames=torch.arange(log_probs.shape[1],device=log_probs.device)[None,:]<lengths[:,None]
    masked=log_probs.masked_fill(~frames[...,None],float('-inf')).transpose(1,2)
    future=F.max_pool1d(F.pad(masked,(0,lookahead),value=float('-inf')),lookahead+1,stride=1).transpose(1,2)
    valid=(targets>0)&frames
    nll=-future.gather(2,targets.clamp_min(0)[...,None]).squeeze(-1)
    nll=nll.masked_fill(~valid,0.)
    return nll.sum(-1)/valid.sum(-1).clamp_min(1)
