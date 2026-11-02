"""Emphasize rare nonblank peaks in known music-only training presentations."""
import math
import torch

def music_peak_blank_loss(log_probs, lengths, target_lengths, fraction=.05):
    if log_probs.ndim!=3 or len(lengths)!=len(log_probs) or len(target_lengths)!=len(log_probs):
        raise ValueError('Need matching batch/time/class probabilities and lengths')
    if not 0<fraction<=1:
        raise ValueError('Invalid tail fraction')
    values=[]
    for i,(length,targets) in enumerate(zip(lengths,target_lengths)):
        n=int(length)
        if not 1<=n<=log_probs.shape[1] or int(targets)<0:
            raise ValueError('Invalid sequence length')
        if int(targets):
            values.append(log_probs[i,:n,0].sum()*0.)
        else:
            k=max(1,math.ceil(n*fraction))
            values.append((-log_probs[i,:n,0]).topk(k).values.mean())
    return torch.stack(values)
