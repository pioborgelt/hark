"""Detached self-model context augmentation; inference and target labels unchanged."""
import torch
from .train_music_context_r77 import normalize


def sample_prefix(backgrounds, rng, device):
    frames = rng.choice([400, 1500, 6000])
    if rng.random() < .5:
        return torch.zeros(1, frames, 64, device=device), {'kind': 'silence', 'frames': frames}
    if len(backgrounds) < 6:
        raise ValueError('Need six distinct retained training music windows')
    indices = rng.sample(range(len(backgrounds)), 6)
    value = torch.cat([normalize(backgrounds[i].float()) for i in indices])
    if len(value) < frames:
        raise ValueError('Retained music too short; never tile the input')
    start = rng.randrange(len(value)-frames+1)
    return value[start:start+frames].unsqueeze(0).to(device), {
        'kind': 'training-music', 'frames': frames, 'indices': indices, 'start_frame': start}


def burnin_state(model, prefix, chunk_frames=400):
    if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[-1] != 64:
        raise ValueError('One normalized64-bin prefix is required')
    previous = model.training
    model.eval()
    try:
        state = None
        # no_grad deliberately creates normal tensors usable by suffix autograd.
        # eval avoids consuming the matched control's dropout RNG stream.
        with torch.no_grad():
            for begin in range(0, prefix.shape[1], chunk_frames):
                _, state = model(prefix[:, begin:begin+chunk_frames], state)
    finally:
        model.train(previous)
    return state


def expand_state(state, batch):
    if state is None:
        return None
    if set(state) != {'gru', 'conv'} or state['gru'].shape[1] != 1:
        raise ValueError('Only the original own causal encoder state is supported')
    if batch < 1:
        raise ValueError('Empty suffix batch')
    return {'gru': state['gru'].detach().expand(-1, batch, -1).contiguous(),
            'conv': [value.detach().expand(batch, -1, -1).contiguous() for value in state['conv']]}
