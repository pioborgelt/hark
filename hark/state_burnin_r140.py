"""Reset or detached silence/music/speech contexts from existing training only."""
import torch
from .train_music_context_r77 import normalize
from .state_burnin_r132 import burnin_state,expand_state


def sample_prefix(backgrounds,speech,rng,device):
    if rng.random()<.5:return None,{'kind':'reset','frames':0}
    kind=rng.choice(['silence','training-music','training-speech'])
    frames=rng.choice([400,1500,6000])
    if kind=='silence':return torch.zeros(1,frames,64,device=device),{'kind':kind,'frames':frames}
    pool,count=(backgrounds,6) if kind=='training-music' else (speech,64)
    if len(pool)<count:raise ValueError('Need enough distinct retained training windows')
    indices=rng.sample(range(len(pool)),count)
    value=torch.cat([normalize(pool[i].float()) for i in indices])
    if len(value)<frames:raise ValueError('Retained prefix too short; never tile')
    start=rng.randrange(len(value)-frames+1)
    return value[start:start+frames].unsqueeze(0).to(device),{
        'kind':kind,'frames':frames,'indices':indices,'start_frame':start}
