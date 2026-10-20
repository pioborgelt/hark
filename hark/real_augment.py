"""Research augmentation with actual background recordings, never test tracks."""
import numpy as np
from scipy.signal import fftconvolve
from .audio import RATE

def real_augment(audio,background,rng,snr=None,room=True):
    x=audio.copy()
    if room and rng.random()<.8:
        n=int(rng.uniform(.15,.55)*RATE)
        t=np.arange(n)/RATE
        rir=rng.normal(size=n)*np.exp(-t/rng.uniform(.04,.12))*.025
        rir[0]=1.
        for _ in range(8):rir[rng.integers(160,min(n,3200))]+=rng.uniform(-.2,.2)
        x=fftconvolve(x,rir)[:len(x)]
    if len(background)<len(x):
        background=np.tile(background,1+len(x)//len(background))
    offset=rng.integers(max(1,len(background)-len(x)+1));noise=background[offset:offset+len(x)]
    power=np.sqrt(np.mean(x*x)+1e-9)
    noise=noise/np.sqrt(np.mean(noise*noise)+1e-9)
    x=x+noise*power*10**(-(rng.uniform(0,25) if snr is None else snr)/20)
    x*=rng.uniform(.04,.8)
    return np.clip(x,-1,1).astype(np.float32)
