import math
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

RATE = 16000
WINDOW = 400
HOP = 160

def read_audio(path):
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != RATE:
        d = math.gcd(rate, RATE)
        x = resample_poly(x, RATE // d, rate // d).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite audio")
    return x

class LogMel(torch.nn.Module):
    """25 ms trailing window, 10 ms hop; no centering or future samples."""
    def __init__(self, bins=40):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WINDOW))
        hz = 700 * (10 ** (torch.linspace(0, math.log10(1 + 7600 / 700), bins + 2)) - 1)
        freq = torch.linspace(0, RATE / 2, 257)
        bank = torch.minimum((freq[None] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None]),
                             (hz[2:, None] - freq[None]) / (hz[2:, None] - hz[1:-1, None])).clamp_min(0)
        self.register_buffer("bank", bank)

    def power_log(self, audio):
        frames = audio.unfold(-1, WINDOW, HOP)
        power = torch.fft.rfft(frames * self.window, n=512).abs().square()
        mel = (power @ self.bank.T).clamp_min(1e-6).log()
        return mel

    def forward(self, audio):
        mel=self.power_log(audio)
        return (mel - mel.mean(-1, keepdim=True)) / mel.std(-1, keepdim=True).clamp_min(0.1)

class FeatureStream:
    def __init__(self, bins=40):
        self.bins=bins
        self.frontend = LogMel(bins)
        self.pending = torch.empty(0)
        self.frames = 0

    def push(self, samples):
        self.pending = torch.cat([self.pending, torch.as_tensor(samples, dtype=torch.float32)])
        if len(self.pending) < WINDOW:
            return torch.empty(0, self.bins)
        n = 1 + (len(self.pending) - WINDOW) // HOP
        feats = self.frontend(self.pending[: (n - 1) * HOP + WINDOW])
        self.pending = self.pending[n * HOP:]
        # Global phase, so arbitrary input chunk sizes produce identical output.
        result = feats[(self.frames % 2)::2]
        self.frames += n
        return result
