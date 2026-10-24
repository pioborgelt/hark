"""Text-conditioned thresholds from generic negatives, without positive audio."""
import hashlib
from pathlib import Path

import numpy as np

from .ctc_absolute import KERNEL_SHA256
from .fast_match import decode

VERSION = 'negative-text-calibration-v1'


def calibrate(tokens, streams, seconds, fixed, thresholds, max_events_per_hour=1.):
    """Least strict feasible path threshold; absolute evidence is never relaxed."""
    if set(streams) != set(seconds) or 'music' not in streams or any(s <= 0 for s in seconds.values()):
        raise ValueError('Complete negative stream durations required')
    hours = sum(seconds.values()) / 3600
    trials = []
    for threshold in sorted(set(thresholds)):
        settings = {**fixed, 'threshold': threshold}
        counts = {name: len(decode(lp, np.asarray(tokens, np.int64), **settings)) for name, lp in streams.items()}
        feasible = counts['music'] == 0 and sum(counts.values()) / hours <= max_events_per_hour
        trials.append({'threshold': threshold, 'events': counts, 'feasible': feasible})
        if feasible:
            return {'parameters': settings, 'feasible': True, 'trials': trials, 'negative_hours': hours}
    return {'parameters': {**fixed, 'threshold': max(thresholds)}, 'feasible': False,
            'trials': trials, 'negative_hours': hours}


def load_negative_cache(descriptor, checkpoint_sha256, base=Path('.')):
    if descriptor['version'] != VERSION or descriptor['checkpoint_sha256'] != checkpoint_sha256 or descriptor['kernel_sha256'] != KERNEL_SHA256:
        raise ValueError('Negative calibration model or decoder changed')
    streams = {}
    for name, info in descriptor['streams'].items():
        path = Path(info['path'])
        if not path.is_absolute():
            path = base / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Negative calibration logits changed')
        streams[name] = np.load(path, mmap_mode='r')
    return streams


def for_query(parameters, query_id):
    if 'per_query' not in parameters:
        return parameters
    if query_id not in parameters['per_query']:
        raise ValueError('Query was not calibrated before final inference')
    return parameters['per_query'][query_id]
