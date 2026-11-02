"""One prospective change: top-five-percent music blank loss weight 0 -> .5."""
FIELDS = ('lr', 'room_probability', 'speech_mix_probability', 'negative_probability',
          'blank_loss_weight', 'alignment_mode', 'target_mapping', 'alignment_weight',
          'target_delay_frames', 'music_peak_weight')


def validate_requested_recipe(requested, inherited):
    if set(requested) != set(FIELDS) or any(k not in inherited for k in FIELDS):
        raise ValueError('Incomplete registered recipe')
    if inherited['music_peak_weight'] != 0. or requested['music_peak_weight'] != .5:
        raise ValueError('Only original zero-peak parent to weight .5 is registered')
    if any(requested[k] != inherited[k] for k in FIELDS if k != 'music_peak_weight'):
        raise ValueError('Only the registered music peak weight may change')
