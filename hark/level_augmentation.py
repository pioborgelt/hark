"""Physical attenuation and the original mel floor, without feature copies."""
import math

def attenuate_power_log(log_power,gain):
    if isinstance(gain,bool) or not math.isfinite(gain) or not 0<gain<=1:
        raise ValueError('Amplitude attenuation must be finite and in(0,1]')
    # max(g² max(P,epsilon),epsilon)==max(g²P,epsilon) for g<=1.
    # Existing fp16 cache quantization remains; this does not recover lost bins.
    return (log_power+2*math.log(gain)).clamp_min(math.log(1e-6))
