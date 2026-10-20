"""Optional PortAudio input. The callback only copies into a bounded queue."""
import json
import queue
import time
import numpy as np
from scipy.signal import firwin, lfilter
from .audio import RATE

class StreamResampler:
    """Causal anti-alias FIR plus linear rate conversion, independent of blocks.

    Output has a fixed 31/input_rate second FIR delay for non-native rates.
    Keeps FIR history and one interpolation sample, never the full recording.
    """
    def __init__(self, source_rate):
        self.rate = int(source_rate)
        if self.rate < 8000:
            raise ValueError("Input rate must be at least 8000 Hz")
        self.fir = firwin(63, .90*min(self.rate, RATE)/2, fs=self.rate)
        self.zi = np.zeros(62)
        self.input_count = 0
        self.output_count = 0
        self.last = np.empty(0)

    def push(self, audio):
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if not np.isfinite(x).all():
            raise ValueError("non-finite microphone samples")
        if self.rate == RATE:
            return x
        if not len(x):
            return x
        x, self.zi = lfilter(self.fir, [1.], x, zi=self.zi)
        start = self.input_count-len(self.last)
        self.input_count += len(x)
        data = np.concatenate((self.last, x))
        # Only output positions for which the interpolation's right sample exists.
        stop = int(np.floor((self.input_count-1)*RATE/self.rate))+1
        indices = np.arange(self.output_count, stop)
        positions = indices*self.rate/RATE-start
        y = np.interp(positions, np.arange(len(data)), data).astype(np.float32)
        self.output_count = stop
        self.last = data[-1:]
        return y

class CaptureQueue:
    def __init__(self, capacity=32):
        self.blocks = queue.Queue(capacity)
        self.frames = 0
        self.dropped_blocks = 0

    def callback(self, indata, frames, time_info, status):
        start = self.frames
        self.frames += frames
        item = (start, np.array(indata, dtype=np.float32, copy=True), bool(status))
        try:
            self.blocks.put_nowait(item)
        except queue.Full:
            # Do not block PortAudio. Sequence numbers expose the dropped interval.
            self.dropped_blocks += 1

class CaptureProcessor:
    def __init__(self, source_rate, detector_factory):
        self.rate = source_rate
        self.factory = detector_factory
        self.expected = 0
        self.base = 0.
        self.detector = detector_factory()
        self.resampler = StreamResampler(source_rate)
        self.resets = 0

    def consume(self, item):
        start, audio, overflow = item
        gap = start != self.expected or overflow
        self.expected = start+len(audio)
        if gap:
            # Never join phrase fragments across missing input.
            self.detector = self.factory()
            self.resampler = StreamResampler(self.rate)
            self.base = start/self.rate
            self.resets += 1
        events = self.detector.push(self.resampler.push(audio))
        for event in events:
            for name in ('time', 'start', 'available_at'):
                event[name] += self.base
        return gap, events

def audio_backend():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as exc:
        raise RuntimeError("Microphone needs PortAudio and the optional package: pip install -e '.[microphone]' (Linux: libportaudio2)") from exc

def list_devices():
    sd = audio_backend()
    return [{"id": i, "name": d['name'], "inputs": d['max_input_channels'],
             "default_rate": d['default_samplerate']} for i, d in enumerate(sd.query_devices())
            if d['max_input_channels'] > 0]

def listen(model, phrase, device=None, rate=None, chunk_ms=80, seconds=None, emit=print):
    from .detect import Detector
    if not phrase.get('live_validated',False):
        raise RuntimeError('Live detection disabled after music false alarms; no microphone was opened')
    sd = audio_backend()
    device = int(device) if device is not None and str(device).isdigit() else device
    info = sd.query_devices(device, 'input')
    rate = int(rate or info['default_samplerate'])
    sd.check_input_settings(device=device, channels=1, dtype='float32', samplerate=rate)
    capture = CaptureQueue()
    processor = CaptureProcessor(rate, lambda: Detector(model, phrase['tokens'], **phrase.get('decoder_parameters', {'threshold': phrase.get('threshold', -3.)})))
    def output(value):
        emit(json.dumps(value, ensure_ascii=False))
    output({"listening": phrase['text'], "device": info['name'], "input_rate": rate,
            "stop": "Ctrl+C", "fir_delay_seconds": 0 if rate == RATE else 31/rate})
    started = time.monotonic()
    try:
        with sd.InputStream(device=device, samplerate=rate, channels=1, dtype='float32',
                            blocksize=max(1, int(rate*chunk_ms/1000)), callback=capture.callback) as stream:
            while seconds is None or time.monotonic()-started < seconds:
                try:
                    item = capture.blocks.get(timeout=.2)
                except queue.Empty:
                    if not stream.active:
                        raise RuntimeError('Microphone stream stopped unexpectedly')
                    continue
                gap, events = processor.consume(item)
                if gap:
                    output({"warning": "Audio gap/overflow: detector state reset", "input_time": item[0]/rate})
                for event in events:
                    output({"hit": phrase['text'], **event})
    except KeyboardInterrupt:
        pass
    finally:
        output({"stopped": True, "seconds": time.monotonic()-started,
                "dropped_blocks": capture.dropped_blocks, "state_resets": processor.resets})
