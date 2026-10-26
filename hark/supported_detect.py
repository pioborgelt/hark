"""Explicit V6 offline matcher; no changes to V5 or microphone admission."""
import hashlib,math
from pathlib import Path
import numpy as np
from .ctc_supported import advance,VERSION,KERNEL_SHA256
from .detect import Detector as OriginalDetector
from .audio import FeatureStream

def source_sha():
    paths=[Path(__file__).with_name(n) for n in ['ctc_supported.py','supported_detect.py','fast_supported.py',
           'supported_contracts.py','supported_cli.py','detect.py','audio.py','model.py','foreground_model.py','text.py','contracts.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()

class Matcher:
    def __init__(self,tokens,threshold=-1.,min_phone_probability=.003,mean_phone_probability=.1,
                 phone_margin=.1,min_blank_probability=.1,min_frame_probability=.1,max_seconds_per_phone=.16,
                 cooldown=1.):
        raw=np.asarray(tokens)
        if raw.ndim!=1 or not len(raw) or raw.dtype.kind not in 'iuf':
            raise ValueError('nonblank integer phrase tokens required')
        if (not np.isfinite(raw).all() or np.any(raw!=np.floor(raw)) or np.min(raw)<=0
                or (raw.dtype.kind=='f' and np.any(raw>=float(2**63)))
                or (raw.dtype.kind=='u' and any(int(t)>2**63-1 for t in raw))):
            raise ValueError('nonblank integer phrase tokens required')
        if not math.isfinite(threshold) and threshold!=np.inf:raise ValueError('invalid threshold')
        for value in [min_phone_probability,mean_phone_probability,min_blank_probability,min_frame_probability]:
            if not 0<value<=1:raise ValueError('positive absolute probability required')
        for value in [phone_margin,max_seconds_per_phone]:
            if not math.isfinite(value) or value<=0:raise ValueError('positive finite evidence/duration required')
        if max_seconds_per_phone>4:raise ValueError('duration scale exceeds hard four-second limit')
        if not math.isfinite(cooldown) or cooldown<0:raise ValueError('invalid cooldown')
        self.tokens=raw.astype(np.int64);self.labels=np.zeros(len(raw)*2-1,dtype=np.int64);self.labels[::2]=self.tokens
        self.threshold=threshold;self.min_logprob=math.log(min_phone_probability);self.mean_logprob=math.log(mean_phone_probability)
        self.phone_margin=phone_margin;self.min_blank_logprob=math.log(min_blank_probability);self.min_frame_logprob=math.log(min_frame_probability)
        self.max_frames=min(200,max(25,math.ceil(max_seconds_per_phone*len(raw)/.02)))
        self.cooldown=int(cooldown/.02);self.frame=-1;self.last=-100000;self.reset()
    def reset(self):
        n=len(self.labels);self.cost=np.full(n,-np.inf);self.starts=np.zeros(n,dtype=np.int64)
        self.worst=np.full(n,np.inf);self.weak_margin=np.full(n,np.inf)
        self.peak=np.full(n,-np.inf);self.peak_margin=np.full(n,-np.inf);self.sums=np.zeros(n);self.margins=np.zeros(n)
    def push(self,log_probs):
        lp=np.asarray(log_probs)
        if lp.ndim!=1 or self.tokens.max()>=len(lp):raise ValueError('invalid vocabulary/log probability shape')
        self.frame+=1
        (self.cost,self.starts,self.worst,self.weak_margin,self.peak,self.peak_margin,self.sums,self.margins,self.last,
         event,start,score,quality,mean,margin,token_margin)=advance(lp,self.tokens,self.labels,self.cost,self.starts,
            self.worst,self.weak_margin,self.peak,self.peak_margin,self.sums,self.margins,self.frame,self.last,
            self.threshold,self.min_logprob,self.mean_logprob,self.phone_margin,self.min_blank_logprob,
            self.min_frame_logprob,self.max_frames,self.cooldown)
        if event:return {'time':round(.025+self.frame*.02,4),'start':round(start*.02,4),'score':float(score),
            'minimum_phone_posterior':math.exp(quality),'geometric_mean_phone_posterior':math.exp(mean),
            'mean_peak_competitor_margin':float(margin),'minimum_peak_competitor_margin':float(token_margin),'decoder_version':VERSION}
        return None

class Detector(OriginalDetector):
    def __init__(self,model,tokens,**parameters):
        self.model=model;self.features=FeatureStream(getattr(model,'bins',40));self.matcher=Matcher(tokens,**parameters)
        self.state=None;self.samples=0
