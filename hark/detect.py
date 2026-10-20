"""CTC background-ratio detector with absolute evidence. Legacy decisions are incompatible."""
import math
import numpy as np
import torch
from .audio import FeatureStream,RATE
from .ctc_absolute import advance,VERSION

class Matcher:
    def __init__(self,tokens,threshold=-3.,max_seconds=4.,cooldown=1.,
                 min_phone_probability=.003,mean_phone_probability=.1,phone_margin=.1,min_blank_probability=.1):
        if len(tokens)==0 or any(t<=0 for t in tokens):raise ValueError('nonblank phrase tokens required')
        if not math.isfinite(threshold) and threshold!=np.inf:raise ValueError('invalid threshold')
        for value in [min_phone_probability,mean_phone_probability,min_blank_probability]:
            if not 0<value<=1:raise ValueError('positive absolute probability required')
        if not math.isfinite(phone_margin) or phone_margin<=0:raise ValueError('positive competition margin required')
        if max_seconds<=0 or cooldown<0:raise ValueError('invalid duration/cooldown')
        self.tokens=np.asarray(tokens,dtype=np.int64);self.labels=np.zeros(len(tokens)*2-1,dtype=np.int64);self.labels[::2]=self.tokens
        self.threshold=threshold;self.min_logprob=math.log(min_phone_probability);self.mean_logprob=math.log(mean_phone_probability)
        self.phone_margin=phone_margin;self.min_blank_logprob=math.log(min_blank_probability)
        self.max_frames=int(max_seconds/.02);self.cooldown=int(cooldown/.02);self.frame=-1;self.last=-100000;self.reset()
    def reset(self):
        n=len(self.labels);self.cost=np.full(n,-np.inf);self.starts=np.zeros(n,dtype=np.int64)
        self.worst=np.full(n,np.inf);self.peak=np.full(n,-np.inf);self.peak_margin=np.full(n,-np.inf)
        self.sums=np.zeros(n);self.margins=np.zeros(n)
    def push(self,log_probs):
        lp=np.asarray(log_probs)
        if lp.ndim!=1 or self.tokens.max()>=len(lp):raise ValueError('invalid vocabulary/log probability shape')
        self.frame+=1
        (self.cost,self.starts,self.worst,self.peak,self.peak_margin,self.sums,self.margins,self.last,
         event,start,score,quality,mean,margin)=advance(lp,self.tokens,self.labels,self.cost,self.starts,self.worst,self.peak,
            self.peak_margin,self.sums,self.margins,self.frame,self.last,self.threshold,self.min_logprob,self.mean_logprob,
            self.phone_margin,self.min_blank_logprob,self.max_frames,self.cooldown)
        if event:return {'time':round(.025+self.frame*.02,4),'start':round(start*.02,4),'score':float(score),
            'minimum_phone_posterior':math.exp(quality),'geometric_mean_phone_posterior':math.exp(mean),
            'mean_peak_competitor_margin':float(margin),'decoder_version':VERSION}
        return None

class Detector:
    def __init__(self,model,tokens,threshold=-3.,min_phone_probability=.003,
                 mean_phone_probability=.1,phone_margin=.1,min_blank_probability=.1):
        self.model=model;self.features=FeatureStream(getattr(model,"bins",40));self.matcher=Matcher(tokens,threshold,
            min_phone_probability=min_phone_probability,mean_phone_probability=mean_phone_probability,
            phone_margin=phone_margin,min_blank_probability=min_blank_probability)
        self.state=None;self.samples=0
    @torch.inference_mode()
    def push(self,audio):
        self.samples+=len(audio);features=self.features.push(audio)
        if not len(features):return []
        lp,self.state=self.model(features[None],self.state);events=[]
        for frame in lp[0].numpy():
            event=self.matcher.push(frame)
            if event:event['available_at']=self.samples/RATE;events.append(event)
        return events
