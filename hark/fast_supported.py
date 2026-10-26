"""Compile exactly the separate V6 production transition, without alteration."""
import math
import numpy as np
from numba import njit
from .ctc_supported import advance
from .supported_detect import Matcher
advance_compiled=njit(cache=False)(advance)

@njit(cache=False,nogil=True)
def _decode(lp,tokens,threshold,min_phone_probability,mean_phone_probability,phone_margin,
            min_blank_probability,min_frame_probability,max_frames,cooldown):
    labels=np.zeros(len(tokens)*2-1,np.int64);labels[::2]=tokens;n=len(labels)
    cost=np.full(n,-np.inf);starts=np.zeros(n,np.int64);worst=np.full(n,np.inf);weak_margin=np.full(n,np.inf)
    peak=np.full(n,-np.inf);pm=np.full(n,-np.inf);sums=np.zeros(n);margins=np.zeros(n);last=-100000
    result=np.empty((len(lp)//max(1,cooldown+1)+2,7),np.float64);count=0
    for frame in range(len(lp)):
        cost,starts,worst,weak_margin,peak,pm,sums,margins,last,event,start,score,quality,mean,margin,minimum_margin=advance_compiled(
            lp[frame],tokens,labels,cost,starts,worst,weak_margin,peak,pm,sums,margins,frame,last,
            threshold,math.log(min_phone_probability),math.log(mean_phone_probability),phone_margin,
            math.log(min_blank_probability),math.log(min_frame_probability),max_frames,cooldown)
        if event:
            result[count,0]=frame;result[count,1]=start;result[count,2]=score;result[count,3]=math.exp(quality)
            result[count,4]=math.exp(mean);result[count,5]=margin;result[count,6]=minimum_margin;count+=1
    return result[:count]

def decode(lp,tokens,threshold,min_phone_probability=.003,mean_phone_probability=.1,phone_margin=.1,
           min_blank_probability=.1,min_frame_probability=.1,max_seconds_per_phone=.16,cooldown=1.):
    checked=Matcher(tokens,threshold,min_phone_probability,mean_phone_probability,phone_margin,
                    min_blank_probability,min_frame_probability,max_seconds_per_phone,cooldown)
    lp=np.asarray(lp)
    if lp.ndim!=2 or checked.tokens.max()>=lp.shape[1]:raise ValueError('invalid vocabulary/log probability shape')
    if lp.dtype==np.float16:lp=lp.astype(np.float32)
    return _decode(lp,checked.tokens,threshold,min_phone_probability,mean_phone_probability,phone_margin,
                   min_blank_probability,min_frame_probability,checked.max_frames,checked.cooldown)
