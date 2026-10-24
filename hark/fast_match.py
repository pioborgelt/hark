"""Research compilation of the SAME absolute-probability transition kernel."""
import math
import numpy as np
from numba import njit
from .ctc_absolute import advance
advance_compiled=njit(cache=False)(advance)

@njit(cache=False,nogil=True)
def _decode(lp,tokens,threshold,min_phone_probability=.003,mean_phone_probability=.1,phone_margin=.1,
           min_blank_probability=.1,max_frames=200,cooldown=50):
    if len(tokens)==0 or np.min(tokens)<=0 or np.max(tokens)>=lp.shape[1]:raise ValueError('invalid tokens')
    if (not 0<min_phone_probability<=1 or not 0<mean_phone_probability<=1 or not 0<min_blank_probability<=1
        or phone_margin<=0 or not math.isfinite(phone_margin)):raise ValueError('invalid absolute evidence parameters')
    labels=np.zeros(len(tokens)*2-1,np.int64);labels[::2]=tokens;n=len(labels)
    cost=np.full(n,-np.inf);starts=np.zeros(n,np.int64);worst=np.full(n,np.inf);peak=np.full(n,-np.inf)
    pm=np.full(n,-np.inf);sums=np.zeros(n);margins=np.zeros(n);last=-100000
    result=np.empty((len(lp),6),np.float64);count=0
    for frame in range(len(lp)):
        cost,starts,worst,peak,pm,sums,margins,last,event,start,score,quality,mean,margin=advance_compiled(
            lp[frame],tokens,labels,cost,starts,worst,peak,pm,sums,margins,frame,last,threshold,math.log(min_phone_probability),
            math.log(mean_phone_probability),phone_margin,math.log(min_blank_probability),max_frames,cooldown)
        if event:
            result[count,0]=frame;result[count,1]=start;result[count,2]=score;result[count,3]=math.exp(quality)
            result[count,4]=math.exp(mean);result[count,5]=margin;count+=1
    return result[:count]


def decode(lp,tokens,threshold,min_phone_probability=.003,mean_phone_probability=.1,phone_margin=.1,
           min_blank_probability=.1,max_frames=200,cooldown=50):
    # Use the production constructor for scalar validation too, avoiding a
    # different acceptance contract around the identical compiled transition.
    from .detect import Matcher
    checked=Matcher(tokens,threshold,max_seconds=max_frames*.02,cooldown=cooldown*.02,
        min_phone_probability=min_phone_probability,mean_phone_probability=mean_phone_probability,
        phone_margin=phone_margin,min_blank_probability=min_blank_probability)
    lp=np.asarray(lp)
    if lp.ndim!=2 or checked.tokens.max()>=lp.shape[1]:raise ValueError('invalid vocabulary/log probability shape')
    if lp.dtype==np.float16:lp=lp.astype(np.float32)
    return _decode(lp,checked.tokens,threshold,min_phone_probability,mean_phone_probability,phone_margin,
                   min_blank_probability,max_frames,cooldown)
