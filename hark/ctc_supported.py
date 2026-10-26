"""V6 research kernel: phone peaks require acoustically confident frames.

Phone states without eligible absolute peaks cannot propagate. This
prevents an exactly uniform prefix being diluted by a strong suffix. Existing
V5 remains unchanged; V6 uses a distinct calibration/enrollment contract.
"""
import math,hashlib
from pathlib import Path
import numpy as np
VERSION='ctc-llr-confident-tempo-v6'
KERNEL_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def advance(lp,tokens,labels,costs,starts,worst,weak_margin,peaks,peak_margin,sums,margins,
            frame,last,threshold,min_logprob,mean_logprob,phone_margin,min_blank_logprob,
            min_frame_logprob,max_frames,cooldown):
    mass=0.;first=-np.inf;second=-np.inf;first_id=-1
    for k in range(len(lp)):
        value=float(lp[k])
        if math.isnan(value) or value>1e-5:raise ValueError('normalized log probabilities required')
        mass+=math.exp(value)
        if value>first:second=first;first=value;first_id=k
        elif value>second:second=value
    if abs(mass-1.)>1e-3:raise ValueError('log probabilities must sum to one')
    frame_confident=first>=max(min_frame_logprob,math.log(1.1/len(lp)))
    n=len(labels);nc=np.full(n,-np.inf);ns=starts.copy();nw=worst.copy();nwm=weak_margin.copy()
    npk=peaks.copy();npm=peak_margin.copy();nsm=sums.copy();nmg=margins.copy()
    for j in range(n):
        best=-np.inf;latest=-1;chosen=0;previous=j
        for move in range(4):
            value=-np.inf;start=0;source=j
            if move==0:value=costs[j];start=starts[j]
            elif move==1 and j>=1:
                source=j-1;value=costs[source];start=starts[source]
                if j>=2 and labels[j]!=0 and labels[j]==labels[j-2] and peaks[source]<min_blank_logprob:
                    value=-np.inf
            elif move==2 and j>=2 and labels[j]!=0 and labels[j]!=labels[j-2]:
                source=j-2;value=costs[source];start=starts[source]
            elif move==3 and j==0:value=0.;start=frame
            if (move==1 or move==2) and labels[source]!=0:
                if peaks[source]<min_logprob:value=-np.inf
            # An unsupported fresh phone must not overwrite an older valid
            # state merely because its pre-emission restart cost is zero.
            if move!=0 and labels[j]!=0:
                if not frame_confident or float(lp[labels[j]])<min_logprob:value=-np.inf
            if frame-start>=max_frames:value=-np.inf
            if value>best or (value==best and start>=latest):best=value;latest=start;chosen=move;previous=source
        label=labels[j];nc[j]=best+(float(lp[label])-first)
        if math.isfinite(best):ns[j]=latest
        q=worst[previous];qm=weak_margin[previous];p=peaks[previous];pm=peak_margin[previous]
        total=sums[previous];margin=margins[previous]
        if chosen==1 or chosen==2:
            if labels[previous]!=0:q=min(q,p);qm=min(qm,pm);total+=p;margin+=pm
            p=-np.inf;pm=-np.inf
        elif chosen==3:q=np.inf;qm=np.inf;p=-np.inf;pm=-np.inf;total=0.;margin=0.
        value=float(lp[label]);competitor=second if label==first_id else first;current_margin=value-competitor
        # A phone peak needs absolute evidence on a non-flat acoustic frame.
        # Blank peaks retain the V5 absolute repeated-phone evidence rule.
        if value>p and (label==0 or (frame_confident and value>=min_logprob)):p=value;pm=current_margin
        nw[j]=q;nwm[j]=qm;npk[j]=p;npm[j]=pm;nsm[j]=total;nmg[j]=margin
    score=nc[-1]/len(tokens);quality=min(nw[-1],npk[-1]);mean=(nsm[-1]+npk[-1])/len(tokens)
    margin=(nmg[-1]+npm[-1])/len(tokens);token_margin=min(nwm[-1],npm[-1]);start=ns[-1]
    event=(math.isfinite(score) and score>=threshold and quality>=min_logprob and mean>=mean_logprob
           and margin>=phone_margin
           and frame-start>=max(2,len(tokens)//2) and frame-last>cooldown)
    if event:
        last=frame;nc[:]=-np.inf;ns[:]=0;nw[:]=np.inf;nwm[:]=np.inf
        npk[:]=-np.inf;npm[:]=-np.inf;nsm[:]=0.;nmg[:]=0.
    return nc,ns,nw,nwm,npk,npm,nsm,nmg,last,event,start,score,quality,mean,margin,token_margin
