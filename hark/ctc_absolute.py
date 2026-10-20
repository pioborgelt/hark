"""One transition kernel shared by streaming Python and compiled research code.

Path score is a Viterbi log likelihood ratio against the unconstrained
frame-wise best-class background. It is NOT a posterior probability. Per-token absolute peaks are
aggregated geometrically; a separate weak-token floor prevents dilution of an
unsupported prefix/suffix. Mean peak margin compares each phone against the
best OTHER class, including blank, at the same peak frame. Repeated phones need
an evidenced blank. No top-relative distribution is treated as confidence.
"""
import math
import hashlib
from pathlib import Path
import numpy as np
VERSION='ctc-llr-absolute-v5'
KERNEL_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def advance(lp,tokens,labels,costs,starts,worst,peaks,peak_margin,sums,margins,
            frame,last,threshold,min_logprob,mean_logprob,phone_margin,min_blank_logprob,max_frames,cooldown):
    mass=0.;first=-np.inf;second=-np.inf;first_id=-1
    for k in range(len(lp)):
        value=float(lp[k])
        if math.isnan(value) or value>1e-5:raise ValueError('normalized log probabilities required')
        mass+=math.exp(value)
        if value>first:second=first;first=value;first_id=k
        elif value>second:second=value
    if abs(mass-1.)>1e-3:raise ValueError('log probabilities must sum to one')
    n=len(labels);nc=np.full(n,-np.inf);ns=starts.copy();nw=worst.copy();npk=peaks.copy()
    npm=peak_margin.copy();nsm=sums.copy();nmg=margins.copy()
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
            if frame-start>=max_frames:value=-np.inf
            if value>best or (value==best and start>=latest):best=value;latest=start;chosen=move;previous=source
        label=labels[j];nc[j]=best+(float(lp[label])-first)
        if math.isfinite(best):ns[j]=latest
        q=worst[previous];p=peaks[previous];pm=peak_margin[previous];total=sums[previous];margin=margins[previous]
        if chosen==1 or chosen==2:
            if labels[previous]!=0:q=min(q,p);total+=p;margin+=pm
            p=-np.inf;pm=-np.inf
        elif chosen==3:q=np.inf;p=-np.inf;pm=-np.inf;total=0.;margin=0.
        if float(lp[label])>p:
            p=float(lp[label]);competitor=second if label==first_id else first;pm=p-competitor
        nw[j]=q;npk[j]=p;npm[j]=pm;nsm[j]=total;nmg[j]=margin
    score=nc[-1]/len(tokens);quality=min(nw[-1],npk[-1]);mean=(nsm[-1]+npk[-1])/len(tokens)
    margin=(nmg[-1]+npm[-1])/len(tokens);start=ns[-1]
    event=(math.isfinite(score) and score>=threshold and quality>=min_logprob and mean>=mean_logprob
           and margin>=phone_margin and frame-start>=max(2,len(tokens)//2) and frame-last>cooldown)
    if event:
        last=frame;nc[:]=-np.inf;ns[:]=0;nw[:]=np.inf;npk[:]=-np.inf;npm[:]=-np.inf;nsm[:]=0.;nmg[:]=0.
    return nc,ns,nw,npk,npm,nsm,nmg,last,event,start,score,quality,mean,margin
