"""Canonical CTC-label timing, extending exact annotated matches conservatively.

CommonPhone annotations describe the realized pronunciation; enrolled canonical
labels can differ, e.g. a vocalized rhotic. A one-to-one substitution supplies
the observed phone's time span to that canonical target. It does not assert
that the canonical IPA symbol was the realized phone. Insertions and combining
marks receive no additional inferred supervision.
"""
import unicodedata
import numpy as np
import torch

def frame_targets(record, vocab, frames):
    target = record['phones']
    source, spans, silences = [], [], []
    for label in record.get('labels', []):
        start,end = label['start'],label['end']
        if label['symbol']=='(...)':
            silences.append((start,end));continue
        chars = [c for c in unicodedata.normalize('NFD',label['symbol'])
                 if c not in ' ˈˌː' and not c.isspace()]
        for i,c in enumerate(chars):
            source.append(c)
            spans.append((start+(end-start)*i/len(chars),start+(end-start)*(i+1)/len(chars)))
    out = np.full(frames,-100,dtype=np.int64)
    if not source:
        return torch.from_numpy(out)
    n,m = len(source),len(target)
    dp = np.zeros((n+1,m+1),dtype=np.int16)
    dp[:,0]=np.arange(n+1);dp[0]=np.arange(m+1)
    for i,s in enumerate(source,1):
        for j,t in enumerate(target,1):
            dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+(s!=t))
    if dp[n,m]/max(n,m)>.5:
        return torch.from_numpy(out)
    pairs=[];i,j=n,m
    while i and j:
        if dp[i,j]==dp[i-1,j-1]+(source[i-1]!=target[j-1]):
            if source[i-1]==target[j-1] or (not unicodedata.combining(source[i-1])
                                           and not unicodedata.combining(target[j-1])):
                pairs.append((i-1,j-1))
            i-=1;j-=1
        elif dp[i,j]==dp[i-1,j]+1:
            i-=1
        else:
            j-=1
    centers=.0125+np.arange(frames)*.02
    for start,end in silences:
        out[(centers>=start)&(centers<end)]=0
    for i,j in pairs:
        if target[j] in vocab:
            start,end=spans[i]
            out[(centers>=start)&(centers<end)]=vocab[target[j]]
    return torch.from_numpy(out)
