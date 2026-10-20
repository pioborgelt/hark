"""Probability-normalized counterexamples and positive controls for the incident."""
import numpy as np
import pytest
from hark.detect import Matcher
from hark.fast_match import decode
from hark.legacy_detect import Matcher as LegacyMatcher

TOKENS=[14,10,37,40,15,1,40]
def logrows(ids,confidence=.97,classes=55):
    p=np.full((len(ids),classes),(1-confidence)/(classes-1),dtype=np.float64)
    for i,token in enumerate(ids):p[i,token]=confidence
    return np.log(p)
def events(lp,tokens=TOKENS,threshold=-6.,probability=.003):
    m=Matcher(tokens,threshold,min_phone_probability=probability)
    return [e for f in lp if (e:=m.push(f)) is not None]

def test_incident_uniform_distribution_is_not_perfect_evidence():
    lp=np.full((500,55),-np.log(55),np.float32)
    legacy=LegacyMatcher(TOKENS,-2.108641086987087)
    old=[e for f in lp if (e:=legacy.push(f)) is not None]
    assert len(old)==10 and old[0]['score']==0
    assert not events(lp,threshold=-100)
    assert len(decode(lp,np.array(TOKENS),-100))==0

@pytest.mark.parametrize('epsilon',[0.,.0001,.005])
def test_near_uniform_switching_favourite_phones_is_rejected(epsilon):
    p=np.full((500,55),1/55)
    for i in range(len(p)):p[i,TOKENS[i%len(TOKENS)]]+=epsilon
    p/=p.sum(1,keepdims=True)
    assert not events(np.log(p),threshold=-100)

def test_weak_phone_sequence_is_rejected_without_disabling_strong_positive():
    ids=[0]*8+[v for t in TOKENS for v in [t,t,0]]+[0]*10
    assert not events(logrows(ids,confidence=.04),threshold=-100)
    assert len(events(logrows(ids)))==1

@pytest.mark.parametrize('ids',[TOKENS[1:],TOKENS[:-1],TOKENS[::-1],[2,3,4,5]*6])
def test_missing_prefix_suffix_and_wrong_background_phone_order(ids):
    lp=logrows([0]*4+[v for t in ids for v in [t,0]]+[0]*10)
    assert not events(lp,threshold=-100)

@pytest.mark.parametrize('tokens',[[1,2],[1,1,2],TOKENS])
def test_complete_short_and_repeated_sequences_and_cooldown(tokens):
    ids=[0]*5+[v for t in tokens for v in [t,t,0]]+[0]*55
    lp=logrows(ids*3)
    slow=events(lp,tokens,threshold=-3)
    fast=decode(lp,np.array(tokens),-3)
    assert len(slow)==len(fast)==3
    np.testing.assert_allclose([e['score'] for e in slow],fast[:,2],atol=1e-10)
    assert all(b['time']-a['time']>1 for a,b in zip(slow,slow[1:]))
    if tokens==[1,1,2]:assert not events(logrows([1]*6+[2]*3+[0]*10),tokens,threshold=-3)

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_production_and_fast_share_every_transition_under_normalized_ties(dtype):
    rng=np.random.default_rng(61);raw=rng.normal(0,3,(2500,7));raw-=raw.max(1,keepdims=True)
    lp=(raw-np.log(np.exp(raw).sum(1,keepdims=True))).astype(dtype)
    lp[200:220]=-np.log(7)
    for threshold in [-1.,-3.,-6.]:
        tokens=[1,2,2,3];slow=events(lp,tokens,threshold)
        fast=decode(lp,np.array(tokens),threshold)
        assert len(slow)==len(fast)
        for a,b in zip(slow,fast):
            assert a['time']==pytest.approx(.025+b[0]*.02)
            assert a['start']==pytest.approx(b[1]*.02)
            assert a['score']==pytest.approx(b[2],abs=1e-10)
            assert a['minimum_phone_posterior']==pytest.approx(b[3],abs=1e-10)

@pytest.mark.parametrize('bad',[np.zeros(55),np.full(55,np.nan),np.full(55,-np.inf)])
def test_invalid_logits_cannot_silently_become_probability_evidence(bad):
    with pytest.raises(ValueError):Matcher(TOKENS).push(bad)
    with pytest.raises(ValueError):decode(bad[None],np.array(TOKENS),-3)

@pytest.mark.parametrize('params',[{'threshold':float('nan')},{'phone_margin':0},
                                   {'min_phone_probability':0},{'mean_phone_probability':float('inf')}])
def test_fast_and_streaming_reject_same_invalid_parameters(params):
    values={'threshold':-3.,**params}
    with pytest.raises(ValueError):Matcher(TOKENS,**values)
    with pytest.raises(ValueError):decode(logrows([0,1,0]),np.array(TOKENS),**values)
