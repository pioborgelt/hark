"""Unsupported-phone counterexamples, duration bounds and exact stream parity."""
import numpy as np
import pytest
from hark.fast_supported import decode
from hark.fast_match import decode as old_decode
from hark.supported_detect import Matcher

def logrows(ids,classes=56,confidence=.98):
    p=np.full((len(ids),classes),(1-confidence)/(classes-1))
    for i,phone in enumerate(ids):
        if phone is None:p[i]=1/classes
        else:p[i,phone]=confidence
    return np.log(p)

def stream(lp,tokens,**settings):
    matcher=Matcher(tokens,**settings)
    return [event for row in lp if (event:=matcher.push(row)) is not None]

@pytest.mark.parametrize('missing',[0,2,5])
def test_one_uniform_phone_cannot_hide_in_an_otherwise_strong_word(missing):
    tokens=list(range(1,7));ids=[]
    for i,phone in enumerate(tokens):ids.extend([None if i==missing else phone]*4)
    lp=logrows(ids)
    if missing in [0,5]:assert len(old_decode(lp,tokens,-.75,mean_phone_probability=.2))>=1
    assert not len(decode(lp,tokens,-.75,mean_phone_probability=.2))
    assert not stream(lp,tokens,threshold=-.75,mean_phone_probability=.2)
    assert len(decode(logrows([p for phone in tokens for p in [phone]*4]),tokens,-.75))==1

@pytest.mark.parametrize('tokens',[[1,2],[1,1,2],[14,10,37,40,15,1,40]])
def test_complete_and_repeated_phrases_keep_blank_evidence_and_cooldown(tokens):
    ids=[0]*5+[p for phone in tokens for p in [phone,phone,0]]+[0]*55
    lp=logrows(ids*3);slow=stream(lp,tokens,threshold=-3);fast=decode(lp,tokens,-3)
    assert len(slow)==len(fast)==3
    assert all(e['minimum_peak_competitor_margin']>=.05 for e in slow)
    if tokens==[1,1,2]:assert not len(decode(logrows([1]*6+[2]*3+[0]*10),tokens,-3))

def test_generic_duration_bound_rejects_sparse_phone_bridges():
    lp=logrows([0]*5+[1]*3+[0]*60+[2]*3+[0]*5)
    assert len(old_decode(lp,[1,2],-.75))==1
    assert not len(decode(lp,[1,2],-.75))
    assert len(decode(lp,[1,2],-.75,max_seconds_per_phone=2.))==1
    assert Matcher([1,2],max_seconds_per_phone=.12).max_frames==25
    assert Matcher([1]*20,max_seconds_per_phone=.12).max_frames==120
    assert Matcher([1]*100,max_seconds_per_phone=.16).max_frames==200

def test_peak_can_qualify_at_lower_phone_probability_on_a_confident_frame():
    p=np.full((4,56),.44/54);p[0,0]=.5;p[0,1]=.06
    p[1]=.92/55;p[1,1]=.08
    p[2:]=.02/55;p[2:,2]=.98
    events=decode(np.log(p),[1,2],-2.,min_frame_probability=.1)
    assert len(events)==1 and events[0,3]==pytest.approx(.06)
    assert events[0,6]==pytest.approx(np.log(.06/.5))

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_fast_and_streaming_use_identical_supported_transitions(dtype):
    rng=np.random.default_rng(61);raw=rng.normal(0,3,(1800,7));lp=(raw-np.logaddexp.reduce(raw,axis=-1)[:,None]).astype(dtype)
    lp[200:220]=-np.log(7);tokens=[1,2,2,3]
    for threshold in [-1.,-3.]:
        for duration in [.12,.16]:
            settings={'threshold':threshold,'max_seconds_per_phone':duration}
            slow=stream(lp,tokens,**settings);fast=decode(lp,tokens,**settings)
            assert len(slow)==len(fast)
            for a,b in zip(slow,fast):
                assert a['time']==pytest.approx(.025+b[0]*.02)
                assert a['start']==pytest.approx(b[1]*.02)
                values=[a[k] for k in ['score','minimum_phone_posterior','geometric_mean_phone_posterior','mean_peak_competitor_margin','minimum_peak_competitor_margin']]
                np.testing.assert_allclose(values,b[2:],atol=1e-10,rtol=1e-10)

@pytest.mark.parametrize('epsilon',[0.,.0001,.005])
def test_uniform_and_weak_sequences_still_cannot_trigger(epsilon):
    p=np.full((500,56),1/56)
    for i in range(len(p)):p[i,1+i%6]+=epsilon
    p/=p.sum(1,keepdims=True)
    assert not len(decode(np.log(p),list(range(1,7)),-100))
    assert not len(decode(logrows([p for phone in range(1,7) for p in [phone]*4],confidence=.04),list(range(1,7)),-100))

@pytest.mark.parametrize('bad',[np.zeros(56),np.full(56,np.nan),np.full(56,-np.inf)])
def test_non_normalized_inputs_are_rejected(bad):
    with pytest.raises(ValueError):Matcher([1,2]).push(bad)
    with pytest.raises(ValueError):decode(bad[None],[1,2],-1.)

@pytest.mark.parametrize('settings',[{'min_frame_probability':0},{'max_seconds_per_phone':float('inf')},
                                   {'max_seconds_per_phone':1e308},{'threshold':float('nan')},{'min_phone_probability':0}])
def test_invalid_parameters_rejected_in_both_paths(settings):
    with pytest.raises(ValueError):Matcher([1,2],**settings)
    with pytest.raises(ValueError):decode(logrows([1,2]),[1,2],**({'threshold':-1.,**settings}))

@pytest.mark.parametrize('tokens',[[1.5,2],['a',2],[True,False],[2**80,1],[float(2**63),1],np.array([2**63,1],dtype=np.uint64)])
def test_invalid_or_overflowing_token_ids_cannot_reach_compiled_indexing(tokens):
    with pytest.raises(ValueError):Matcher(tokens)
    with pytest.raises(ValueError):decode(logrows([1,2]),tokens,-1.)

@pytest.mark.parametrize('classes',[3,7,56])
def test_uniform_prefix_is_rejected_even_with_small_user_frame_floor(classes):
    lp=logrows([None]*6+[2]*6,classes=classes)
    assert not len(decode(lp,[1,2],-100,min_frame_probability=.001))
