import pytest
import torch
from hark.model import AcousticModel, guard_temperature_checkpoint
from hark.residual_acoustic_model import ResidualAcousticModel, MODEL_KIND, load_model

def parent():
    torch.manual_seed(632)
    model=AcousticModel(5,32,2,.1,bins=8).eval()
    checkpoint={'model':model.state_dict(),'hidden':32,'causal_layers':2,'dropout':.1,
        'bins':8,'vocab':{c:i+1 for i,c in enumerate('abcd')}}
    return model,checkpoint

def test_zero_head_preserves_parent_outputs_at_any_temperature():
    original,checkpoint=parent()
    model=ResidualAcousticModel(5,32,2,.1,8,16).eval()
    model.initialize_parent(checkpoint)
    x=torch.randn(2,37,8)
    for temperature in [1.,1.25]:
        original.posterior_temperature=model.posterior_temperature=temperature
        with torch.no_grad():a,_=original(x);b,_=model(x)
        assert torch.equal(a,b)

def test_stream_chunks_and_future_audio_preserve_causality():
    _,checkpoint=parent()
    model=ResidualAcousticModel(5,32,2,.1,8,16).eval();model.initialize_parent(checkpoint)
    torch.nn.init.normal_(model.adapter.head.weight,std=.1)
    x=torch.randn(1,43,8)
    with torch.no_grad():
        whole,_=model(x);pieces=[];state=None
        for begin,end in [(0,3),(3,17),(17,18),(18,43)]:
            value,state=model(x[:,begin:end],state);pieces.append(value)
        assert torch.allclose(whole,torch.cat(pieces,1),atol=1e-6,rtol=1e-6)
        changed=x.clone();changed[:,25:]=torch.randn_like(changed[:,25:])*4
        future,_=model(changed)
        assert torch.equal(whole[:,:25],future[:,:25])

def test_both_encoders_train_after_zero_head_initialization():
    _,checkpoint=parent()
    model=ResidualAcousticModel(5,32,2,.1,8,16);model.initialize_parent(checkpoint)
    before={k:v.detach().clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
    x=torch.randn(2,25,8);target=torch.randint(0,5,(2,25))
    for _ in range(3):
        optimizer.zero_grad();lp,_=model(x)
        loss=torch.nn.functional.nll_loss(lp.reshape(-1,5),target.reshape(-1));loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert all(not torch.equal(v,before[k]) for k,v in model.named_parameters())

def test_native_checkpoint_roundtrip_and_old_reader_refusal(tmp_path):
    _,checkpoint=parent()
    model=ResidualAcousticModel(5,32,2,.1,8,16).eval();model.initialize_parent(checkpoint)
    checkpoint.update(model=model.state_dict(),model_kind=MODEL_KIND,adapter_hidden=16,
        posterior_temperature=1.25)
    path=tmp_path/'own.pt';torch.save(guard_temperature_checkpoint(checkpoint),path)
    loaded,_=load_model(path);x=torch.randn(1,23,8);model.posterior_temperature=1.25
    with torch.no_grad():a,_=model(x);b,_=loaded(x)
    assert torch.equal(a,b)
    from hark.model import load_model as original
    with pytest.raises(RuntimeError):original(path)

def test_wrong_parent_architecture_is_rejected():
    _,checkpoint=parent()
    model=ResidualAcousticModel(5,32,2,.1,8,16)
    with pytest.raises(ValueError):model.initialize_parent(checkpoint|{'bins':64})
