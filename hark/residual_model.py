"""Own frozen phonetic base plus a zero-head raw-mel residual correction.

No external parameters. This is a separate checkpoint kind; older strict
readers reject its prefixed state and contract buffer rather than discarding
the correction. The ordinary loader and all historical models are unchanged.
"""
import hashlib,struct
from pathlib import Path
import torch
from .model import AcousticModel,checkpoint_parameters,checkpoint_temperature,load_model as load_original
KIND='own-residual-mel-ctc-v1'
CONTRACT_KEY='_residual_mel_contract_bytes'

def contract_bytes(classes,hidden,causal_layers,bins,residual_hidden,residual_layers):
    return torch.tensor(list(b'RMCTC1'+struct.pack('>6H',classes,hidden,causal_layers,bins,residual_hidden,residual_layers)),dtype=torch.uint8)

def parameter_digest(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(key.encode()+b'\0'+str(value.dtype).encode()+str(tuple(value.shape)).encode()+b'\0')
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

class ResidualBranch(AcousticModel):
    def log_probabilities(self,hidden):
        # Add unconstrained class corrections before the final normalization.
        return self.head(hidden)

class ResidualAcousticModel(torch.nn.Module):
    def __init__(self,classes,hidden=384,causal_layers=3,dropout=.1,bins=64,residual_hidden=128,residual_layers=2):
        super().__init__();self.bins=bins;self.hidden=hidden;self.posterior_temperature=1.
        self.base=AcousticModel(classes,hidden,causal_layers,dropout,bins)
        self.base.requires_grad_(False);self.base.eval()
        self.residual=ResidualBranch(classes,residual_hidden,residual_layers,0.,bins)
        torch.nn.init.zeros_(self.residual.head.weight);torch.nn.init.zeros_(self.residual.head.bias)
        self.register_buffer(CONTRACT_KEY,contract_bytes(classes,hidden,causal_layers,bins,residual_hidden,residual_layers))
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def forward(self,features,state=None):
        base_state=None if state is None else state['base']
        residual_state=None if state is None else state['residual']
        with torch.no_grad():base,base_state=self.base(features,base_state)
        delta,residual_state=self.residual(features,residual_state)
        logits=(base+delta)/self.posterior_temperature
        return torch.log_softmax(logits,-1),{'base':base_state,'residual':residual_state}

def make_model(parent,device='cpu',residual_hidden=128,residual_layers=2):
    if checkpoint_temperature(parent)!=1. or parent.get('model_kind') is not None:raise ValueError('Own uncalibrated ordinary parent required')
    model=ResidualAcousticModel(len(parent['vocab'])+1,parent['hidden'],parent['causal_layers'],parent['dropout'],parent['bins'],residual_hidden,residual_layers)
    model.base.load_state_dict(checkpoint_parameters(parent));model.to(device);return model

def assert_frozen(model,parent):
    state=model.base.state_dict()
    if any(not torch.equal(state[k].cpu(),v.cpu()) for k,v in checkpoint_parameters(parent).items()):raise RuntimeError('Frozen own phonetic base changed')
    if any(p.requires_grad or p.grad is not None for p in model.base.parameters()):raise RuntimeError('Base entered gradient computation')

def load_model(path,device='cpu'):
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck.get('model_kind')!=KIND:return load_original(path,device)
    temperature=checkpoint_temperature(ck)
    expected=contract_bytes(len(ck['vocab'])+1,ck['hidden'],ck['causal_layers'],ck['bins'],ck['residual_hidden'],ck['residual_layers'])
    actual=ck['model'].get(CONTRACT_KEY)
    if actual is None or actual.dtype!=torch.uint8 or not torch.equal(actual.cpu(),expected):raise ValueError('Residual model contract changed')
    model=ResidualAcousticModel(len(ck['vocab'])+1,ck['hidden'],ck['causal_layers'],ck['dropout'],ck['bins'],ck['residual_hidden'],ck['residual_layers']).to(device)
    model.load_state_dict(checkpoint_parameters(ck));model.posterior_temperature=temperature;model.eval()
    if parameter_digest(model.base.state_dict())!=ck['frozen_base_parameter_sha256']:raise ValueError('Frozen base identity changed')
    return model,ck
