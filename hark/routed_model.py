"""One own checkpoint: unchanged shared base for DE, own correction for EN.

The language comes from enrollment, never from a wakeword-specific rule or
audio language identification. Dual output is only for efficient evaluation.
"""
import torch
from .model import AcousticModel,checkpoint_temperature,checkpoint_parameters
from .residual_model import (ResidualAcousticModel,ResidualBranch,contract_bytes,
    parameter_digest,CONTRACT_KEY,load_model as load_previous)
KIND='own-language-routed-residual-v1'
ROUTE_KEY='_language_route_contract_bytes'
POLICY={'de':{'posterior_branch':'base','decoder_family':'v5'},
        'en':{'posterior_branch':'residual','decoder_family':'v7'}}

def route_bytes():return torch.tensor(list(b'LRCTC1:de=base/v5;en=residual/v7'),dtype=torch.uint8)

class RawLogitBase(AcousticModel):
    def log_probabilities(self,hidden):return self.head(self.dropout(hidden))

class DualLanguageModel(torch.nn.Module):
    def __init__(self,ck):
        super().__init__();classes=len(ck['vocab'])+1;self.classes=classes;self.bins=ck['bins'];self.hidden=ck['hidden']
        self.posterior_temperature=checkpoint_temperature(ck)
        self.base=RawLogitBase(classes,ck['hidden'],ck['causal_layers'],ck['dropout'],ck['bins']);self.base.requires_grad_(False)
        self.residual=ResidualBranch(classes,ck['residual_hidden'],ck['residual_layers'],0.,ck['bins'])
        self.register_buffer(CONTRACT_KEY,contract_bytes(classes,ck['hidden'],ck['causal_layers'],ck['bins'],ck['residual_hidden'],ck['residual_layers']))
        self.register_buffer(ROUTE_KEY,route_bytes())
    def forward(self,features,state=None):
        with torch.no_grad():logits,base_state=self.base(features,None if state is None else state['base'])
        delta,residual_state=self.residual(features,None if state is None else state['residual'])
        base=logits.log_softmax(-1)
        de=(logits/self.posterior_temperature).log_softmax(-1)
        en=torch.log_softmax((base+delta)/self.posterior_temperature,-1)
        return torch.cat((de,en),-1),{'base':base_state,'residual':residual_state}

def validate_checkpoint(ck):
    if ck.get('model_kind')!=KIND or ck.get('language_route')!=POLICY:raise ValueError('Language route contract changed')
    if checkpoint_temperature(ck)!=1.25:raise ValueError('Routed v1 binds the preselected T1.25')
    actual=ck['model'].get(ROUTE_KEY)
    if actual is None or actual.dtype!=torch.uint8 or not torch.equal(actual.cpu(),route_bytes()):raise ValueError('Missing routed model buffer')
    expected=contract_bytes(len(ck['vocab'])+1,ck['hidden'],ck['causal_layers'],ck['bins'],ck['residual_hidden'],ck['residual_layers'])
    actual=ck['model'].get(CONTRACT_KEY)
    if actual is None or actual.dtype!=torch.uint8 or not torch.equal(actual.cpu(),expected):raise ValueError('Residual shape contract changed')
    base={k[5:]:v for k,v in checkpoint_parameters(ck).items() if k.startswith('base.')}
    if parameter_digest(base)!=ck['frozen_base_parameter_sha256'] or ck['routed_base_parameter_sha256']!=ck['frozen_base_parameter_sha256']:raise ValueError('Routed base identity changed')

def load_model(path,device='cpu',*,language='dual'):
    if language not in ['dual','de','en']:raise ValueError('Explicit de/en enrollment language required')
    ck=torch.load(path,map_location=device,weights_only=True)
    if ck.get('model_kind')!=KIND:
        if language!='dual':raise ValueError('A routed checkpoint is required')
        return load_previous(path,device)
    validate_checkpoint(ck);state=checkpoint_parameters(ck);classes=len(ck['vocab'])+1
    if language=='dual':
        model=DualLanguageModel(ck).to(device);model.load_state_dict(state)
    elif language=='de':
        model=AcousticModel(classes,ck['hidden'],ck['causal_layers'],ck['dropout'],ck['bins']).to(device)
        model.load_state_dict({k[5:]:v for k,v in state.items() if k.startswith('base.')});model.posterior_temperature=1.25
    else:
        model=ResidualAcousticModel(classes,ck['hidden'],ck['causal_layers'],ck['dropout'],ck['bins'],ck['residual_hidden'],ck['residual_layers']).to(device)
        model.load_state_dict({k:v for k,v in state.items() if k!=ROUTE_KEY});model.posterior_temperature=1.25
    model.eval();model.routed_language=language;return model,ck
