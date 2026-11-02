"""Additional own causal acoustic capacity with an exact parent initialization.

Both encoders remain trainable. The extra encoder's initially zero output head
preserves the parent's logits; it is not an external model or a frozen teacher.
"""
import torch
from .model import AcousticModel, checkpoint_parameters, checkpoint_temperature, TEMPERATURE_KEY

MODEL_KIND='own-residual-acoustic-v1'

class LogitEncoder(AcousticModel):
    def log_probabilities(self, x):
        return self.head(self.dropout(x))

class ResidualAcousticModel(LogitEncoder):
    def __init__(self, classes, hidden=384, causal_layers=3, dropout=.1, bins=64, adapter_hidden=256):
        super().__init__(classes,hidden,causal_layers,dropout,bins)
        if not 16<=adapter_hidden<=1024:
            raise ValueError('Invalid additional encoder width')
        self.adapter_hidden=adapter_hidden
        self.adapter=LogitEncoder(classes,adapter_hidden,causal_layers,dropout,bins)
        torch.nn.init.zeros_(self.adapter.head.weight)
        torch.nn.init.zeros_(self.adapter.head.bias)

    def forward(self, features, state=None):
        base,base_state=super().forward(features,None if state is None else state['base'])
        residual,residual_state=self.adapter(features,None if state is None else state['adapter'])
        logits=(base+residual)/self.posterior_temperature
        return logits.log_softmax(-1),{'base':base_state,'adapter':residual_state}

    def initialize_parent(self, checkpoint):
        if checkpoint.get('model_kind') or checkpoint['hidden']!=self.hidden:
            raise ValueError('Need a matching original own encoder parent')
        if checkpoint.get('bins',40)!=self.bins or checkpoint.get('causal_layers',0)!=self.causal_layers:
            raise ValueError('Parent frontend/causal architecture mismatch')
        state=checkpoint_parameters(checkpoint)
        if any(k.startswith('adapter.') for k in state):
            raise ValueError('Unexpected parent additional encoder')
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or set(missing)!={k for k in self.state_dict() if k.startswith('adapter.')}:
            raise ValueError('Incomplete own parent initialization')

def load_model(path,device='cpu'):
    from .model import load_model as original
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get('model_kind')!=MODEL_KIND:
        return original(path,device)
    model=ResidualAcousticModel(len(checkpoint['vocab'])+1,checkpoint['hidden'],
        checkpoint.get('causal_layers',0),checkpoint.get('dropout',0.),checkpoint.get('bins',40),
        checkpoint['adapter_hidden']).to(device)
    if TEMPERATURE_KEY in checkpoint['model']:
        model.register_buffer(TEMPERATURE_KEY,torch.zeros(8,dtype=torch.uint8,device=device))
    model.load_state_dict(checkpoint['model'])
    model.posterior_temperature=checkpoint_temperature(checkpoint)
    model.eval()
    return model,checkpoint
