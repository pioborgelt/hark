import math
import struct
import torch

TEMPERATURE_KEY = '_posterior_temperature_bytes'
TEMPERATURE_CONTRACT = 'posterior-temperature-buffer-v1'

def checkpoint_temperature(ck):
    temperature = float(ck.get('posterior_temperature', 1.))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid checkpoint posterior temperature')
    guard = ck['model'].get(TEMPERATURE_KEY)
    contract = ck.get('inference_contract')
    if guard is not None:
        expected = torch.tensor(list(struct.pack('>d', temperature)), dtype=torch.uint8)
        if (contract != TEMPERATURE_CONTRACT or guard.dtype != torch.uint8
                or guard.shape != (8,) or not torch.equal(guard.cpu(), expected)):
            raise ValueError('Invalid posterior temperature contract')
    elif contract is not None:
        raise ValueError('Missing posterior temperature contract buffer')
    return temperature

def checkpoint_parameters(ck):
    checkpoint_temperature(ck)
    if ck.get('model_kind')=='own-foreground-gru-v1':
        from .foreground_model import validate_checkpoint
        validate_checkpoint(ck)
    return {k: v for k, v in ck['model'].items() if k != TEMPERATURE_KEY}

def guard_temperature_checkpoint(ck):
    """Make older strict readers reject calibrated exports they cannot interpret."""
    temperature = checkpoint_temperature(ck)
    return {**ck, 'inference_contract': TEMPERATURE_CONTRACT,
            'model': {**ck['model'], TEMPERATURE_KEY: torch.tensor(
                list(struct.pack('>d', temperature)), dtype=torch.uint8)}}

class AcousticModel(torch.nn.Module):
    def __init__(self, classes, hidden=128, causal_layers=0, dropout=0., bins=40):
        super().__init__()
        self.hidden = hidden
        self.bins = bins
        self.posterior_temperature = 1.
        self.input = torch.nn.Sequential(torch.nn.Linear(bins, hidden), torch.nn.LayerNorm(hidden), torch.nn.SiLU())
        self.causal_layers=causal_layers
        self.temporal=torch.nn.ModuleList([
            torch.nn.ModuleDict({"depthwise":torch.nn.Conv1d(hidden,hidden,5,dilation=2**i,groups=hidden),
                                 "pointwise":torch.nn.Conv1d(hidden,hidden,1),"norm":torch.nn.LayerNorm(hidden)})
            for i in range(causal_layers)])
        self.dropout=torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
        self.head = torch.nn.Linear(hidden, classes)

    def forward(self, features, state=None):
        x=self.input(features)
        if not self.causal_layers:
            x,state=self.gru(self.dropout(x),state)
            return self.log_probabilities(x),state
        histories=[]
        for i,block in enumerate(self.temporal):
            context=4*2**i
            history=(state["conv"][i] if state is not None else x.new_zeros(x.shape[0],x.shape[2],context))
            joined=torch.cat([history,x.transpose(1,2)],dim=-1)
            residual=block["pointwise"](torch.nn.functional.silu(block["depthwise"](joined))).transpose(1,2)
            x=x+self.dropout(torch.nn.functional.silu(block["norm"](residual)))
            histories.append(joined[:,:,-context:].clone())
        x,recurrent=self.gru(self.dropout(x),None if state is None else state["gru"])
        return self.log_probabilities(x),{"gru":recurrent,"conv":histories}

    def log_probabilities(self,x):
        logits=self.head(self.dropout(x))
        if self.posterior_temperature!=1.:logits=logits/self.posterior_temperature
        return logits.log_softmax(-1)

def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    temperature = checkpoint_temperature(ck)
    if ck.get('model_kind')=='own-foreground-gru-v1':
        from .foreground_model import ForegroundAcousticModel,validate_checkpoint
        foreground_hidden=validate_checkpoint(ck)
        model=ForegroundAcousticModel(len(ck['vocab'])+1,ck['hidden'],ck.get('causal_layers',0),ck.get('dropout',0.),ck.get('bins',40),foreground_hidden).to(device)
    else:
        model = AcousticModel(len(ck["vocab"]) + 1, ck["hidden"],ck.get("causal_layers",0),ck.get("dropout",0.),ck.get("bins",40)).to(device)
    if TEMPERATURE_KEY in ck['model']:
        model.register_buffer(TEMPERATURE_KEY, torch.zeros(8, dtype=torch.uint8, device=device))
    model.load_state_dict(ck["model"])
    model.posterior_temperature=temperature
    model.eval()
    return model, ck
