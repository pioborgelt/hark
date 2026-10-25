"""Own recurrent foreground confidence; no query-dependent parameters."""
import struct
import torch
from .model import AcousticModel

KIND='own-foreground-gru-v1'
CONTRACT_KEY='_foreground_contract_bytes'

def contract_bytes(hidden):
    return torch.tensor(list(b'FGV1'+struct.pack('>I',hidden)),dtype=torch.uint8)

def validate_checkpoint(ck):
    hidden=ck.get('foreground_hidden')
    if ck.get('model_kind')!=KIND or isinstance(hidden,bool) or not isinstance(hidden,int) or not 1<=hidden<=512:
        raise ValueError('Invalid foreground model contract')
    guard=ck['model'].get(CONTRACT_KEY)
    expected=contract_bytes(hidden)
    if not isinstance(guard,torch.Tensor) or guard.dtype!=torch.uint8 or guard.shape!=(8,) or not torch.equal(guard.cpu(),expected):
        raise ValueError('Invalid foreground model contract buffer')
    return hidden

class ForegroundAcousticModel(AcousticModel):
    def __init__(self,classes,hidden=384,causal_layers=3,dropout=.1,bins=64,foreground_hidden=64):
        super().__init__(classes,hidden,causal_layers,dropout,bins)
        if causal_layers<1:raise ValueError('Foreground model requires causal convolution state')
        self.foreground_hidden=foreground_hidden
        rng=torch.get_rng_state()
        self.foreground_gru=torch.nn.GRU(hidden,foreground_hidden,batch_first=True)
        self.foreground_head=torch.nn.Linear(foreground_hidden,1)
        torch.nn.init.constant_(self.foreground_head.bias,4.)
        torch.set_rng_state(rng)
        self.register_buffer(CONTRACT_KEY,contract_bytes(foreground_hidden))
        self.foreground_logits=None

    def freeze_backbone(self):
        self.requires_grad_(False)
        self.foreground_gru.requires_grad_(True);self.foreground_head.requires_grad_(True)
        self.eval()

    def forward(self,features,state=None):
        x=self.input(features);histories=[]
        for i,block in enumerate(self.temporal):
            context=4*2**i
            history=state['conv'][i] if state is not None else x.new_zeros(x.shape[0],x.shape[2],context)
            joined=torch.cat([history,x.transpose(1,2)],dim=-1)
            residual=block['pointwise'](torch.nn.functional.silu(block['depthwise'](joined))).transpose(1,2)
            x=x+self.dropout(torch.nn.functional.silu(block['norm'](residual)))
            histories.append(joined[:,:,-context:].clone())
        x,recurrent=self.gru(self.dropout(x),None if state is None else state['gru'])
        original=super().log_probabilities(x)
        foreground,foreground_state=self.foreground_gru(x,None if state is None else state['foreground'])
        logits=self.foreground_head(foreground).squeeze(-1);self.foreground_logits=logits
        log_speech=torch.nn.functional.logsigmoid(logits)
        log_background=torch.nn.functional.logsigmoid(-logits)
        blank=torch.logaddexp(log_background,log_speech+original[...,0])
        phones=original[...,1:]+log_speech[...,None]
        lp=torch.cat([blank[...,None],phones],dim=-1)
        return lp,{'gru':recurrent,'conv':histories,'foreground':foreground_state}
