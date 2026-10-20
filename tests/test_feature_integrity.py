import hashlib
import torch
from scripts.feature_integrity import feature_digest


def test_control_feature_identity_ignores_serialization_but_detects_wrong_values(tmp_path):
    features={'b':[torch.arange(8,dtype=torch.float16).reshape(2,4)],'a':[torch.ones(3,4,dtype=torch.float16)]}
    p=tmp_path/'first.pt';q=tmp_path/'different-root-name.pt'
    torch.save({'features':features,'manifest_hash':'one'},p)
    torch.save({'features':dict(reversed(list(features.items()))),'manifest_hash':'two'},q)
    assert hashlib.sha256(p.read_bytes()).hexdigest()!=hashlib.sha256(q.read_bytes()).hexdigest()
    a=torch.load(p,weights_only=True)['features'];b=torch.load(q,weights_only=True)['features']
    assert feature_digest(a)==feature_digest(b)
    b['b'][0][0,0]=42
    assert feature_digest(a)!=feature_digest(b)
