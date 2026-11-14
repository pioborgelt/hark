"""Serialization-independent identity of keyed cached feature tensors."""
import hashlib,json

def feature_digest(features):
    digest=hashlib.sha256()
    for key in sorted(features):
        variants=features[key]
        digest.update(json.dumps([key,len(variants)],separators=(',',':')).encode())
        for tensor in variants:
            digest.update(json.dumps([str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
