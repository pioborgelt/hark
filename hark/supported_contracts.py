"""Separate schema4 identity for the explicit V6 offline research path."""
import hashlib,math,json
from pathlib import Path
from .contracts import sha,identity,seal
from .ctc_supported import VERSION,KERNEL_SHA256
from .supported_detect import source_sha,Matcher
from .text import LEXICON_SHA256,PRONUNCIATION_VERSIONS
SCHEMA=4
PARAMETERS={'threshold','min_phone_probability','mean_phone_probability','phone_margin',
            'min_blank_probability','min_frame_probability','max_seconds_per_phone'}

def check_settings(data,checkpoint):
    if data.get('schema_version')!=SCHEMA or data.get('decoder_version')!=VERSION or data.get('kernel_sha256')!=KERNEL_SHA256:
        raise ValueError('V6 requires its separate schema4 calibration/enrollment')
    if data.get('decoder_source_sha256')!=source_sha():raise ValueError('V6 inference implementation changed')
    if data.get('checkpoint_sha256')!=sha(checkpoint):raise ValueError('Model weights changed')
    pronunciation=data.get('pronunciation_version')
    if pronunciation not in PRONUNCIATION_VERSIONS:raise ValueError('Unknown pronunciation version')
    if pronunciation=='de-lexicon-v1' and data.get('pronunciation_lexicon_sha256')!=LEXICON_SHA256:raise ValueError('Pronunciation lexicon changed')
    if data.get('live_validated') is not False:raise ValueError('V6 has offline research validation only')

def validate_parameters(parameters,tokens):
    if set(parameters)!=PARAMETERS:raise ValueError('Incomplete versioned V6 settings')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in parameters.values()):raise ValueError('Nonfinite or nonnumeric V6 setting')
    Matcher(tokens,**parameters)

def validate(phrase,checkpoint,vocab):
    check_settings(phrase,checkpoint)
    if phrase.get('enrollment_sha256')!=identity({k:v for k,v in phrase.items() if k!='enrollment_sha256'}):raise ValueError('Enrollment identity changed')
    if phrase.get('vocab_sha256')!=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest():raise ValueError('Vocabulary changed')
    validate_parameters(phrase.get('decoder_parameters',{}),phrase.get('tokens',[]))
    if not set(phrase['tokens'])<=set(vocab.values()):raise ValueError('Unknown vocabulary token')
    path=phrase.get('calibration_path')
    if path and Path(path).exists() and sha(path)!=phrase.get('calibration_sha256'):raise ValueError('Calibration changed after enrollment')
