"""Version and bind model, score, settings, calibration and enrollment identity."""
import hashlib,json,math
from pathlib import Path
from .ctc_absolute import VERSION,KERNEL_SHA256
from .text import PRONUNCIATION_VERSIONS,LEXICON_SHA256
SCHEMA=3
PARAMETERS={'threshold','min_phone_probability','mean_phone_probability','phone_margin','min_blank_probability'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(data):return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seal(phrase):
    phrase['enrollment_sha256']=identity({k:v for k,v in phrase.items() if k!='enrollment_sha256'})
    return phrase

def validate(phrase,checkpoint,vocab):
    if phrase.get('schema_version')!=SCHEMA or phrase.get('decoder_version')!=VERSION or phrase.get('kernel_sha256')!=KERNEL_SHA256:
        raise ValueError('Retired relative-score enrollment: enroll again with ctc-llr-absolute-v5 calibration; old thresholds cannot be reused')
    if phrase.get('checkpoint_sha256')!=sha(checkpoint):raise ValueError('Model weights changed; enroll again')
    pronunciation=phrase.get('pronunciation_version','espeak-v1')
    if pronunciation not in PRONUNCIATION_VERSIONS:raise ValueError('Unknown pronunciation version')
    if pronunciation=='de-lexicon-v1' and phrase.get('pronunciation_lexicon_sha256')!=LEXICON_SHA256:raise ValueError('Pronunciation lexicon changed; enroll again')
    digest=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest()
    if phrase.get('vocab_sha256')!=digest:raise ValueError('different model vocabulary')
    if phrase.get('enrollment_sha256')!=identity({k:v for k,v in phrase.items() if k!='enrollment_sha256'}):
        raise ValueError('Enrollment settings or calibration identity changed; enroll again')
    if set(phrase.get('decoder_parameters',{}))!=PARAMETERS:raise ValueError('Incomplete versioned decoder settings')
    if any(not math.isfinite(v) for v in phrase['decoder_parameters'].values()):raise ValueError('Nonfinite decoder settings')
    # Existing source calibration, when available, must match its content hash.
    path=phrase.get('calibration_path')
    if path and Path(path).exists() and sha(path)!=phrase.get('calibration_sha256'):
        raise ValueError('Calibration changed after enrollment')
