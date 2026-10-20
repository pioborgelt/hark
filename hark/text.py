"""Deterministic text enrollment, independent of acoustic training."""
import re
import hashlib
import json
from functools import lru_cache
from piper.phonemize_espeak import EspeakPhonemizer

PRONUNCIATION_VERSIONS=('espeak-v1','de-lexicon-v1')
LEXICON={'version':'de-lexicon-v1','language':'de','lemma':'okay','standard':['o','k','e'],
         'espeak_word':['o','k','ɛ','ɪ'],
         'source':'https://www.duden.de/rechtschreibung/okay_abgemacht_einverstanden'}
LEXICON_SHA256=hashlib.sha256(json.dumps(LEXICON,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

@lru_cache(maxsize=1)
def phonemizer():
    return EspeakPhonemizer()

@lru_cache(maxsize=10000)
def phones(text, language, version="legacy-lower-v1", pronunciation_version="espeak-v1"):
    if language not in ("de", "en"):
        raise ValueError("language must be de or en")
    if not text.strip():
        raise ValueError("phrase must not be empty")
    # Use one canonical English pronunciation across US/GB synthesizers.
    if version not in ("legacy-lower-v1","espeak-case-v2"):
        raise ValueError("Unknown text/phoneme version")
    if pronunciation_version not in PRONUNCIATION_VERSIONS:raise ValueError('Unknown pronunciation version')
    source=text.strip().lower() if version=="legacy-lower-v1" else text.strip()
    seq = phonemizer().phonemize("de" if language == "de" else "en-us", source)
    # IPA codepoints, with stress, length and punctuation removed. No word-boundary
    # tokens: silence and word spaces are not acoustically interchangeable.
    ignored=" ˈˌː.,!?;:\"'()\n\r-"
    if pronunciation_version=='de-lexicon-v1' and language=='de' and re.search(r'\bokay\b',text,re.IGNORECASE):
        # Work on emitted phonemic WORDS. Rewriting source spelling can change
        # unrelated acronym pronunciations through eSpeak's context rules.
        result=[]
        for sentence in seq:
            word=[]
            def append_word():
                result.extend(LEXICON['standard'] if word==LEXICON['espeak_word'] else word)
                word.clear()
            for p in sentence:
                if p.isspace():append_word()
                elif p not in ignored:word.append(p)
            append_word()
        return result
    return [p for s in seq for p in s if p not in ignored and not p.isspace()]

def encode(text, language, vocab, version="legacy-lower-v1", pronunciation_version="espeak-v1"):
    seq = phones(text, language, version, pronunciation_version)
    if not seq:
        raise ValueError("phrase has no pronounceable tokens")
    missing = sorted(set(seq) - set(vocab))
    if missing:
        raise ValueError(f"Untrained phonemes {missing}; use a supported phrase")
    return [vocab[p] for p in seq]
