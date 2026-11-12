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


def encode_multi(text, language, vocab, version="legacy-lower-v1", pronunciation_version="espeak-v1"):
    version = version or "legacy-lower-v1"
    base = encode(text, language, vocab, version, pronunciation_version)
    if language != "en":
        return [base]
    seqs = [base]
    ignored = " ˈˌː.,!?;:\"'()\n\r-"
    source = text.strip().lower() if version == "legacy-lower-v1" else text.strip()
    rp = [p for s in phonemizer().phonemize("en-gb-x-rp", source) for p in s if p not in ignored and not p.isspace()]
    if rp and all(p in vocab for p in rp):
        alt = [vocab[p] for p in rp]
        if (len(alt) >= len(base) or len(alt) >= 8) and alt not in seqs:
            seqs.append(alt)
    inv = {i: p for p, i in vocab.items()}
    names = [inv[t] for t in base]
    if "ɾ" in names and "t" in vocab:
        t_seq = [vocab["t"] if inv[t] == "ɾ" else (vocab["ɜ"] if inv[t] == "ɚ" and "ɜ" in vocab else t) for t in base]
        if t_seq not in seqs:
            seqs.append(t_seq)
    low = text.strip().lower()
    if low.endswith("a") and names[-1] in ("ə", "ɐ"):
        pairs = ([("a", "a"), ("a", "ɛ"), ("ɐ", "ɛ"), ("ɐ", "ɚ")]
                 if low.startswith("a") and names[0] in ("ɐ", "ə")
                 else [(names[0], x) for x in ("a", "ɛ", "ɚ")])
        for init_p, end_p in pairs:
            if init_p in vocab and end_p in vocab:
                cand = list(base)
                cand[0] = vocab[init_p]
                cand[-1] = vocab[end_p]
                if cand not in seqs:
                    seqs.append(cand)
    return seqs

