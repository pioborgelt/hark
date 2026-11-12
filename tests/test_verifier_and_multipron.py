import numpy as np
import torch

from hark.audio import FeatureStream, LogMel
from hark.text import encode, encode_multi, phones
from hark.verifier import BiPhonemeVerifier, score_trigger_queries


def _sample_vocab():
    chars = sorted(set(
        phones("alexa computer jarvis smart mirror snowboy view glass", "en")
        + phones("okay hallo computer", "de")
        + list("aɐəɛɪɔʊuɜɚɾtkpbdɡfszʃʒmnŋlvɹjhw")
    ))
    return {p: i + 1 for i, p in enumerate(chars)}


def test_encode_multi_deterministic_variants():
    vocab = _sample_vocab()
    base_comp = encode("computer", "en", vocab)
    multi_comp = encode_multi("computer", "en", vocab)
    assert multi_comp[0] == base_comp
    assert len(multi_comp) >= 2
    assert len({tuple(s) for s in multi_comp}) == len(multi_comp)

    multi_alexa = encode_multi("alexa", "en", vocab)
    assert len(multi_alexa) >= 3

    multi_de = encode_multi("okay", "de", vocab)
    assert multi_de == [encode("okay", "de", vocab)]


def test_verifier_window_dedup_and_query_equivalence():
    torch.manual_seed(7)
    verifier = BiPhonemeVerifier(
        classes=42, hidden=96, gru_hidden=64, gru_layers=2, attn_dim=64, attn_heads=2, bins=64
    ).eval()

    # Check win_idx deduplication matches duplicated-window batching
    w0 = torch.randn(48, 64)
    w1 = torch.randn(56, 64)
    uniq = torch.zeros(2, 56, 64)
    uniq[0, :48] = w0
    uniq[1, :56] = w1
    u_lens = torch.tensor([48, 56])
    win_idx = torch.tensor([0, 0, 1])
    bounds = torch.tensor([[10, 34], [10, 34], [12, 42]])
    toks = torch.tensor([[3, 7, 11, 5, 0], [3, 9, 11, 5, 2], [4, 8, 12, 6, 0]])
    tlens = torch.tensor([4, 5, 4])

    dup = uniq[win_idx]
    d_lens = u_lens[win_idx]
    with torch.no_grad():
        lg_dedup, _ = verifier.verify_windows(uniq, u_lens, bounds, toks, tlens, win_idx=win_idx)
        lg_dup, _ = verifier.verify_windows(dup, d_lens, bounds, toks, tlens, win_idx=None)
    assert torch.allclose(lg_dedup, lg_dup, atol=1e-5)

    # Check score_trigger_queries audio slicing matches FeatureStream 50 Hz features exactly
    rng = np.random.RandomState(11)
    audio = (rng.randn(16000 * 4) * 0.08).astype(np.float32)
    feats = FeatureStream(64).push(audio).numpy()
    lm = LogMel(64)
    queries = [
        (72, 40, ((3, 7, 11, 5), (3, 8, 11, 5))),
        (135, 98, ((4, 8, 12, 6),)),
    ]
    sc_audio = score_trigger_queries(verifier, audio, queries, v_front=lm, is_features=False)
    sc_feats = score_trigger_queries(verifier, feats, queries, is_features=True)
    for q in queries:
        assert abs(sc_audio[q] - sc_feats[q]) < 1e-5
