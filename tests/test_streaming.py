import json
import numpy as np
import pytest
import torch
from hark.audio import FeatureStream, LogMel
from hark.legacy_detect import Matcher, Detector
from hark.model import AcousticModel
from hark.train import audit

torch.set_num_threads(1)

@pytest.mark.parametrize("layers",[0,2])
def test_arbitrary_chunks_equal_offline_and_no_future_leakage(layers):
    torch.manual_seed(7)
    x = torch.randn(16317)
    stream = FeatureStream()
    actual = torch.cat([stream.push(x[i:i+137]) for i in range(0,len(x),137)])
    expected = LogMel()(x)[::2]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    m = AcousticModel(7,32,causal_layers=layers).eval()
    with torch.no_grad():
        full,_ = m(expected[None])
        prefix,_ = m(expected[None,:17])
        state=None; chunks=[]
        for f in expected.split(9):
            lp,state=m(f[None],state); chunks.append(lp)
    torch.testing.assert_close(torch.cat(chunks,dim=1),full,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(prefix,full[:,:17],atol=1e-6,rtol=1e-5)

def frames(ids, classes=5):
    lp = np.full((len(ids),classes),-20.)
    for i,v in enumerate(ids): lp[i,v]=0
    return lp

def test_entire_phrase_required_and_state_crosses_chunks():
    m=Matcher([1,2,3],threshold=-.1)
    assert not any(m.push(x) for x in frames([0,1,1,0,2,2,0,0,0]))
    assert m.push(frames([3])[0]) is not None

def test_repeated_phone_requires_blank_and_order_matters():
    m=Matcher([1,1,2],threshold=-.1)
    assert not any(m.push(x) for x in frames([1,1,1,2,2,0]))
    m=Matcher([1,1,2],threshold=-.1)
    assert any(m.push(x) for x in frames([1,0,1,0,2]))
    m=Matcher([1,2,3],threshold=-.1)
    assert not any(m.push(x) for x in frames([3,0,2,0,1,0]))

def test_matcher_memory_does_not_grow_with_stream_duration():
    m=Matcher([1,2],threshold=-.1)
    for x in frames([0]*3000): assert m.push(x) is None
    assert m.cost.shape==(3,)
    assert m.starts.shape==(3,)

@pytest.mark.parametrize("field",["speaker","text","origin_id"])
def test_split_audit_rejects_leakage(field):
    a={"split":"train","speaker":"a","text":"a","origin_id":"a"}
    b={"split":"test","speaker":"b","text":"b","origin_id":"b"}
    b[field]=a[field]
    with pytest.raises(ValueError,match="leakage"): audit([a,b])

def test_generated_manifest_has_disjoint_phrases_speakers_and_origins():
    from pathlib import Path
    p=Path("data/manifest.jsonl")
    if not p.exists(): pytest.skip("generate corpus first")
    audit([json.loads(line) for line in p.read_text().splitlines()])

def test_trained_detector_is_invariant_to_audio_chunk_boundaries():
    from pathlib import Path
    from hark.model import load_model
    from hark.audio import read_audio
    from hark.text import encode
    checkpoint=Path("artifacts/prototype/best.pt")
    example=Path("artifacts/examples/de_02.wav")
    if not checkpoint.exists() or not example.exists():pytest.skip("train/evaluate prototype first")
    model,ck=load_model(checkpoint)
    audio=read_audio(example)
    def run(chunk):
        detector=Detector(model,encode("hallo blauer computer","de",ck["vocab"]),-1.9031397875617533)
        return [e for i in range(0,len(audio),chunk) for e in detector.push(audio[i:i+chunk])]
    a,b=run(137),run(1280)
    assert len(a)==len(b)>0
    assert [e["time"] for e in a]==[e["time"] for e in b]
    np.testing.assert_allclose([e["score"] for e in a],[e["score"] for e in b],atol=1e-5)

def test_checkpoint_resume_matches_uninterrupted_optimizer_updates(tmp_path):
    import soundfile as sf
    from hark.train import main
    records=[]
    for i in range(3):
        wav=tmp_path/f"{i}.wav"
        t=np.arange(4800)/16000
        sf.write(wav,.2*np.sin(2*np.pi*(200+i*75)*t),16000)
        records.append({"id":str(i),"origin_id":str(i),"speaker":str(i),"text":str(i),
                        "phones":["a","b"] if i%2 else ["b","a"],"path":str(wav),"split":"train"})
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in records))
    common=["--manifest",str(manifest),"--hidden","16","--threads","1","--batch","2"]
    whole=tmp_path/"whole";resumed=tmp_path/"resumed"
    main(common+["--out",str(whole),"--epochs","3"])
    main(common+["--out",str(resumed),"--epochs","2"])
    main(common+["--out",str(resumed),"--epochs","3","--resume",str(resumed/"last.pt")])
    a=torch.load(whole/"last.pt",weights_only=False)
    b=torch.load(resumed/"last.pt",weights_only=False)
    assert a["step"]==b["step"]==6
    for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],atol=0,rtol=0)
