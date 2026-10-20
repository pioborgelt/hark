import torch
from hark.alignment import frame_targets

def test_exact_phones_and_silence_are_supervised_without_filling_unknown_audio():
    record={"phones":["a","b"],"labels":[{"start":0.,"end":.04,"symbol":"(...)"},
        {"start":.04,"end":.10,"symbol":"a"},{"start":.10,"end":.16,"symbol":"b"}]}
    target=frame_targets(record,{"a":1,"b":2},10)
    assert target.tolist()==[0,0,1,1,1,2,2,2,-100,-100]

def test_incompatible_annotation_produces_no_frame_targets():
    r={"phones":["a","b","c"],"labels":[{"start":0.,"end":1.,"symbol":"xyz"}]}
    assert torch.all(frame_targets(r,{"a":1,"b":2,"c":3},20)==-100)
