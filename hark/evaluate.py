"""Dev-only threshold choice; held-out phrases/voices, hard negatives and real speech.

The short reused negative recordings establish functionality, not deployment FAR.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from .audio import LogMel, read_audio, RATE
from .legacy_detect import Matcher, Detector
from .model import load_model
from .text import encode
from .train import audit, augment, edit_distance

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def music(seconds=30,seed=9):
    rng=np.random.default_rng(seed)
    t=np.arange(int(seconds*RATE))/RATE
    audio=np.zeros_like(t)
    for i,f in enumerate([130.81,164.81,196.,261.63,329.63,392.]):
        audio += .04*np.sin(2*np.pi*f*t+rng.uniform(0,6.28))*(.5+.5*np.sin(2*np.pi*(.7+i*.13)*t))
    audio += .015*rng.normal(size=len(t))
    return audio.astype(np.float32)

def max_score(logprobs,tokens):
    m=Matcher(tokens,threshold=np.inf)
    best=-np.inf;at=None
    for lp in logprobs:
        m.push(lp)
        if m.frame-m.starts[-1]>=max(2,len(tokens)//2):
            score=m.cost[-1]/len(tokens)
            if score>best:best=float(score);at=.025+m.frame*.02
    return best,at

def events(logprobs,tokens,threshold):
    m=Matcher(tokens,threshold=threshold)
    return [e for lp in logprobs if (e:=m.push(lp)) is not None]

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="artifacts/prototype/best.pt")
    p.add_argument("--manifest",default="data/manifest.jsonl")
    p.add_argument("--out",default="reports/evaluation.json")
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    model,ck=load_model(a.checkpoint)
    records=rows(a.manifest);audit(records)
    frontend=LogMel();cache={}
    @torch.inference_mode()
    def infer(audio):
        return model(frontend(torch.from_numpy(audio))[None,::2])[0][0].numpy()
    def logits(r):
        if r["id"] not in cache:cache[r["id"]]=infer(read_audio(r["path"]))
        return cache[r["id"]]
    def tokens(r):return encode(r["text"],r["language"],ck["vocab"])
    unsupported=[]
    supported=[]
    for r in records:
        if r["split"]=="train":continue
        try:tokens(r);supported.append(r)
        except ValueError:unsupported.append({"id":r["id"],"split":r["split"],"text":r["text"]})
    # Each dev phrase is paired with its own audio and the five nearest phonemic
    # neighbors in the development set. Test scores never choose the threshold.
    calibration={}
    for lang in ("de","en"):
        dev=[r for r in supported if r["split"]=="dev" and r["language"]==lang]
        pairs=[]
        for r in dev:
            tok=tokens(r)
            pairs.append((max_score(logits(r),tok)[0],True))
            neighbors=sorted([s for s in dev if s["id"]!=r["id"]],key=lambda s:edit_distance(tok,tokens(s)))[:5]
            pairs.extend((max_score(logits(s),tok)[0],False) for s in neighbors)
        positives=[s for s,y in pairs if y];negatives=[s for s,y in pairs if not y]
        candidates=sorted(set([-.01,-.5,-1.,-2.,-3.,-4.,-6.,-8.,-12.]+[s for s,y in pairs if np.isfinite(s)]),reverse=True)
        scored=[]
        for threshold in candidates:
            miss=float(np.mean([s<threshold for s in positives]))
            false=float(np.mean([s>=threshold for s in negatives]))
            scored.append((.5*(miss+false),false,-threshold,threshold,miss))
        _,f,_,threshold,miss=min(scored)
        calibration[lang]={"threshold":threshold,"dev_miss_rate":miss,"dev_pair_false_positive_rate":f,
                           "positive_pairs":len(positives),"negative_pairs":len(negatives),
                           "positive_scores":positives,"negative_scores":negatives}
        print(json.dumps({"calibrated":lang,**{k:v for k,v in calibration[lang].items() if not k.endswith("scores")}}),flush=True)
    realpath=Path("data/commonphone/manifest.jsonl")
    real=rows(realpath) if realpath.exists() else []
    real_ids={r["speaker"] for r in real}
    assert not real_ids & {r["speaker"] for r in records}
    results=[];latencies=[];cpu=[];negative_unique={}
    examples=Path("artifacts/examples");examples.mkdir(parents=True,exist_ok=True)
    for lang in ("de","en"):
        test=[r for r in supported if r["split"]=="test" and r["language"]==lang]
        threshold=calibration[lang]["threshold"]
        reals=[r for r in real if r["language"]==lang]
        speech=np.concatenate([read_audio(r["path"]) for r in reals]) if reals else np.zeros(RATE,dtype=np.float32)
        tonal=music(30,18 if lang=="de" else 19)
        negative_unique[lang]={"real_speech_seconds":len(speech)/RATE,"generated_music_seconds":len(tonal)/RATE,
                               "real_utterances":len(reals),"real_speakers":len({r["speaker"] for r in reals})}
        speech_lp=infer(speech);music_lp=infer(tonal)
        sf.write(examples/f"negative_{lang}.wav",np.concatenate([speech,tonal]),RATE)
        for index,r in enumerate(test):
            tok=tokens(r)
            # Streaming event criterion permits a detection during the labeled
            # target, with .5 s suffix allowance. Outer padding is not speech.
            audio=read_audio(r["path"])
            audio=np.pad(audio,(RATE//2,RATE//2))
            sf.write(examples/f"{lang}_{index:02}.wav",audio,RATE)
            detector=Detector(model,tok,threshold);ev=[]
            start=time.perf_counter()
            for i in range(0,len(audio),1280):ev+=detector.push(audio[i:i+1280])
            elapsed=time.perf_counter()-start
            cpu.append({"audio_seconds":len(audio)/RATE,"cpu_seconds":elapsed})
            # End annotation estimated from last sample above -40 dB relative to
            # peak; this is an acoustic proxy, not a hand-annotated boundary.
            active=np.flatnonzero(np.abs(audio)>max(np.max(np.abs(audio))*.01,1e-5))
            end=(active[-1]+1)/RATE if len(active) else len(audio)/RATE
            valid=[e for e in ev if .6<=e["time"]<=end+.5]
            if valid:latencies.append(valid[0]["available_at"]-end)
            hard=sorted([s for s in test if s["id"]!=r["id"]],key=lambda s:edit_distance(tok,tokens(s)))[:3]
            hard_events=sum(bool(events(logits(s),tok,threshold)) for s in hard)
            false_speech=len(events(speech_lp,tok,threshold));false_music=len(events(music_lp,tok,threshold))
            noisy=augment(audio,np.random.default_rng(index+333))
            noisy_events=events(infer(noisy),tok,threshold)
            sf.write(examples/f"{lang}_{index:02}_augmented.wav",noisy,RATE)
            result={"id":r["id"],"text":r["text"],"language":lang,"speaker":r["speaker"],
                    "threshold":threshold,"detected":bool(valid),"events":ev,"end_proxy":end,
                    "augmented_detected":any(.6<=e["time"]<=end+.5 for e in noisy_events),
                    "hard_negative_utterances":len(hard),"hard_negative_false_utterances":hard_events,
                    "real_negative_events":false_speech,"music_negative_events":false_music,
                    "negative_seconds":(len(speech)+len(tonal))/RATE,"example":str(examples/f"{lang}_{index:02}.wav")}
            results.append(result)
            print(json.dumps({"evaluated":lang,"index":index,"detected":bool(valid),"false_speech":false_speech,"false_music":false_music}),flush=True)
    total_seconds=sum(x["negative_seconds"] for x in results)
    false=sum(x["real_negative_events"]+x["music_negative_events"] for x in results)
    misses=sum(not x["detected"] for x in results)
    aggregate={"phrases":len(results),"misses":misses,"miss_rate":misses/len(results),
               "augmented_miss_rate":sum(not x["augmented_detected"] for x in results)/len(results),
               "false_events":false,"negative_query_hours":total_seconds/3600,
               "false_events_per_query_hour":false/(total_seconds/3600),
               "hard_negative_false_utterances":sum(r["hard_negative_false_utterances"] for r in results),
               "hard_negative_utterances":sum(r["hard_negative_utterances"] for r in results),
               "latency_proxy_seconds_median":float(np.median(latencies)) if latencies else None,
               "latency_proxy_seconds_p95":float(np.percentile(latencies,95)) if latencies else None,
               "cpu_real_time_factor":sum(r["cpu_seconds"] for r in cpu)/sum(r["audio_seconds"] for r in cpu),
               "cpu_audio_seconds":sum(r["audio_seconds"] for r in cpu),"cpu_seconds":sum(r["cpu_seconds"] for r in cpu)}
    output={"checkpoint":a.checkpoint,"checkpoint_sha256":hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
            "epoch":ck["epoch"],"parameters":ck["parameters"],"calibration":calibration,"aggregate":aggregate,
            "unique_negative_audio":negative_unique,"unsupported":unsupported,"phrases":results,
            "limitations":["Small synthetic positive set and four training voices; no real-positive recall measurement.",
                           "Negative recordings are reused across queries. Query-hours are not independent unique audio hours.",
                           "Latency is relative to an amplitude-derived end proxy and can be negative (early firing).",
                           "Generated tonal accompaniment is not a representative music collection.",
                           "Threshold optimized on synthetic dev pairs; no claim of reliable production FAR."]}
    Path(a.out).write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))
    calibration_path=Path(a.checkpoint).parent/"calibration.json"
    calibration_path.write_text(json.dumps({"checkpoint_sha256":output["checkpoint_sha256"],
                                            "languages":calibration},indent=2,allow_nan=False))
    print(json.dumps(aggregate),flush=True)

if __name__=="__main__":main()
