"""Real unseen phrases/speakers, dev-only FAR calibration, historical comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import soundfile as sf
import torch
from .audio import LogMel,read_audio,RATE
from .legacy_detect import Detector
from .evaluate import max_score,events,music
from .model import load_model
from .text import encode
from .train import audit,edit_distance,augment

def key(text):return " ".join(re.findall(r"\w+",text.lower()))

def phone_key(seq):return "|"+"|".join(seq)+"|"

def negative_for(candidate,query):
    return phone_key(query["phones"]) not in phone_key(candidate["phones"])

def select_queries(records,split,lang,vocab,ngrams,limit):
    selected=[]
    for r in sorted(records,key=lambda s:s["id"]):
        if r["split"]!=split or r["language"]!=lang or r["source"]=="piper":continue
        words=key(r["text"]).split()
        if not 2<=len(words)<=5 or not .4<r["seconds"]<3.8:continue
        if key(r["text"]) in ngrams:continue
        try:tokens=encode(r["text"],lang,vocab)
        except ValueError:continue
        if 6<=len(tokens)<=32:selected.append(r)
        if len(selected)>=limit:break
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",default="artifacts/real-v1/best.pt")
    p.add_argument("--out",default="reports/real_v1_evaluation.json")
    p.add_argument("--queries",type=int,default=40);p.add_argument("--negative-seconds",type=float,default=900)
    p.add_argument("--calibration-only",action="store_true")
    p.add_argument("--calibration-file")
    p.add_argument("--threads",type=int,default=2);a=p.parse_args();torch.set_num_threads(a.threads)
    model,ck=load_model(a.checkpoint);front=LogMel()
    records=[json.loads(s) for s in Path("data/real-v1/manifest.jsonl").read_text().splitlines()];audit(records)
    ngrams=set()
    for r in records:
        if r["split"]!="train":continue
        words=key(r["text"]).split()
        for n in range(2,6):
            for i in range(len(words)-n+1):ngrams.add(" ".join(words[i:i+n]))
    checksum=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    frozen=json.loads(Path(a.calibration_file).read_text()) if a.calibration_file else None
    if frozen and frozen["checkpoint_sha256"]!=checksum:raise ValueError("Frozen calibration checkpoint mismatch")
    out=Path(a.checkpoint).parent;examples=out/"examples";examples.mkdir(exist_ok=True)
    cache={}
    @torch.inference_mode()
    def infer(audio):
        # Bounded memory, continuous acoustic state across longer recordings.
        features=front(torch.from_numpy(audio))[::2];state=None;chunks=[]
        for x in features.split(512):
            lp,state=model(x[None],state);chunks.append(lp[0].numpy())
        return np.concatenate(chunks)
    def tokens(r):return encode(r["text"],r["language"],ck["vocab"])
    def logits(r):
        if r["id"] not in cache:cache[r["id"]]=infer(np.pad(read_audio(r["path"]),(8000,8000)))
        return cache[r["id"]]
    queries={(s,l):select_queries(records,s,l,ck["vocab"],ngrams,a.queries) for s in ["dev","test"] for l in ["de","en"]}
    selection={s+"_"+l:{"selected":len(v),"speakers":len({r["speaker"] for r in v}),
        "eligible_rule":"2-5 words, 6-32 supported IPA tokens, duration .4-3.8s; absent from all training 2-5-grams"}
        for (s,l),v in queries.items()}
    all_phrasekeys=[" "+key(r["text"])+" " for items in queries.values() for r in items]
    all_phonekeys=[phone_key(r["phones"]) for items in queries.values() for r in items]
    negative={};negative_meta={};background_audio={}
    for split in (["dev"] if a.calibration_only else ["test"] if frozen else ["dev","test"]):
        for lang in ["de","en"]:
            chosen=[];seconds=0.
            for r in sorted(records,key=lambda r:r["id"]):
                if r["split"]!=split or r["language"]!=lang or r["source"]=="piper":continue
                sentence=" "+key(r["text"])+" "
                if any(phrase in sentence for phrase in all_phrasekeys):continue
                if any(phrase in phone_key(r["phones"]) for phrase in all_phonekeys):continue
                chosen.append(r);seconds+=r["seconds"]
                if seconds>=a.negative_seconds:break
            speech=np.concatenate([read_audio(r["path"]) for r in chosen])
            # Generated instrumental tones remain explicitly separate from speech.
            tonal=music(60,123 if split=="dev" else 321)
            negative[split,lang]=(infer(speech),infer(tonal))
            if split=="test":background_audio[lang]=speech
            negative_meta[split+"_"+lang]={"speech_seconds":len(speech)/RATE,"music_seconds":len(tonal)/RATE,
                "utterances":len(chosen),"speakers":len({r["speaker"] for r in chosen}),"ids":[r["id"] for r in chosen]}
            if split=="test":sf.write(examples/f"negative_{lang}.wav",np.concatenate([speech,tonal]),RATE)
            print(json.dumps({"negative_prepared":split+"_"+lang,"seconds":seconds}),flush=True)
    calibration=frozen["calibration"] if frozen else {}
    for lang in ["de","en"]:
        if frozen:continue
        dev=queries["dev",lang];positive=[];pair_negative=[];background=[]
        speech_lp,music_lp=negative["dev",lang]
        # Fixed dev queries sample continuous speech/music in addition to close
        # sentence pairs. Thresholds enforce zero events on these observed streams.
        for i,r in enumerate(dev):
            tok=tokens(r);positive.append(max_score(logits(r),tok)[0])
            nearest=sorted([s for s in dev if s["id"]!=r["id"] and negative_for(s,r)],key=lambda s:edit_distance(tok,tokens(s)))[:3]
            pair_negative.extend(max_score(logits(s),tok)[0] for s in nearest)
            if i<8:
                background.extend([max_score(speech_lp,tok)[0],max_score(music_lp,tok)[0]])
        candidates=sorted(set(positive+pair_negative+[-.01,-.1,-.25,-.5,-1.,-2.,-3.]),reverse=True)
        finite=[v for v in candidates if np.isfinite(v)]
        ranked=[]
        for threshold in finite:
            miss=float(np.mean(np.array(positive)<threshold));fp=float(np.mean(np.array(pair_negative)>=threshold))
            ranked.append((.5*(miss+fp),fp,-threshold,threshold))
        pair_threshold=min(ranked)[-1]
        threshold=max(pair_threshold,max(background)+.01)
        # CTC relative scores cannot exceed 0. A requirement above 0 disables
        # detection; retain it honestly rather than silently violating the target.
        calibration[lang]={"threshold":threshold,"pair_threshold":pair_threshold,
            "dev_miss_rate":float(np.mean(np.array(positive)<threshold)),
            "dev_pair_false_positive_rate":float(np.mean(np.array(pair_negative)>=threshold)),
            "dev_continuous_max_score":max(background),"continuous_queries":min(8,len(dev)),
            "positive_pairs":len(positive),"negative_pairs":len(pair_negative),"positive_scores":positive,
            "negative_scores":pair_negative,"background_scores":background}
        print(json.dumps({"calibrated":lang,"threshold":threshold,"dev_miss_rate":calibration[lang]["dev_miss_rate"]}),flush=True)
    if a.calibration_only:
        Path(a.out).write_text(json.dumps({"checkpoint":a.checkpoint,"checkpoint_sha256":checksum,
            "calibration":calibration,"selection":selection,"negative_audio":negative_meta,
            "dev_ctc":ck["best"],"parameters":ck["parameters"],"epoch":ck["epoch"]},ensure_ascii=False,indent=2,allow_nan=False))
        return
    results=[];timings=[];delays=[]
    for lang in ["de","en"]:
        test=queries["test",lang];threshold=calibration[lang]["threshold"]
        balanced=calibration[lang]["pair_threshold"]
        speech_lp,music_lp=negative["test",lang]
        for i,r in enumerate(test):
            tok=tokens(r);audio=np.pad(read_audio(r["path"]),(8000,8000))
            detector=Detector(model,tok,threshold);found=[];started=time.perf_counter()
            for j in range(0,len(audio),1280):found.extend(detector.push(audio[j:j+1280]))
            timings.append({"seconds":len(audio)/RATE,"cpu_seconds":time.perf_counter()-started})
            active=np.flatnonzero(np.abs(audio)>max(np.max(np.abs(audio))*.01,1e-5))
            end=(active[-1]+1)/RATE
            valid=[e for e in found if .5<=e["time"]<=end+.5]
            balanced_found=events(logits(r),tok,balanced)
            balanced_valid=any(.5<=e["time"]<=end+.5 for e in balanced_found)
            if valid:delays.append(valid[0]["available_at"]-end)
            nearest=sorted([s for s in test if s["id"]!=r["id"] and negative_for(s,r)],key=lambda s:edit_distance(tok,tokens(s)))[:3]
            false_pairs=sum(bool(events(logits(s),tok,threshold)) for s in nearest)
            # Limit long continuous evaluation to 8 fixed queries/language; keep
            # exact durations and event counts per phrase in the report.
            bg_speech=len(events(speech_lp,tok,threshold)) if i<8 else None
            bg_music=len(events(music_lp,tok,threshold)) if i<8 else None
            balanced_bg_speech=len(events(speech_lp,tok,balanced)) if i<8 else None
            balanced_bg_music=len(events(music_lp,tok,balanced)) if i<8 else None
            noisy=augment(audio,np.random.default_rng(900+i))
            noise_events=events(infer(noisy),tok,threshold)
            example=examples/f"{lang}_{i:02}.wav";sf.write(example,audio,RATE)
            background=background_audio[lang]
            offset=(i*63281)%max(1,len(background)-RATE*4)
            prefix=background[offset:offset+RATE*2]
            suffix=background[offset+RATE*2:offset+RATE*4]
            stream=np.concatenate([prefix,audio,suffix])
            embedded=events(infer(stream),tok,threshold)
            embedded_ok=any(2+.5<=e["time"]<=2+end+.5 for e in embedded)
            sf.write(examples/f"{lang}_{i:02}_stream.wav",stream,RATE)
            sf.write(examples/f"{lang}_{i:02}_augmented.wav",noisy,RATE)
            results.append({"id":r["id"],"text":r["text"],"language":lang,"speaker":r["speaker"],
                "detected":bool(valid),"events":found,"end_proxy":end,"example":str(example),
                "balanced_detected":balanced_valid,
                "balanced_continuous_speech_events":balanced_bg_speech,"balanced_continuous_music_events":balanced_bg_music,
                "embedded_detected":embedded_ok,
                "augmented_detected":any(.5<=e["time"]<=end+.5 for e in noise_events),
                "false_hard_pairs":false_pairs,"hard_pairs":len(nearest),
                "continuous_speech_events":bg_speech,"continuous_music_events":bg_music,
                "continuous_seconds":sum(negative_meta["test_"+lang][k] for k in ["speech_seconds","music_seconds"]) if i<8 else 0})
            print(json.dumps({"tested":lang,"index":i,"detected":bool(valid),"false_speech":bg_speech}),flush=True)
    # Historical synthetic test: dev/test remain completely excluded from training.
    historical=[]
    for r in records:
        if r["source"]!="piper" or r["split"]!="test":continue
        try:tok=tokens(r)
        except ValueError:historical.append({"text":r["text"],"supported":False,"detected":False});continue
        found=events(logits(r),tok,calibration[r["language"]]["threshold"])
        audio=np.pad(read_audio(r["path"]),(8000,8000))
        active=np.flatnonzero(np.abs(audio)>max(np.abs(audio).max()*.01,1e-5));end=(active[-1]+1)/RATE
        historical.append({"text":r["text"],"supported":True,"detected":any(.6<=e["time"]<=end+.5 for e in found)})
    negative_seconds=sum(r["continuous_seconds"] for r in results)
    false=sum((r["continuous_speech_events"] or 0)+(r["continuous_music_events"] or 0) for r in results)
    balanced_false=sum((r["balanced_continuous_speech_events"] or 0)+(r["balanced_continuous_music_events"] or 0) for r in results)
    aggregate={"phrases":len(results),"detected":sum(r["detected"] for r in results),
        "miss_rate":sum(not r["detected"] for r in results)/len(results),
        "balanced_miss_rate":sum(not r["balanced_detected"] for r in results)/len(results),
        "balanced_false_events":balanced_false,"balanced_false_events_per_query_hour":balanced_false/(negative_seconds/3600),
        "augmented_miss_rate":sum(not r["augmented_detected"] for r in results)/len(results),
        "embedded_miss_rate":sum(not r["embedded_detected"] for r in results)/len(results),
        "hard_false_pairs":sum(r["false_hard_pairs"] for r in results),"hard_pairs":sum(r["hard_pairs"] for r in results),
        "false_events":false,"negative_query_hours":negative_seconds/3600,"false_events_per_query_hour":false/(negative_seconds/3600),
        "unique_test_negative_seconds":sum(v["speech_seconds"]+v["music_seconds"] for k,v in negative_meta.items() if k.startswith("test")),
        "historical_synthetic_detected":sum(r["detected"] for r in historical),"historical_synthetic_phrases":len(historical),
        "cpu_real_time_factor":sum(r["cpu_seconds"] for r in timings)/sum(r["seconds"] for r in timings),
        "latency_proxy_median":float(np.median(delays)) if delays else None,
        "latency_proxy_p95":float(np.percentile(delays,95)) if delays else None}
    if frozen:negative_meta={**frozen["negative_audio"],**negative_meta}
    report={"checkpoint":a.checkpoint,"checkpoint_sha256":checksum,"epoch":ck["epoch"],"parameters":ck["parameters"],
        "aggregate":aggregate,"calibration":calibration,"negative_audio":negative_meta,"phrases":results,"historical":historical,"selection":selection,
        "limitations":["Whole short real utterances, plus concatenated speech context; not naturally recorded wakewords in conversation.",
            "Negative audio reused across queries; query-hours are not unique independent hours.",
            "Generated tonal music is not representative real music.","Latency uses an amplitude-derived endpoint; early detections can have negative delay.",
            "Only supported phonemes/short utterances evaluated; selection rules and counts must accompany results."]}
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    (out/"calibration.json").write_text(json.dumps({"checkpoint_sha256":checksum,"languages":calibration},indent=2,allow_nan=False))
    print(json.dumps(aggregate),flush=True)

if __name__=="__main__":main()
