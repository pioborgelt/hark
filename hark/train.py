import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from scipy.signal import fftconvolve
from torch.nn.utils.rnn import pad_sequence
from .audio import LogMel, read_audio, RATE
from .model import AcousticModel
from .alignment import frame_targets

def audit(records):
    for field in ("origin_id", "speaker", "text"):
        groups = {s: {r[field] for r in records if r["split"] == s} for s in ("train", "dev", "test")}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if groups[a] & groups[b]:
                raise ValueError(f"{field} leakage between {a} and {b}")

def augment(x, rng):
    """Generated tonal music/noise, short sparse room tail and distance gain."""
    x = x.copy()
    if rng.random() < .7:
        tail = np.zeros(rng.integers(800, 4000), dtype=np.float32)
        tail[0] = 1
        for _ in range(12):
            i = rng.integers(100, len(tail))
            tail[i] = rng.uniform(-.25, .25) * np.exp(-i / 1600)
        x = fftconvolve(x, tail)[:len(x)]
    t = np.arange(len(x)) / RATE
    noise = rng.normal(size=len(x))
    for _ in range(4):
        noise += rng.uniform(.2, .8) * np.sin(2*np.pi*rng.uniform(80, 2000)*t + rng.uniform(0, 6.28))
    rms = np.sqrt(np.mean(x*x) + 1e-9)
    noise = noise / np.sqrt(np.mean(noise*noise))
    x += noise * rms * 10 ** (-rng.uniform(10, 30) / 20)
    return (x * rng.uniform(.15, 1)).astype(np.float32)

FEATURE_VERSION="logmel-v1-waveaug-v1"

def cache_features(records, cache, manifest_hash, half=False, background_manifest=None, background_variants=3, prepared_version=None):
    version=prepared_version or FEATURE_VERSION+("-fp16" if half else "-fp32")
    if background_manifest:
        version += "-realbg-"+hashlib.sha256(Path(background_manifest).read_bytes()).hexdigest()+f"-n{background_variants}"
    if cache.exists():
        saved = torch.load(cache, weights_only=False)
        if saved["manifest_hash"] == manifest_hash and saved.get("feature_version")==version:
            return saved["features"]
    if prepared_version:raise ValueError("Prepared feature cache missing or manifest/version mismatch")
    backgrounds = []
    if background_manifest:
        backgrounds = [read_audio(r["path"]) for r in (json.loads(s) for s in Path(background_manifest).read_text().splitlines())]
        if not backgrounds:raise ValueError("empty real background manifest")
    frontend = LogMel()
    features = {}
    rng = np.random.default_rng(123)
    for i, r in enumerate(records):
        audio = read_audio(r["path"])
        versions = [audio]
        if r["split"] == "train":
            if backgrounds:
                from .real_augment import real_augment
                versions += [real_augment(audio, backgrounds[rng.integers(len(backgrounds))], rng)
                             for _ in range(background_variants)]
            else:
                versions += [augment(audio, rng)]
        features[r["id"]] = [frontend(torch.from_numpy(x))[::2].to(torch.float16 if half else torch.float32).contiguous() for x in versions]
        if (i+1) % 200 == 0:
            print(json.dumps({"cached": i+1}), flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest_hash": manifest_hash,"feature_version":version,"features": features}, cache)
    return features

def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a):
        new = [i+1]
        for j, y in enumerate(b):
            new.append(min(new[-1]+1, row[j+1]+1, row[j]+(x != y)))
        row = new
    return row[-1]

def collapse(ids):
    return [x for i, x in enumerate(ids) if x != 0 and (i == 0 or x != ids[i-1])]

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.jsonl")
    p.add_argument("--out", default="artifacts/prototype")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--causal-layers",type=int,default=0)
    p.add_argument("--dropout",type=float,default=0.)
    p.add_argument("--bucket-size",type=int,default=0)
    p.add_argument("--cosine",action="store_true")
    p.add_argument("--patience",type=int,default=0)
    p.add_argument("--cache-only",action="store_true")
    p.add_argument("--feature-half",action="store_true")
    p.add_argument("--feature-cache")
    p.add_argument("--background-manifest")
    p.add_argument("--background-variants",type=int,default=3)
    p.add_argument("--prepared-feature-version")
    p.add_argument("--deletion-weight",type=float,default=0.)
    p.add_argument("--deletion-margin",type=float,default=2.)
    p.add_argument("--alignment-weight",type=float,default=0.)
    p.add_argument("--warmstart")
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--resume")
    p.add_argument("--max-seconds", type=float, default=2400)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    raw = Path(args.manifest).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records = [json.loads(line) for line in raw.decode().splitlines()]
    audit(records)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vocab = {c: i+1 for i, c in enumerate(sorted({c for r in records if r["split"] == "train" for c in r["phones"]}))}
    train = [r for r in records if r["split"] == "train"]
    dev = [r for r in records if r["split"] == "dev" and set(r["phones"]) <= set(vocab)]
    feats = cache_features(records,Path(args.feature_cache) if args.feature_cache else out / "features.pt",digest,args.feature_half,args.background_manifest,args.background_variants,args.prepared_feature_version)
    if args.cache_only:
        print(json.dumps({"cached_records":len(feats),"manifest_hash":digest}),flush=True)
        return
    model = AcousticModel(len(vocab)+1,args.hidden,args.causal_layers,args.dropout).to(args.device)
    parent_hash=None
    if args.warmstart:
        parent=torch.load(args.warmstart,map_location=args.device,weights_only=False)
        if parent["vocab"]!=vocab:raise ValueError("warmstart vocabulary mismatch")
        model.load_state_dict(parent["model"])
        parent_hash=hashlib.sha256(Path(args.warmstart).read_bytes()).hexdigest()
    alignments={}
    if args.alignment_weight:
        for i,r in enumerate(train):
            alignments[r["id"]]=frame_targets(r,vocab,len(feats[r["id"]][0]))
            if (i+1)%2000==0:print(json.dumps({"frame_alignments":i+1}),flush=True)
        valid=sum(int((v!=-100).sum()) for v in alignments.values());total=sum(len(v) for v in alignments.values())
        print(json.dumps({"alignment_frame_coverage":valid/total}),flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*.05) if args.cosine else None
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    raw_ctc = torch.nn.CTCLoss(blank=0,reduction="none",zero_infinity=False)
    start, best, step, stale = 0, float("inf"), 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        if ck["manifest_hash"] != digest or ck["vocab"] != vocab:
            raise ValueError("resume manifest/vocabulary mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        start, best, step = ck["epoch"], ck["best"], ck["step"]
        stale=ck.get("stale",0)
        if scheduler is not None and ck.get("scheduler") is not None:scheduler.load_state_dict(ck["scheduler"])
        if "training_order" in ck:
            by_id={r["id"]:r for r in train}
            train=[by_id[k] for k in ck["training_order"]]
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if args.device.startswith("cuda") and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])

    def batch(rows, training):
        xs, ys, aux = [], [], []
        for r in rows:
            choices = feats[r["id"]]
            x = choices[random.randrange(len(choices)) if training else 0].float().clone()
            if training and random.random() < .5:
                f = random.randrange(36); x[:, f:f+4] = 0
                t = random.randrange(max(1, len(x)-8)); x[t:t+8] = 0
            xs.append(x); ys.append(torch.tensor([vocab[c] for c in r["phones"]],dtype=torch.long))
            aux.append(alignments[r["id"]] if training and args.alignment_weight else torch.full((len(x),),-100,dtype=torch.long))
        return (pad_sequence(xs, batch_first=True).to(args.device), torch.cat(ys).to(args.device),
                torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys]),
                pad_sequence(aux,batch_first=True,padding_value=-100).to(args.device))

    started = time.monotonic()
    for epoch in range(start, args.epochs):
        model.train(); random.shuffle(train)
        if args.bucket_size:
            train=[r for i in range(0,len(train),args.bucket_size) for r in sorted(train[i:i+args.bucket_size],key=lambda r:len(feats[r["id"]][0]))]
        losses = [];aux_losses=[]
        for i in range(0, len(train), args.batch):
            x, y, il, tl, aux = batch(train[i:i+args.batch], True)
            lp, _ = model(x)
            if args.deletion_weight:
                raw_loss=raw_ctc(lp.transpose(0,1),y,il,tl)
                scales=torch.where(tl>0,tl,il).to(args.device)
                weights=torch.where(tl>0,1.,.25).to(args.device)
                ctc_loss=(raw_loss/scales*weights).mean()
                negatives=[];negative_lengths=[];offset=0
                for length in tl.tolist():
                    seq=y[offset:offset+length];offset+=length
                    if length>1:seq=seq[1:] if step%2 else seq[:-1]
                    negatives.append(seq);negative_lengths.append(len(seq))
                negative_loss=raw_ctc(lp.transpose(0,1),torch.cat(negatives),il,torch.tensor(negative_lengths))
                valid=(tl>1).to(args.device)
                deletion_loss=(args.deletion_margin+raw_loss-negative_loss).clamp_min(0)[valid].mean() if valid.any() else lp.new_zeros(())
            else:
                ctc_loss = ctc(lp.transpose(0,1), y, il, tl)
                deletion_loss=lp.new_zeros(())
            aux_loss=torch.nn.functional.nll_loss(lp.transpose(1,2),aux,ignore_index=-100) if args.alignment_weight and (aux!=-100).any() else lp.new_zeros(())
            loss=ctc_loss+args.alignment_weight*aux_loss+args.deletion_weight*deletion_loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite CTC loss; check target/audio lengths")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step(); step += 1; losses.append(ctc_loss.item());aux_losses.append(aux_loss.item())
        model.eval(); dv, edits, tokens = [], 0, 0
        with torch.inference_mode():
            for i in range(0, len(dev), args.batch):
                rows = dev[i:i+args.batch]
                x, y, il, tl, _ = batch(rows, False)
                lp, _ = model(x)
                dv.append(ctc(lp.transpose(0,1), y, il, tl).item())
                for j,r in enumerate(rows):
                    target = [vocab[c] for c in r["phones"]]
                    pred = collapse(lp[j,:il[j]].argmax(-1).cpu().tolist())
                    edits += edit_distance(target,pred); tokens += len(target)
        validation = float(np.mean(dv)) if dv else float(np.mean(losses))
        is_best = validation < best; best = min(best, validation)
        stale=0 if is_best else stale+1
        if scheduler is not None:scheduler.step()
        ck = {"model":model.state_dict(), "optimizer":opt.state_dict(), "vocab":vocab,
              "hidden":args.hidden, "epoch":epoch+1, "step":step, "best":best,
              "causal_layers":args.causal_layers,"dropout":args.dropout,"stale":stale,
              "scheduler":scheduler.state_dict() if scheduler is not None else None,
              "parent_checkpoint_sha256":parent_hash,
              "manifest_hash":digest, "python_rng":random.getstate(), "numpy_rng":np.random.get_state(),
              "torch_rng":torch.get_rng_state(), "training_order":[r["id"] for r in train],
              "cuda_rng":torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None,
              "args":vars(args), "parameters":sum(p.numel() for p in model.parameters())}
        tmp = out / "checkpoint.tmp"; torch.save(ck,tmp); tmp.replace(out / "last.pt")
        if is_best:
            torch.save(ck,out / "best.pt")
        metrics = {"epoch":epoch+1,"step":step,"train_ctc":float(np.mean(losses)),
                   "train_frame_nll":float(np.mean(aux_losses)),
                   "dev_ctc":validation,"dev_phone_error_rate":edits/tokens if tokens else None,
                   "dev_unsupported":sum(r["split"]=="dev" for r in records)-len(dev),
                   "wall_seconds":time.monotonic()-started,"device":args.device,"lr":opt.param_groups[0]["lr"]}
        with (out / "train.jsonl").open("a") as f:
            f.write(json.dumps(metrics)+"\n")
        print(json.dumps(metrics),flush=True)
        if time.monotonic()-started >= args.max_seconds or (args.patience and stale>=args.patience):
            break

if __name__ == "__main__":
    main()
