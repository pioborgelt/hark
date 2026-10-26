"""Own-parent consistency and tail-weighted music blank training.

The teacher is our own frozen parent, never an external ASR model. Text labels
remain the original training IPA; there are no enrolled-call-dependent losses.
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .model import AcousticModel, checkpoint_parameters
from .train import audit, collapse, edit_distance
from .train_power import normalize, mix_power, music_crop, cached_speech
from .train_foreground import sha


def negative_blank_loss(log_probs, lengths, fraction):
    """Per-example blank NLL on the hardest valid fraction of frames."""
    if not 0 < fraction <= 1:
        raise ValueError('Invalid tail fraction')
    if len(log_probs) != len(lengths):
        raise ValueError('Length count')
    values=[]
    for row,n in zip(log_probs,lengths):
        n=int(n)
        if not 0 < n <= len(row):raise ValueError('Invalid valid length')
        k=max(1,math.ceil(n*fraction))
        values.append((-row[:n,0]).topk(k).values.mean())
    return torch.stack(values)


def own_parent_kl(student, teacher, lengths, temperature=2.):
    """Aligned, padding-masked KL, with no gradient to the own teacher."""
    if student.shape != teacher.shape or temperature <= 0:raise ValueError('KL contract')
    s=torch.log_softmax(student/temperature,-1)
    t=torch.log_softmax(teacher.detach()/temperature,-1)
    frame=(t.exp()*(t-s)).sum(-1)*temperature**2
    mask=torch.arange(frame.shape[1],device=frame.device)[None,:]<lengths.to(frame.device)[:,None]
    if torch.any(lengths<=0) or torch.any(lengths>frame.shape[1]):raise ValueError('Invalid KL length')
    return (frame*mask).sum(-1)/lengths.to(frame.device)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--warmstart',required=True);p.add_argument('--out',required=True)
    p.add_argument('--tail-fraction',type=float,required=True);p.add_argument('--blank-weight',type=float,required=True)
    p.add_argument('--anchor-weight',type=float,default=.5)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=96)
    p.add_argument('--max-steps',type=int,default=6000);p.add_argument('--max-seconds',type=float,default=1400)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--lr',type=float,default=.00015)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke-rows',type=int,default=0)
    a=p.parse_args(argv)
    if not 0<a.tail_fraction<=1 or a.blank_weight<=0 or a.anchor_weight<0:raise ValueError('Loss settings')
    torch.set_num_threads(a.threads);random.seed(404);np.random.seed(404);torch.manual_seed(404)
    manifest=Path('data/round4-onset/manifest.jsonl');manifest_hash=sha(manifest)
    rows=[json.loads(s) for s in manifest.read_text().splitlines()];audit(rows)
    cache=torch.load('data/round4/raw_features.pt',map_location='cpu',weights_only=True,mmap=True)
    delta=torch.load('data/round4-onset/delta.pt',map_location='cpu',weights_only=True,mmap=True)
    if cache['manifest_hash']!=delta['base_manifest_hash'] or delta['manifest_hash']!=manifest_hash:raise ValueError('Delta provenance')
    if cache['feature_version']!=delta['feature_version'] or cache['feature_version']!='logmel64-power-v1':raise ValueError('Frontend provenance')
    feats=dict(cache['features']);feats.update(delta['features'])
    extra=torch.load('data/round5-musicplus/backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    negative=torch.load('data/round8/room_backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    if extra['version']!='round5-training-backgrounds-v1' or negative['version']!='room-negative-waveform-v1':raise ValueError('Background versions')
    if sha('data/round4/raw_features.pt')!=negative['dry_cache_sha256'] or sha('data/round5-musicplus/backgrounds.pt')!=negative['extra_cache_sha256']:raise ValueError('Background provenance')
    backgrounds=list(cache['backgrounds'])+list(extra['backgrounds']);room_bg=negative['backgrounds']
    if len(backgrounds)!=84 or len(room_bg)!=84 or negative['origins'][24:]!=extra['origins']:raise ValueError('Background order')
    if any(x.shape!=y.shape for x,y in zip(backgrounds,room_bg)):raise ValueError('Room timing')
    music=torch.load('data/round5-waveform/features.pt',map_location='cpu',weights_only=True,mmap=True)
    room=torch.load('data/round7-room/features.pt',map_location='cpu',weights_only=True,mmap=True)
    train_ids={r['id'] for r in rows if r['split']=='train' and r['phones']}
    for content,version in [(music,'music-waveform-v1'),(room,'room-waveform-v2')]:
        if content['manifest_hash']!=manifest_hash or content['feature_version']!=cache['feature_version'] or content['augmentation_version']!=version:raise ValueError('Speech cache provenance')
        if set(content['features'])!=train_ids:raise ValueError('Speech cache keys')
    if set(room['mixed_features'])!=train_ids:raise ValueError('Room/music cache keys')
    parent=torch.load(a.warmstart,map_location='cpu',weights_only=True);parent_hash=sha(a.warmstart)
    vocab={c:i+1 for i,c in enumerate(sorted({c for r in rows if r['split']=='train' for c in r['phones']}))}
    if parent['vocab']!=vocab or parent['hidden']!=384 or parent['causal_layers']!=3 or parent['bins']!=64:raise ValueError('Own parent contract')
    def fresh_model():
        m=AcousticModel(len(vocab)+1,384,3,.1,64).to(a.device)
        m.load_state_dict(checkpoint_parameters(parent));return m
    model=fresh_model();teacher=fresh_model().eval();teacher.requires_grad_(False)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev' and set(r['phones'])<=set(vocab)]
    if a.smoke_rows:train=train[:a.smoke_rows];dev=dev[:8]
    targets={r['id']:torch.tensor([vocab[c] for c in r['phones']],dtype=torch.long) for r in train+dev}
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    ctc=torch.nn.CTCLoss(reduction='none',zero_infinity=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if any((out/name).exists() for name in ['last.pt','plan.json']):raise RuntimeError('Preserve experiment')
    (out/'plan.json').write_text(json.dumps({'args':vars(a),'manifest_hash':manifest_hash,'parent_checkpoint_sha256':parent_hash,
        'own_teacher_sha256':parent_hash,'teacher_temperature':2.,'teacher_frozen':True,'negative_probability':.5,
        'room_probability_both_classes':.5,'speech_mix_probability':.5,'aligned_augmentation_no_prelude':True,
        'no_query_dependent_training':True,'no_external_teacher_labels_or_parameters':True},indent=2)+'\n')
    def batch(rs,training):
        xs=[];clean=[];ys=[]
        for r in rs:
            pure=training and random.random()<.5
            y=torch.empty(0,dtype=torch.long) if pure else targets[r['id']]
            selected_room=training and random.random()<.5
            bg=room_bg if selected_room else backgrounds
            if pure:x=music_crop(bg);base=x
            else:
                base=feats[r['id']].float();rfeat=room['features'][r['id']] if training and selected_room and len(y) else None
                x=cached_speech(base,room=rfeat)
                if training and len(y) and random.random()<.5:
                    if random.random()<.5:
                        x=cached_speech(base,room=rfeat,music=music['features'][r['id']],room_music=room['mixed_features'][r['id']],variant=random.randrange(2))
                    else:
                        noise=random.choice(bg).float();n=len(x)
                        if len(noise)<n:noise=noise.repeat(n//len(noise)+1,1)
                        j=random.randrange(len(noise)-n+1)
                        x=mix_power(x,noise[j:j+n],random.uniform(-2,5) if random.random()<.1 else random.uniform(5,30))
            if len(base)!=len(x):raise RuntimeError('Teacher alignment')
            clean.append(normalize(base));x=normalize(x)
            if training and random.random()<.25:
                x=x.clone();f=random.randrange(61);x[:,f:f+4]=0
                t=random.randrange(max(1,len(x)-2));x[t:t+2]=0
            xs.append(x);ys.append(y)
        return (pad_sequence(xs,batch_first=True).to(a.device),torch.cat(ys).to(a.device),
                torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys]),pad_sequence(clean,batch_first=True).to(a.device))
    started=time.monotonic();step=0
    for epoch in range(a.epochs):
        random.shuffle(train);order=[r for i in range(0,len(train),768) for r in sorted(train[i:i+768],key=lambda r:len(feats[r['id']]))]
        metrics_list=[];model.train()
        for i in range(0,len(order),a.batch):
            lr=a.lr*(.05+.95*(1+math.cos(math.pi*min(step/a.max_steps,1)))/2)
            for group in opt.param_groups:group['lr']=lr
            x,y,il,tl,clean=batch(order[i:i+a.batch],True);lp,_=model(x)
            positive=(tl>0).to(a.device)
            raw=ctc(lp.transpose(0,1),y,il,tl)
            speech=raw/tl.clamp_min(1).to(a.device)
            blank=negative_blank_loss(lp,il,a.tail_fraction)
            with torch.no_grad():teacher_lp,_=teacher(clean)
            kl=own_parent_kl(lp,teacher_lp,il)
            per=torch.where(positive,speech+a.anchor_weight*kl,a.blank_weight*blank)
            loss=per.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1
            metrics_list.append([loss.item(),float(speech[positive].mean()) if positive.any() else 0.,float(blank[~positive].mean()) if (~positive).any() else 0.,float(kl[positive].mean()) if positive.any() else 0.])
            if step>=a.max_steps:break
        dv=[];edits=tokens=0;model.eval()
        with torch.inference_mode():
            for i in range(0,len(dev),a.batch):
                rs=dev[i:i+a.batch];x,y,il,tl,_=batch(rs,False);lp,_=model(x)
                dv.append((ctc(lp.transpose(0,1),y,il,tl)/torch.where(tl>0,tl,il).to(a.device)).mean().item())
                for j,r in enumerate(rs):
                    truth=targets[r['id']].tolist();pred=collapse(lp[j,:il[j]].argmax(-1).cpu().tolist())
                    edits+=edit_distance(truth,pred);tokens+=len(truth)
        if any(not torch.equal(v.cpu(),checkpoint_parameters(parent)[k]) for k,v in teacher.state_dict().items()):raise RuntimeError('Own teacher changed')
        ck={k:parent[k] for k in ['vocab','hidden','bins','causal_layers','dropout','feature_version','text_version']}
        ck.update({'model':model.state_dict(),'epoch':epoch+1,'step':step,'manifest_hash':manifest_hash,'parent_checkpoint_sha256':parent_hash,
                   'own_teacher_sha256':parent_hash,'parameters':sum(x.numel() for x in model.parameters()),'args':vars(a)})
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        avg=np.mean(metrics_list,axis=0)
        metrics={'epoch':epoch+1,'step':step,'train_loss':float(avg[0]),'train_speech_ctc':float(avg[1]),'train_music_blank':float(avg[2]),
                 'train_own_parent_kl':float(avg[3]),'dev_ctc':float(np.mean(dv)),'dev_phone_error_rate':edits/tokens,
                 'own_teacher_frozen_exact':True,'wall_seconds':time.monotonic()-started,'lr':lr,'device':a.device}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if step>=a.max_steps or time.monotonic()-started>=a.max_seconds:break

if __name__=='__main__':main()
