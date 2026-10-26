"""Train only an own raw-mel correction; freeze the complete phonetic base."""
import argparse,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .residual_model import make_model,assert_frozen,parameter_digest,KIND
from .train_foreground import sha
from .train import audit,collapse,edit_distance
from .train_power import normalize,mix_power,music_crop,cached_speech
from .train_hard_music import negative_blank_loss

def clean_anchor_loss(student,teacher,lengths,temperature=2.):
    """Own clean-to-clean posterior KL, emphasizing teacher phone frames."""
    if student.shape!=teacher.shape or temperature<=0:raise ValueError('Clean anchor contract')
    if torch.any(lengths<=0) or torch.any(lengths>student.shape[1]):raise ValueError('Clean anchor lengths')
    target=teacher.detach();s=torch.log_softmax(student/temperature,-1);t=torch.log_softmax(target/temperature,-1)
    frame=(t.exp()*(t-s)).sum(-1)*temperature**2
    mask=torch.arange(frame.shape[1],device=frame.device)[None,:]<lengths.to(frame.device)[:,None]
    weights=(.25+1-target[...,0].exp())*mask
    return ((frame*weights).sum(-1)/weights.sum(-1)).clamp_min(0.)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--warmstart',required=True);p.add_argument('--out',required=True)
    p.add_argument('--anchor-weight',type=float,required=True);p.add_argument('--residual-hidden',type=int,default=192)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=96);p.add_argument('--max-steps',type=int,default=6000)
    p.add_argument('--max-seconds',type=float,default=1200);p.add_argument('--epochs',type=int,default=60);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke-rows',type=int,default=0)
    a=p.parse_args(argv)
    if a.anchor_weight<=0:raise ValueError('Positive clean anchor required')
    torch.set_num_threads(a.threads);random.seed(404);np.random.seed(404);torch.manual_seed(404)
    manifest=Path('data/round4-onset/manifest.jsonl');manifest_hash=sha(manifest)
    rows=list(map(json.loads,manifest.read_text().splitlines()));audit(rows)
    raw=torch.load('data/round4/raw_features.pt',map_location='cpu',weights_only=True,mmap=True)
    delta=torch.load('data/round4-onset/delta.pt',map_location='cpu',weights_only=True,mmap=True)
    if raw['manifest_hash']!=delta['base_manifest_hash'] or delta['manifest_hash']!=manifest_hash or raw['feature_version']!=delta['feature_version'] or raw['feature_version']!='logmel64-power-v1':raise ValueError('Feature provenance')
    feats=dict(raw['features']);feats.update(delta['features'])
    extra=torch.load('data/round5-musicplus/backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    negative=torch.load('data/round8/room_backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    if extra['version']!='round5-training-backgrounds-v1' or negative['version']!='room-negative-waveform-v1':raise ValueError('Background versions')
    if sha('data/round4/raw_features.pt')!=negative['dry_cache_sha256'] or sha('data/round5-musicplus/backgrounds.pt')!=negative['extra_cache_sha256']:raise ValueError('Background provenance')
    backgrounds=list(raw['backgrounds'])+list(extra['backgrounds']);room_bg=negative['backgrounds']
    if len(backgrounds)!=84 or len(room_bg)!=84 or negative['origins'][24:]!=extra['origins'] or any(x.shape!=y.shape for x,y in zip(backgrounds,room_bg)):raise ValueError('Background identity/timing')
    music=torch.load('data/round5-waveform/features.pt',map_location='cpu',weights_only=True,mmap=True)
    room=torch.load('data/round7-room/features.pt',map_location='cpu',weights_only=True,mmap=True)
    train_ids={r['id'] for r in rows if r['split']=='train' and r['phones']}
    for content,version in [(music,'music-waveform-v1'),(room,'room-waveform-v2')]:
        if content['manifest_hash']!=manifest_hash or content['feature_version']!=raw['feature_version'] or content['augmentation_version']!=version or set(content['features'])!=train_ids:raise ValueError('Speech augmentation provenance')
    if set(room['mixed_features'])!=train_ids:raise ValueError('Room/music pairing')
    parent=torch.load(a.warmstart,map_location='cpu',weights_only=True);parent_hash=sha(a.warmstart)
    vocab={c:i+1 for i,c in enumerate(sorted({c for r in rows if r['split']=='train' for c in r['phones']}))}
    if parent['vocab']!=vocab or parent['hidden']!=384 or parent['causal_layers']!=3 or parent['bins']!=64:raise ValueError('Own frozen parent required')
    model=make_model(parent,a.device,residual_hidden=a.residual_hidden);assert_frozen(model,parent)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev' and set(r['phones'])<=set(vocab)]
    if a.smoke_rows:train=train[:a.smoke_rows];dev=dev[:8]
    targets={r['id']:torch.tensor([vocab[c] for c in r['phones']],dtype=torch.long) for r in train+dev}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() or (out/'plan.json').exists():raise RuntimeError('Preserve experiment')
    plan={'args':vars(a),'manifest_hash':manifest_hash,'model_kind':KIND,'parent_checkpoint_sha256':parent_hash,
        'base_frozen_exact':True,'residual_head_initialization':'zero; old predictions up to final logprob renormalization rounding',
        'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),'clean_anchor_weight':a.anchor_weight,
        'clean_anchor':'own clean base posterior vs own clean residual student, T2, frameweight .25+(1-p_blank); no external teacher',
        'negative_probability':.5,'negative_tail_fraction':.02,'negative_blank_weight':.5,'room_probability_both_classes':.5,
        'speech_mix_probability':.5,'no_query_dependent_training':True,'no_external_teacher_labels_or_parameters':True}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    started=time.monotonic();clean_cache={};cache_bytes=0
    ordered=sorted([r for r in train if r['phones']],key=lambda r:len(feats[r['id']]))
    with torch.no_grad():
        for i in range(0,len(ordered),a.batch):
            rs=ordered[i:i+a.batch];xs=[normalize(feats[r['id']].float()) for r in rs]
            lp,_=model.base(pad_sequence(xs,batch_first=True).to(a.device))
            for j,(r,x) in enumerate(zip(rs,xs)):
                value=lp[j,:len(x)].detach().clone();clean_cache[r['id']]=value;cache_bytes+=value.numel()*value.element_size()
            if cache_bytes>4*1024**3:raise RuntimeError('Bounded own clean teacher cache exceeded')
    print(json.dumps({'clean_teacher_cache_ready':True,'rows':len(clean_cache),'bytes':cache_bytes,'storage':'ephemeral device RAM only','seconds':time.monotonic()-started}),flush=True)
    params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-4)
    ctc=torch.nn.CTCLoss(reduction='none',zero_infinity=False)
    def batch(rs,training):
        xs=[];ys=[];clean=[];teachers=[];positive_lengths=[]
        for r in rs:
            pure=training and random.random()<.5;y=torch.empty(0,dtype=torch.long) if pure else targets[r['id']]
            selected_room=training and random.random()<.5;bg=room_bg if selected_room else backgrounds
            if pure:x=music_crop(bg)
            else:
                base=feats[r['id']].float();rf=room['features'][r['id']] if training and selected_room and len(y) else None
                x=cached_speech(base,room=rf)
                if training and len(y) and random.random()<.5:
                    if random.random()<.5:x=cached_speech(base,room=rf,music=music['features'][r['id']],room_music=room['mixed_features'][r['id']],variant=random.randrange(2))
                    else:
                        noise=random.choice(bg).float();n=len(x)
                        if len(noise)<n:noise=noise.repeat(n//len(noise)+1,1)
                        j=random.randrange(len(noise)-n+1);x=mix_power(x,noise[j:j+n],random.uniform(-2,5) if random.random()<.1 else random.uniform(5,30))
            if training and len(y):
                clean.append(normalize(feats[r['id']].float()));teachers.append(clean_cache[r['id']]);positive_lengths.append(len(clean[-1]))
            x=normalize(x)
            if training and random.random()<.25:
                x=x.clone();f=random.randrange(61);x[:,f:f+4]=0;t=random.randrange(max(1,len(x)-2));x[t:t+2]=0
            xs.append(x);ys.append(y)
        positive=(pad_sequence(clean,batch_first=True).to(a.device),pad_sequence(teachers,batch_first=True),torch.tensor(positive_lengths)) if clean else None
        return (pad_sequence(xs,batch_first=True).to(a.device),torch.cat(ys).to(a.device),torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys]),positive)
    step=0
    for epoch in range(a.epochs):
        random.shuffle(train);order=[r for i in range(0,len(train),768) for r in sorted(train[i:i+768],key=lambda r:len(feats[r['id']]))]
        values=[];model.train()
        for i in range(0,len(order),a.batch):
            lr=a.lr*(.05+.95*(1+math.cos(math.pi*min(step/a.max_steps,1)))/2)
            for group in opt.param_groups:group['lr']=lr
            x,y,il,tl,positive=batch(order[i:i+a.batch],True);lp,_=model(x);mask=(tl>0).to(a.device)
            speech=ctc(lp.transpose(0,1),y,il,tl)/tl.clamp_min(1).to(a.device)
            blank=negative_blank_loss(lp,il,.02);loss=torch.where(mask,speech,.5*blank).mean();anchor=lp.new_zeros(())
            if positive:
                clean,teacher,lengths=positive;correction,_=model.residual(clean)
                student=torch.log_softmax(teacher+correction,-1);anchor=clean_anchor_loss(student,teacher,lengths).sum()/len(x)
                loss=loss+a.anchor_weight*anchor
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,5);opt.step();step+=1
            values.append([loss.item(),speech[mask].detach().mean().item() if mask.any() else 0.,blank[~mask].detach().mean().item() if (~mask).any() else 0.,anchor.item()])
            if step>=a.max_steps:break
        dv=[];edits=tokens=0;model.eval()
        with torch.inference_mode():
            for i in range(0,len(dev),a.batch):
                rs=dev[i:i+a.batch];x,y,il,tl,_=batch(rs,False);lp,_=model(x)
                dv.append((ctc(lp.transpose(0,1),y,il,tl)/torch.where(tl>0,tl,il).to(a.device)).mean().item())
                for j,r in enumerate(rs):
                    truth=targets[r['id']].tolist();pred=collapse(lp[j,:il[j]].argmax(-1).cpu().tolist());edits+=edit_distance(truth,pred);tokens+=len(truth)
        assert_frozen(model,parent)
        ck={k:parent[k] for k in ['vocab','hidden','causal_layers','dropout','bins','feature_version','text_version']}
        ck.update({'model':model.state_dict(),'model_kind':KIND,'residual_hidden':a.residual_hidden,'residual_layers':2,
            'epoch':epoch+1,'step':step,'manifest_hash':manifest_hash,'parent_checkpoint_sha256':parent_hash,'backbone_frozen':True,
            'frozen_base_parameter_sha256':parameter_digest(model.base.state_dict()),'args':vars(a),
            'trainable_parameters':plan['trainable_parameters'],'parameters':sum(v.numel() for v in model.parameters()),'teacher_cache_bytes':cache_bytes})
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        avg=np.mean(values,axis=0);r={'epoch':epoch+1,'step':step,'train_loss':float(avg[0]),'train_speech_ctc':float(avg[1]),'train_music_tail_blank':float(avg[2]),
            'train_clean_anchor':float(avg[3]),'dev_ctc':float(np.mean(dv)),'dev_phone_error_rate':edits/tokens,'base_frozen_exact':True,
            'trainable_parameters':plan['trainable_parameters'],'wall_seconds':time.monotonic()-started,'lr':lr,'device':a.device}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
        if step>=a.max_steps or time.monotonic()-started>=a.max_seconds:break
if __name__=='__main__':main()
