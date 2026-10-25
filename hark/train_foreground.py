"""Train only an own foreground head; preserve every phonetic parameter."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .foreground_model import ForegroundAcousticModel,KIND,CONTRACT_KEY
from .model import checkpoint_parameters
from .train import audit,collapse,edit_distance
from .train_power import normalize,mix_power,music_crop,cached_speech,normalized_ctc_loss

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def foreground_loss(logits,targets,lengths):
    mask=torch.arange(logits.shape[1],device=logits.device)[None,:]<lengths.to(logits.device)[:,None]
    values=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction='none')
    return ((values*mask).sum(-1)/lengths.to(logits.device)).mean()

def make_model(parent,device='cpu'):
    model=ForegroundAcousticModel(len(parent['vocab'])+1,parent['hidden'],parent['causal_layers'],parent['dropout'],parent['bins'])
    missing,unexpected=model.load_state_dict(checkpoint_parameters(parent),strict=False)
    expected={CONTRACT_KEY}|{k for k in model.state_dict() if k.startswith(('foreground_gru.','foreground_head.'))}
    if set(missing)!=expected or unexpected:raise ValueError('Only new own foreground parameters may be missing')
    model.to(device);model.freeze_backbone()
    return model

def assert_frozen(model,parent):
    current=model.state_dict()
    for k,v in checkpoint_parameters(parent).items():
        if not torch.equal(current[k].cpu(),v.cpu()):raise RuntimeError('Frozen phonetic parameter changed: '+k)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--warmstart',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=96)
    p.add_argument('--max-steps',type=int,default=6000);p.add_argument('--max-seconds',type=float,default=1000)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke-rows',type=int,default=0)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    random.seed(404);np.random.seed(404);torch.manual_seed(404)
    manifest=Path('data/round4-onset/manifest.jsonl');manifest_hash=sha(manifest)
    rows=[json.loads(s) for s in manifest.read_text().splitlines()];audit(rows)
    cache=torch.load('data/round4/raw_features.pt',map_location='cpu',weights_only=True,mmap=True)
    delta=torch.load('data/round4-onset/delta.pt',map_location='cpu',weights_only=True,mmap=True)
    if cache['manifest_hash']!=delta['base_manifest_hash'] or delta['manifest_hash']!=manifest_hash:raise ValueError('Delta provenance')
    if cache['feature_version']!=delta['feature_version'] or cache['feature_version']!='logmel64-power-v1':raise ValueError('Frontend provenance')
    feats=dict(cache['features']);feats.update(delta['features'])
    extra=torch.load('data/round5-musicplus/backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    negative=torch.load('data/round8/room_backgrounds.pt',map_location='cpu',weights_only=True,mmap=True)
    if extra['version']!='round5-training-backgrounds-v1' or negative['version']!='room-negative-waveform-v1':raise ValueError('Background version')
    if sha('data/round4/raw_features.pt')!=negative['dry_cache_sha256'] or sha('data/round5-musicplus/backgrounds.pt')!=negative['extra_cache_sha256']:raise ValueError('Background provenance')
    backgrounds=list(cache['backgrounds'])+list(extra['backgrounds']);room_bg=negative['backgrounds']
    if len(backgrounds)!=84 or len(room_bg)!=84 or negative['origins'][24:]!=extra['origins']:raise ValueError('Background order')
    if any(x.shape!=y.shape for x,y in zip(backgrounds,room_bg)):raise ValueError('Background timing')
    music=torch.load('data/round5-waveform/features.pt',map_location='cpu',weights_only=True,mmap=True)
    room=torch.load('data/round7-room/features.pt',map_location='cpu',weights_only=True,mmap=True)
    train_ids={r['id'] for r in rows if r['split']=='train' and r['phones']}
    for content,version in [(music,'music-waveform-v1'),(room,'room-waveform-v2')]:
        if content['manifest_hash']!=manifest_hash or content['feature_version']!=cache['feature_version'] or content['augmentation_version']!=version:raise ValueError('Speech cache provenance')
        if set(content['features'])!=train_ids:raise ValueError('Speech cache keys')
    if set(room['mixed_features'])!=train_ids:raise ValueError('Paired room cache keys')
    parent=torch.load(a.warmstart,map_location='cpu',weights_only=True);parent_hash=sha(a.warmstart)
    vocab={c:i+1 for i,c in enumerate(sorted({c for r in rows if r['split']=='train' for c in r['phones']}))}
    if parent['vocab']!=vocab or parent['hidden']!=384 or parent['causal_layers']!=3 or parent['bins']!=64:raise ValueError('Own parent contract')
    model=make_model(parent,a.device);assert_frozen(model,parent)
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev' and set(r['phones'])<=set(vocab)]
    if a.smoke_rows:train=train[:a.smoke_rows];dev=dev[:8]
    targets={r['id']:torch.tensor([vocab[c] for c in r['phones']],dtype=torch.long) for r in train+dev}
    trainable=[x for x in model.parameters() if x.requires_grad]
    if sum(x.numel() for x in trainable)!=86465:raise RuntimeError('Unexpected trainable parameter count')
    opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=1e-4)
    ctc=torch.nn.CTCLoss(reduction='none',zero_infinity=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() or (out/'best.pt').exists():raise RuntimeError('Preserve trained experiment')
    (out/'plan.json').write_text(json.dumps({'args':vars(a),'manifest_hash':manifest_hash,'parent_sha256':parent_hash,
        'model_kind':KIND,'backbone_frozen':True,'trainable_parameters':86465,'negative_probability':.5,
        'room_probability_both_classes':.5,'speech_mix_probability':.5,'foreground_loss_weight':.25,
        'foreground_targets':'zero pure music and preludes; one original speech interval; padding ignored',
        'no_query_dependent_training':True},indent=2)+'\n')
    def batch(rs,training):
        xs=[];ys=[];labels=[]
        for r in rs:
            pure=training and random.random()<.5
            y=torch.empty(0,dtype=torch.long) if pure else targets[r['id']]
            room_selected=training and random.random()<.5
            bg=room_bg if room_selected else backgrounds
            if pure:x=music_crop(bg);label=torch.zeros(len(x))
            else:
                base=feats[r['id']].float()
                rfeat=room['features'][r['id']] if training and room_selected and len(y) else None
                x=cached_speech(base,room=rfeat);pre=0
                if training and len(y) and random.random()<.5:
                    noise=random.choice(bg).float();n=len(x)
                    if len(noise)<n+50:noise=noise.repeat((n+50)//len(noise)+1,1)
                    j=random.randrange(len(noise)-n+1)
                    if random.random()<.5:
                        x=cached_speech(base,room=rfeat,music=music['features'][r['id']],room_music=room['mixed_features'][r['id']],variant=random.randrange(2))
                    else:x=mix_power(x,noise[j:j+n],random.uniform(-2,5) if random.random()<.1 else random.uniform(5,30))
                    if random.random()<.5:
                        pre=random.randint(10,50);x=torch.cat([noise[j-pre:j] if j>=pre else noise[:pre],x])
                label=torch.ones(len(x)) if len(y) else torch.zeros(len(x));label[:pre]=0
            x=normalize(x)
            if training and random.random()<.25:
                x=x.clone();f=random.randrange(61);x[:,f:f+4]=0
                t=random.randrange(max(1,len(x)-2));x[t:t+2]=0
            xs.append(x);ys.append(y);labels.append(label)
        return (pad_sequence(xs,batch_first=True).to(a.device),torch.cat(ys).to(a.device),
                torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys]),pad_sequence(labels,batch_first=True).to(a.device))
    started=time.monotonic();step=0;best=float('inf')
    for epoch in range(a.epochs):
        random.shuffle(train);order=[r for i in range(0,len(train),768) for r in sorted(train[i:i+768],key=lambda r:len(feats[r['id']]))]
        losses=[];ctcs=[];bces=[];model.eval()
        # cuDNN requires training mode for the new GRU's backward pass.
        # The frozen backbone stays in eval mode; this single-layer GRU has no dropout.
        model.foreground_gru.train();model.foreground_head.train()
        for i in range(0,len(order),a.batch):
            lr=a.lr*(.05+.95*(1+math.cos(math.pi*min(step/a.max_steps,1)))/2)
            for group in opt.param_groups:group['lr']=lr
            x,y,il,tl,fg=batch(order[i:i+a.batch],True);lp,_=model(x)
            phone_loss=normalized_ctc_loss(ctc,lp,y,il,tl,1.)
            aux=foreground_loss(model.foreground_logits,fg,il);loss=phone_loss+.25*aux
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(trainable,5);opt.step();step+=1
            losses.append(loss.item());ctcs.append(phone_loss.item());bces.append(aux.item())
            if step>=a.max_steps:break
        dv=[];edits=tokens=0;model.eval()
        with torch.inference_mode():
            for i in range(0,len(dev),a.batch):
                rs=dev[i:i+a.batch];x,y,il,tl,fg=batch(rs,False);lp,_=model(x)
                dv.append(normalized_ctc_loss(ctc,lp,y,il,tl,1.).item())
                for j,r in enumerate(rs):
                    truth=targets[r['id']].tolist();pred=collapse(lp[j,:il[j]].argmax(-1).cpu().tolist())
                    edits+=edit_distance(truth,pred);tokens+=len(truth)
        val=float(np.mean(dv));improved=val<best;best=min(best,val);assert_frozen(model,parent)
        ck={k:parent[k] for k in ['vocab','hidden','bins','causal_layers','dropout','feature_version','text_version']}
        ck.update({'model':model.state_dict(),'model_kind':KIND,'foreground_hidden':64,'epoch':epoch+1,'step':step,
            'manifest_hash':manifest_hash,'parent_checkpoint_sha256':parent_hash,'backbone_frozen':True,
            'trainable_parameters':86465,'parameters':sum(x.numel() for x in model.parameters()),'best':best,'args':vars(a)})
        for name in (['best','last'] if improved else ['last']):
            torch.save(ck,out/(name+'.tmp'));(out/(name+'.tmp')).replace(out/(name+'.pt'))
        metrics={'epoch':epoch+1,'step':step,'train_loss':float(np.mean(losses)),'train_ctc':float(np.mean(ctcs)),
            'train_foreground_bce':float(np.mean(bces)),'dev_ctc':val,'dev_phone_error_rate':edits/tokens,
            'frozen_backbone_exact':True,'wall_seconds':time.monotonic()-started,'lr':lr,'device':a.device}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if step>=a.max_steps or time.monotonic()-started>=a.max_seconds:break
if __name__=='__main__':main()
