"""Own causal encoder study with fresh power-domain music per presentation.

Mixing mel powers approximates uncorrelated waveform addition. Evaluation uses
actual waveform mixtures, never this approximation. No pretrained parameters.
"""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .model import AcousticModel, checkpoint_parameters
from .train import audit, collapse, edit_distance


def normalize(power_log):
    return (power_log-power_log.mean(-1,keepdim=True))/power_log.std(-1,keepdim=True).clamp_min(.1)


def mix_power(clean, background, snr):
    """Set global power SNR before addition; inputs are log mel powers."""
    if clean.shape != background.shape:raise ValueError('power shape mismatch')
    shift=torch.logsumexp(clean.flatten(),0)-torch.logsumexp(background.flatten(),0)-snr*math.log(10)/10
    return torch.logaddexp(clean,background+shift)


def music_crop(backgrounds, minimum=100, maximum=400):
    """A fresh contiguous 2..8 second music window, with no speech labels."""
    noise=random.choice(backgrounds)
    length=random.randint(minimum,maximum)
    if len(noise)<length:noise=noise.repeat(length//len(noise)+1,1)
    start=random.randrange(len(noise)-length+1)
    return noise[start:start+length].float()


def cached_speech(base, room=None, music=None, room_music=None, variant=None):
    """Keep the chosen room transformation when selecting a waveform mixture."""
    if variant is None:
        value = base if room is None else room
    else:
        if room is not None and room_music is None:
            raise ValueError('Room and waveform mixtures require paired room music')
        value = (music if room is None else room_music)[variant]
    if value.shape != base.shape:raise ValueError('Cached augmentation changed speech timing or bins')
    return value.float()


def normalized_ctc_loss(ctc,lp,y,input_lengths,target_lengths,blank_weight=.25):
    losses=ctc(lp.transpose(0,1),y,input_lengths,target_lengths)
    scale=torch.where(target_lengths>0,target_lengths,input_lengths).to(lp.device)
    weight=torch.where(target_lengths>0,1.,blank_weight).to(lp.device)
    return (losses/scale*weight).mean()


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/round4/manifest.jsonl');p.add_argument('--features',default='data/round4/raw_features.pt')
    p.add_argument('--feature-delta')
    p.add_argument('--room-features')
    p.add_argument('--room-probability',type=float,default=.5)
    p.add_argument('--waveform-features')
    p.add_argument('--extra-backgrounds')
    p.add_argument('--negative-probability',type=float,default=0.)
    p.add_argument('--blank-loss-weight',type=float,default=.25)
    p.add_argument('--speech-mix-probability',type=float,default=.75)
    p.add_argument('--out',required=True);p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--causal-layers',type=int,default=2);p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch',type=int,default=96);p.add_argument('--max-seconds',type=float,default=1000)
    p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--lr',type=float,default=.0015)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--smoke-rows',type=int,default=0)
    p.add_argument('--warmstart')
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    if not 0<=a.negative_probability<1 or a.blank_loss_weight<=0:raise ValueError('invalid negative training settings')
    if not 0<=a.speech_mix_probability<=1:raise ValueError('invalid speech mixing probability')
    if not 0<=a.room_probability<=1:raise ValueError('invalid room probability')
    random.seed(404);np.random.seed(404);torch.manual_seed(404)
    raw=Path(a.manifest).read_bytes();sha=hashlib.sha256(raw).hexdigest();rows=[json.loads(s) for s in raw.decode().splitlines()];audit(rows)
    cache=torch.load(a.features,map_location='cpu',weights_only=False)
    if a.feature_delta:
        delta=torch.load(a.feature_delta,map_location='cpu',weights_only=False)
        if cache['manifest_hash']!=delta['base_manifest_hash'] or cache['feature_version']!=delta['feature_version']:raise ValueError('feature delta/base mismatch')
        cache['features'].update(delta['features']);cache['manifest_hash']=delta['manifest_hash']
    if cache['manifest_hash']!=sha or cache['feature_version']!='logmel64-power-v1':raise ValueError('cache provenance mismatch')
    feats=cache['features'];bg=list(cache['backgrounds']);room={};room_music={};room_hash=None;extra_hash=None;waveform={};waveform_hash=None
    if a.extra_backgrounds:
        extra=torch.load(a.extra_backgrounds,map_location='cpu',weights_only=False)
        if extra['version']!='round5-training-backgrounds-v1' or not extra['backgrounds']:raise ValueError('invalid training backgrounds')
        if len(extra['backgrounds'])!=len(extra['origins']):raise ValueError('background origin count mismatch')
        bg.extend(extra['backgrounds']);extra_hash=hashlib.sha256(Path(a.extra_backgrounds).read_bytes()).hexdigest()
    if a.waveform_features:
        mixed=torch.load(a.waveform_features,map_location='cpu',weights_only=False)
        if mixed['manifest_hash']!=sha or mixed['feature_version']!=cache['feature_version'] or mixed['augmentation_version']!='music-waveform-v1':raise ValueError('waveform cache provenance mismatch')
        waveform=mixed['features'];waveform_hash=hashlib.sha256(Path(a.waveform_features).read_bytes()).hexdigest()
        if set(waveform)!={r['id'] for r in rows if r['split']=='train' and r['phones']}:raise ValueError('waveform features must contain exactly train speech')
    if a.room_features:
        room_cache=torch.load(a.room_features,map_location='cpu',weights_only=False)
        if room_cache['manifest_hash']!=sha or room_cache['feature_version']!=cache['feature_version'] or room_cache['augmentation_version'] not in ['room-waveform-v1','room-waveform-v2']:raise ValueError('room cache provenance mismatch')
        room=room_cache['features'];room_hash=hashlib.sha256(Path(a.room_features).read_bytes()).hexdigest()
        if set(room)!={r['id'] for r in rows if r['split']=='train' and r['phones']}:raise ValueError('room features must contain exactly train speech')
        if room_cache['augmentation_version']=='room-waveform-v2':
            room_music=room_cache['mixed_features']
            if set(room_music)!=set(room) or any(len(v)!=2 for v in room_music.values()):raise ValueError('Paired room/music cache must contain exactly two variants for every train speech row')
        if waveform and not room_music:raise ValueError('Room and waveform mixtures require a paired v2 cache')
    vocab={c:i+1 for i,c in enumerate(sorted({c for r in rows if r['split']=='train' for c in r['phones']}))}
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev' and set(r['phones'])<=set(vocab)]
    if a.smoke_rows:train=train[:a.smoke_rows];dev=dev[:min(8,a.smoke_rows)]
    target={r['id']:torch.tensor([vocab[c] for c in r['phones']],dtype=torch.long) for r in train+dev}
    model=AcousticModel(len(vocab)+1,a.hidden,a.causal_layers,.1,bins=64).to(a.device)
    parent_hash=None
    if a.warmstart:
        parent=torch.load(a.warmstart,map_location=a.device,weights_only=False)
        if parent['vocab']!=vocab or parent.get('bins',40)!=64:raise ValueError('own-parent vocabulary/frontend mismatch')
        if parent['hidden']!=a.hidden or parent.get('causal_layers',0)!=a.causal_layers:raise ValueError('own-parent architecture mismatch')
        model.load_state_dict(checkpoint_parameters(parent));parent_hash=hashlib.sha256(Path(a.warmstart).read_bytes()).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    ctc=torch.nn.CTCLoss(reduction='none',zero_infinity=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise RuntimeError('refuse overwriting trained experiment')
    (out/'plan.json').write_text(json.dumps({'args':vars(a),'manifest_hash':sha,'initialization':'own checkpoint' if a.warmstart else 'random seed 404; no external weights','parent_checkpoint_sha256':parent_hash,
        'feature_version':cache['feature_version'],'text_version':'espeak-case-v2','mix':f'power addition; SNR 5..30 dB, 10 percent -2..5 dB; {100*(1-a.speech_mix_probability):g} percent clean',
        'speech_mix_probability':a.speech_mix_probability,
        'background_prelude':'50 percent 0.2..1.0 s; no target labels','blank_loss_weight':a.blank_loss_weight,
        'negative_probability':a.negative_probability,'negative_sampling':'fresh contiguous 2..8 s crop; replaces presentation before speech augmentation',
        'extra_backgrounds_sha256':extra_hash,
        'waveform_feature_sha256':waveform_hash,'waveform_mixture_probability_within_mixed_speech':.5 if waveform else 0,
        'room_feature_sha256':room_hash,'room_probability':a.room_probability if room else 0,
        'paired_room_music':bool(room_music)},indent=2))
    def batch(rs,training):
        xs=[];ys=[]
        for r in rs:
            negative=training and a.negative_probability>0 and random.random()<a.negative_probability
            y=torch.empty(0,dtype=torch.long) if negative else target[r['id']]
            x=music_crop(bg) if negative else feats[r['id']].float()
            selected_room=room.get(r['id']) if training and not negative and r['id'] in room and random.random()<a.room_probability else None
            if selected_room is not None:x=cached_speech(x,room=selected_room)
            if training and len(y) and random.random()<a.speech_mix_probability:
                noise=random.choice(bg).float();n=len(x)
                if len(noise)<n+50:noise=noise.repeat((n+50)//len(noise)+1,1)
                j=random.randrange(len(noise)-n+1)
                if waveform and random.random()<.5:
                    variant=random.randrange(len(waveform[r['id']]))
                    x=cached_speech(x,room=selected_room,music=waveform[r['id']],room_music=room_music.get(r['id']),variant=variant)
                else:
                    x=mix_power(x,noise[j:j+n],random.uniform(-2,5) if random.random()<.1 else random.uniform(5,30))
                if random.random()<.5:
                    pre=random.randint(10,50);x=torch.cat([noise[max(0,j-pre):j] if j>=pre else noise[:pre],x])
            x=normalize(x)
            if training and random.random()<.25:
                x=x.clone();f=random.randrange(61);x[:,f:f+4]=0
                t=random.randrange(max(1,len(x)-2));x[t:t+2]=0
            xs.append(x);ys.append(y)
        return pad_sequence(xs,batch_first=True).to(a.device),torch.cat(ys).to(a.device),torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys])
    def loss_for(lp,y,il,tl):
        return normalized_ctc_loss(ctc,lp,y,il,tl,a.blank_loss_weight)
    started=time.monotonic();best=float('inf');step=0
    for epoch in range(a.epochs):
        # Time-aware decay also converges if the prospective wall limit wins.
        progress=max(epoch/a.epochs,min(.99,(time.monotonic()-started)/a.max_seconds))
        lr=a.lr*(.05+.95*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:group['lr']=lr
        random.shuffle(train);order=[r for i in range(0,len(train),768) for r in sorted(train[i:i+768],key=lambda r:len(feats[r['id']]))]
        model.train();losses=[]
        for i in range(0,len(order),a.batch):
            if a.max_steps:
                progress=min(step/a.max_steps,1.)
                lr=a.lr*(.05+.95*(1+math.cos(math.pi*progress))/2)
                for group in opt.param_groups:group['lr']=lr
            x,y,il,tl=batch(order[i:i+a.batch],True);lp,_=model(x);loss=loss_for(lp,y,il,tl)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTC loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1;losses.append(loss.item())
            if a.max_steps and step>=a.max_steps:break
        model.eval();edits=tokens=0;dv=[]
        with torch.inference_mode():
            for i in range(0,len(dev),a.batch):
                rs=dev[i:i+a.batch];x,y,il,tl=batch(rs,False);lp,_=model(x);dv.append(loss_for(lp,y,il,tl).item())
                for j,r in enumerate(rs):
                    truth=target[r['id']].tolist();pred=collapse(lp[j,:il[j]].argmax(-1).cpu().tolist());edits+=edit_distance(truth,pred);tokens+=len(truth)
        val=float(np.mean(dv));improved=val<best;best=min(best,val)
        ck={'model':model.state_dict(),'vocab':vocab,'hidden':a.hidden,'bins':64,'causal_layers':a.causal_layers,'dropout':.1,'epoch':epoch+1,'step':step,'best':best,
            'manifest_hash':sha,'feature_version':cache['feature_version'],'text_version':'espeak-case-v2','initialization':'own checkpoint' if a.warmstart else 'random','parent_checkpoint_sha256':parent_hash,'args':vars(a),'parameters':sum(p.numel() for p in model.parameters())}
        ck['pronunciation_label_versions']=sorted({r.get('phone_pronunciation_version','espeak-v1') for r in train})
        ck['room_feature_sha256']=room_hash
        if improved:
            torch.save(ck,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt')
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        metrics={'epoch':epoch+1,'step':step,'train_ctc':float(np.mean(losses)),'dev_ctc':val,'dev_phone_error_rate':edits/tokens,'dev_unsupported':sum(r['split']=='dev' for r in rows)-len(dev),
                 'wall_seconds':time.monotonic()-started,'lr':lr,'parameters':ck['parameters'],'device':a.device}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if time.monotonic()-started>=a.max_seconds or (a.max_steps and step>=a.max_steps):break

if __name__=='__main__':main()
