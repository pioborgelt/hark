"""Local GPU learning with RAM-only annotations and effective batch96.

Mixing mel powers approximates uncorrelated waveform addition. Evaluation uses
actual waveform mixtures, never this approximation. No pretrained parameters.
"""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .model import AcousticModel, checkpoint_parameters
from .residual_acoustic_model import ResidualAcousticModel, MODEL_KIND
from .music_peak_loss import music_peak_blank_loss
from .music_context_sampling import contiguous_music_crop
from .training_music_extension import load_extension
from .train import audit, collapse, edit_distance
from .timed_conditional_phone_loss import conditional_timed_phone_loss
from .timed_phone_loss import timed_phone_loss
from .timed_target_delay import delay_targets
from .canonical_alignment import frame_targets as canonical_targets
from .parent_training_recipe import recipe_from_checkpoint
from .alignment import frame_targets
from .training_stop_policy import stop_after_epoch
from .segment_confusion_r162 import aligned_windows
from .segment_training_r168 import microbatch_backward


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


def normalized_ctc_loss(ctc,lp,y,input_lengths,target_lengths,blank_weight=.25,losses=None):
    if losses is None:losses=ctc(lp.transpose(0,1),y,input_lengths,target_lengths)
    scale=torch.where(target_lengths>0,target_lengths,input_lengths).to(lp.device)
    weight=torch.where(target_lengths>0,1.,blank_weight).to(lp.device)
    return (losses/scale*weight).mean()


def hash_file(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_expansion(rows,base_rows,allow_base_only=False):
    """Every original row stays exact; added records are new train-only speech."""
    base={r['id']:r for r in base_rows}
    full={r['id']:r for r in rows}
    if len(full)!=len(rows) or len(base)!=len(base_rows) or not set(base)<=set(full):
        raise ValueError('Duplicate or missing original record')
    if any(full[i]!=r for i,r in base.items()):
        raise ValueError('Original row changed')
    added=[r for i,r in full.items() if i not in base]
    if (not added and not allow_base_only) or any(r['split']!='train' or r.get('expansion_version')!='round64-vox-gold-v1'
        or r.get('gold_transcript') is not True or r.get('labels') or not r['phones'] for r in added):
        raise ValueError('Expansion must contain only new gold-transcript train speech without timed labels')
    if set(r['origin_id'] for r in added)&set(r['origin_id'] for r in base_rows):
        raise ValueError('Duplicate source origin')
    return {r['id'] for r in base_rows if r['split']=='train' and r['phones']}


def main(argv=None):
    if Path.cwd()!=Path(__file__).resolve().parent.parent:
        raise RuntimeError('Local study reuses only the existing read-only base feature caches')
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/round4/manifest.jsonl');p.add_argument('--features',default='data/round4/raw_features.pt')
    p.add_argument('--feature-delta')
    p.add_argument('--base-manifest',default='data/round4-onset/manifest.jsonl')
    p.add_argument('--allow-base-only',action='store_true')
    p.add_argument('--room-features')
    p.add_argument('--adapter-hidden',type=int,choices=[0,256],default=0)
    p.add_argument('--microbatch',type=int,default=24)
    p.add_argument('--alignment-weight',type=float,required=True)
    p.add_argument('--alignment-mode',choices=['conditional','unconditional'],required=True)
    p.add_argument('--target-mapping',choices=['exact','canonical'],required=True)
    p.add_argument('--target-delay-frames',type=int,choices=[0,4],required=True)
    p.add_argument('--music-peak-weight',type=float,choices=[0.,.5],required=True)
    p.add_argument('--room-probability',type=float,default=.5)
    p.add_argument('--waveform-features')
    p.add_argument('--extra-backgrounds')
    p.add_argument('--expanded-backgrounds',required=True)
    p.add_argument('--expanded-music-manifest',required=True)
    p.add_argument('--negative-min-frames',type=int,choices=[100,400],required=True)
    p.add_argument('--negative-max-frames',type=int,choices=[400,1199],required=True)
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
    p.add_argument('--segment-weight',type=float,choices=[0.,1.],required=True)
    p.add_argument('--decoy-plan',required=True)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    if (a.negative_min_frames,a.negative_max_frames) not in [(100,400),(400,1199)]:raise ValueError('Registered short/long music arms only')
    if a.adapter_hidden or not a.allow_base_only:raise ValueError('Original local base data and own base encoder only')
    if not a.warmstart:raise ValueError('A validated raw own parent is required')
    if not 1<=a.microbatch<=a.batch:raise ValueError('Invalid microbatch')
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.7)
    if a.alignment_weight not in [0.,.05,.15]:raise ValueError('Registered supervision arms only')
    if not 0<=a.negative_probability<1 or a.blank_loss_weight<=0:raise ValueError('invalid negative training settings')
    if not 0<=a.speech_mix_probability<=1:raise ValueError('invalid speech mixing probability')
    if not 0<=a.room_probability<=1:raise ValueError('invalid room probability')
    random.seed(404);np.random.seed(404);torch.manual_seed(404)
    raw=Path(a.manifest).read_bytes();sha=hashlib.sha256(raw).hexdigest();rows=[json.loads(s) for s in raw.decode().splitlines()];audit(rows)
    base_raw=Path(a.base_manifest).read_bytes()
    base_sha=hashlib.sha256(base_raw).hexdigest()
    if base_sha!='f160853716cd34d2a67555ec7e9ef66ee2f2c21b780c7c7f8c200ce783637210':
        raise ValueError('Original base manifest changed')
    base_rows=[json.loads(v) for v in base_raw.decode().splitlines()]
    base_train_speech=validate_expansion(rows,base_rows,a.allow_base_only)
    cache=torch.load(a.features,map_location='cpu',weights_only=True,mmap=True)
    if a.feature_delta:
        delta=torch.load(a.feature_delta,map_location='cpu',weights_only=True,mmap=True)
        if cache['manifest_hash']!=delta['base_manifest_hash'] or cache['feature_version']!=delta['feature_version']:raise ValueError('feature delta/base mismatch')
        cache['features'].update(delta['features']);cache['manifest_hash']=delta['manifest_hash']
    if cache['manifest_hash']!=sha or cache['feature_version']!='logmel64-power-v1':raise ValueError('cache provenance mismatch')
    feats=cache['features'];bg=list(cache['backgrounds']);room={};room_music={};room_hash=None;extra_hash=None;waveform={};waveform_hash=None
    if a.extra_backgrounds:
        extra=torch.load(a.extra_backgrounds,map_location='cpu',weights_only=True,mmap=True)
        if extra['version']!='round5-training-backgrounds-v1' or not extra['backgrounds']:raise ValueError('invalid training backgrounds')
        if len(extra['backgrounds'])!=len(extra['origins']):raise ValueError('background origin count mismatch')
        bg.extend(extra['backgrounds']);extra_hash=hash_file(a.extra_backgrounds)
    extension,extension_origins,old_count=load_extension(a.expanded_backgrounds,a.expanded_music_manifest)
    if len(bg)!=old_count or len(extension)!=160:raise ValueError('Registered244-origin music pool changed')
    if any(len(v)<400 for v in extension):raise ValueError('Both arms require at least8s of real contiguous music per origin')
    negative_bg=list(extension)
    bg.extend(extension)
    if a.waveform_features:
        mixed=torch.load(a.waveform_features,map_location='cpu',weights_only=True,mmap=True)
        if mixed['manifest_hash']!=base_sha or mixed['feature_version']!=cache['feature_version'] or mixed['augmentation_version']!='music-waveform-v1':raise ValueError('waveform cache provenance mismatch')
        waveform=mixed['features'];waveform_hash=hash_file(a.waveform_features)
        if set(waveform)!=base_train_speech:raise ValueError('waveform features must contain exactly train speech')
    if a.room_features:
        room_cache=torch.load(a.room_features,map_location='cpu',weights_only=True,mmap=True)
        if room_cache['manifest_hash']!=base_sha or room_cache['feature_version']!=cache['feature_version'] or room_cache['augmentation_version'] not in ['room-waveform-v1','room-waveform-v2']:raise ValueError('room cache provenance mismatch')
        room=room_cache['features'];room_hash=hash_file(a.room_features)
        if set(room)!=base_train_speech:raise ValueError('room features must contain exactly train speech')
        if room_cache['augmentation_version']=='room-waveform-v2':
            room_music=room_cache['mixed_features']
            if set(room_music)!=set(room) or any(len(v)!=2 for v in room_music.values()):raise ValueError('Paired room/music cache must contain exactly two variants for every train speech row')
        if waveform and not room_music:raise ValueError('Room and waveform mixtures require a paired v2 cache')
    vocab={c:i+1 for i,c in enumerate(sorted({c for r in rows if r['split']=='train' for c in r['phones']}))}
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev' and set(r['phones'])<=set(vocab)]
    if a.smoke_rows:train=train[:a.smoke_rows];dev=dev[:min(8,a.smoke_rows)]
    target={r['id']:torch.tensor([vocab[c] for c in r['phones']],dtype=torch.long) for r in train+dev}
    # Derive only original train annotations in RAM. No full local cache file,
    # archive extraction, copied features or teacher-generated training labels.
    alignments={r['id']:frame_targets(r,vocab,len(feats[r['id']])) for r in train}
    coverage={'rows':len(train),'all_frames':sum(len(v) for v in alignments.values()),
        'annotated_phone_frames':sum(int((v>0).sum()) for v in alignments.values())}
    for language in ['de','en']:
        coverage[language+'_annotated_rows']=sum(bool((alignments[r['id']]>0).any()) for r in train if r['language']==language)
    if not a.smoke_rows:
        expected={'rows':25078,'all_frames':7768131,'annotated_phone_frames':2183654,'de_annotated_rows':7264,'en_annotated_rows':8554}
        original_ids={r['id'] for r in base_rows if r['split']=='train'}
        actual={'rows':len(original_ids),'all_frames':sum(len(alignments[i]) for i in original_ids),
            'annotated_phone_frames':sum(int((alignments[i]>0).sum()) for i in original_ids)}
        for language in ['de','en']:
            actual[language+'_annotated_rows']=sum(bool((alignments[r['id']]>0).any()) for r in base_rows
                if r['split']=='train' and r['language']==language)
        if actual!=expected:raise ValueError('Original annotation coverage changed')
        if any(bool((a>=0).any()) for i,a in alignments.items() if i not in original_ids):
            raise ValueError('New gold transcripts have no original timed phone annotations')
    annotation_source_sha256=hash_file('hark/alignment.py')

    if a.target_mapping=='canonical':
        extended={r['id']:canonical_targets(r,vocab,len(feats[r['id']])) for r in train}
        for ident,value in alignments.items():
            if not torch.equal(extended[ident][value>=0],value[value>=0]):
                raise ValueError('Canonical extension changed existing exact annotation')
        alignments=extended
        coverage['extended_phone_frames']=sum(int((v>0).sum()) for v in alignments.values())
        annotation_source_sha256=hash_file('hark/canonical_alignment.py')

    model=(ResidualAcousticModel(len(vocab)+1,a.hidden,a.causal_layers,.1,64,a.adapter_hidden) if a.adapter_hidden else AcousticModel(len(vocab)+1,a.hidden,a.causal_layers,.1,bins=64)).to(a.device)
    parent_hash=None
    if a.warmstart:
        parent=torch.load(a.warmstart,map_location=a.device,weights_only=False)
        inherited=recipe_from_checkpoint(parent)
        requested={k:getattr(a,k) for k in ['lr','room_probability','speech_mix_probability','negative_probability','blank_loss_weight','alignment_mode','target_mapping','alignment_weight','target_delay_frames','music_peak_weight']}
        if any(requested[k]!=inherited[k] for k in requested):raise ValueError('Continuation recipe differs from validated raw parent')
        if parent['vocab']!=vocab or parent.get('bins',40)!=64:raise ValueError('own-parent vocabulary/frontend mismatch')
        if parent['hidden']!=a.hidden or parent.get('causal_layers',0)!=a.causal_layers:raise ValueError('own-parent architecture mismatch')
        if a.adapter_hidden:model.initialize_parent(parent)
        else:model.load_state_dict(checkpoint_parameters(parent))
        parent_hash=hash_file(a.warmstart)
    decoy_path=Path(a.decoy_plan);decoy_spec=json.loads(decoy_path.read_text())
    if decoy_spec['parent_checkpoint_sha256']!=parent_hash:raise ValueError('Wrong own-head decoy source')
    from .contrastive_ctc_r110 import head_neighbors
    expected=head_neighbors(checkpoint_parameters(parent)['head.weight'],3)
    neighbors={int(k):v for k,v in decoy_spec['neighbors'].items()}
    if neighbors!=expected:raise ValueError('Decoys must derive from the registered own parent head')
    forbidden=tuple(tuple(p) for p in decoy_spec['protected_phone_sequences']);segment_rng=random.Random(831)
    segments={r['id']:aligned_windows(r,vocab,len(feats[r['id']]),forbidden=forbidden) for r in train}
    segment_coverage={'rows_with_windows':sum(bool(v) for v in segments.values()),'windows':sum(map(len,segments.values()))}
    print(json.dumps({'segment_coverage':segment_coverage}),flush=True)
    if not forbidden or decoy_spec['margin']!=1.:raise ValueError('Missing full-call protections or wrong margin')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    ctc=torch.nn.CTCLoss(reduction='none',zero_infinity=False)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() or (out/'plan.json').exists():raise RuntimeError('Preserve experiment')
    (out/'plan.json').write_text(json.dumps({'args':vars(a),'manifest_hash':sha,'initialization':'own checkpoint' if a.warmstart else 'random seed 404; no external weights','parent_checkpoint_sha256':parent_hash,
        'segment_weight':a.segment_weight,'segment_coverage':segment_coverage,'segment_lengths':[4,8,12],'segment_margin':1.,'decoy_plan_sha256':hash_file(a.decoy_plan),'decoy_source':'own frozen parent head metadata; generic exact-aligned 4/8/12-codepoint window; one feasible insertion/deletion/substitution; private RNG831; protected full-call truths and decoys excluded; no transcript replacement',
        'feature_version':cache['feature_version'],'text_version':'espeak-case-v2','mix':f'power addition; SNR 5..30 dB, 10 percent -2..5 dB; {100*(1-a.speech_mix_probability):g} percent clean',
        'speech_mix_probability':a.speech_mix_probability,
        'background_prelude':'50 percent 0.2..1.0 s; no target labels','blank_loss_weight':a.blank_loss_weight,
        'negative_probability':a.negative_probability,'negative_sampling':f'fresh contiguous {a.negative_min_frames/50:g}..{a.negative_max_frames/50:g}s crop; replaces presentation before speech augmentation','expanded_music_sha256':hash_file(a.expanded_backgrounds),'background_pool_size':len(bg),'negative_music_pool_size':len(negative_bg),
        'extra_backgrounds_sha256':extra_hash,
        'waveform_feature_sha256':waveform_hash,'waveform_mixture_probability_within_mixed_speech':.5 if waveform else 0,
        'room_feature_sha256':room_hash,'room_probability':a.room_probability if room else 0,
        'alignment_mode':a.alignment_mode,'target_mapping':a.target_mapping,'target_delay_frames':a.target_delay_frames,'music_peak_weight':a.music_peak_weight,'parent_training_recipe':inherited,'paired_room_music':bool(room_music),'alignment_weight':a.alignment_weight,'timed_supervision':'Original annotation coverage checked before optional canonical extension. Explicit inherited unconditional/conditional loss and label delay; future window0..60ms. All weights trainable.', 'annotation_storage':'RAM only, no full local annotation file','annotation_coverage':coverage,'annotation_source_sha256':annotation_source_sha256,'base_manifest_sha256':base_sha,'new_train_rows':len(rows)-len(base_rows),'new_data_augmentation':'New naturally recorded parliamentary speech receives online power-music augmentation; paired waveform/room caches apply only to unchanged original rows.'},indent=2))
    def batch(rs,training):
        xs=[];ys=[];aux=[];windows=[]
        for r in rs:
            negative=training and a.negative_probability>0 and random.random()<a.negative_probability
            y=torch.empty(0,dtype=torch.long) if negative else target[r['id']]
            x=contiguous_music_crop(negative_bg,a.negative_min_frames,a.negative_max_frames) if negative else feats[r['id']].float()
            label=delay_targets(alignments[r['id']],a.target_delay_frames) if training and not negative else torch.full((len(x),),-100,dtype=torch.long)
            selected_room=room.get(r['id']) if training and not negative and r['id'] in room and random.random()<a.room_probability else None
            if selected_room is not None:x=cached_speech(x,room=selected_room)
            if training and len(y) and random.random()<a.speech_mix_probability:
                noise=random.choice(bg).float();n=len(x)
                if len(noise)<n+50:noise=noise.repeat((n+50)//len(noise)+1,1)
                j=random.randrange(len(noise)-n+1)
                if waveform and r['id'] in waveform and random.random()<.5:
                    variant=random.randrange(len(waveform[r['id']]))
                    x=cached_speech(x,room=selected_room,music=waveform[r['id']],room_music=room_music.get(r['id']),variant=variant)
                else:
                    x=mix_power(x,noise[j:j+n],random.uniform(-2,5) if random.random()<.1 else random.uniform(5,30))
                if random.random()<.5:
                    pre=random.randint(10,50);prefix=noise[max(0,j-pre):j] if j>=pre else noise[:pre];x=torch.cat([prefix,x]);label=torch.cat([torch.full((len(prefix),),-100,dtype=torch.long),label])
            x=normalize(x)
            if training and random.random()<.25:
                x=x.clone();f=random.randrange(61);x[:,f:f+4]=0
                t=random.randrange(max(1,len(x)-2));x[t:t+2]=0
            shift=len(x)-len(feats[r['id']]) if not negative else 0
            windows.append([(b+shift,e+shift,t) for b,e,t in segments[r['id']]] if training and not negative else [])
            xs.append(x);ys.append(y);aux.append(label)
        return pad_sequence(xs,batch_first=True).to(a.device),torch.cat(ys).to(a.device),torch.tensor([len(x) for x in xs]),torch.tensor([len(y) for y in ys]),pad_sequence(aux,batch_first=True,padding_value=-100).to(a.device),windows
    def loss_for(lp,y,il,tl):
        return normalized_ctc_loss(ctc,lp,y,il,tl,a.blank_loss_weight)
    started=time.monotonic();best=float('inf');step=0
    for epoch in range(a.epochs):
        # Time-aware decay also converges if the prospective wall limit wins.
        progress=max(epoch/a.epochs,min(.99,(time.monotonic()-started)/a.max_seconds))
        lr=a.lr*(.05+.95*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:group['lr']=lr
        random.shuffle(train);order=[r for i in range(0,len(train),768) for r in sorted(train[i:i+768],key=lambda r:len(feats[r['id']]))]
        model.train();losses=[];aux_losses=[];peak_losses=[];segment_losses=[]
        for i in range(0,len(order),a.batch):
            if a.max_steps:
                progress=min(step/a.max_steps,1.)
                lr=a.lr*(.05+.95*(1+math.cos(math.pi*progress))/2)
                for group in opt.param_groups:group['lr']=lr
            x,y,il,tl,labels,windows=batch(order[i:i+a.batch],True)
            opt.zero_grad();ctc_scalar,aux_scalar,peak_scalar,segment_scalar=microbatch_backward(model,ctc,x,y,il,tl,labels,a.alignment_weight,a.blank_loss_weight,a.microbatch,a.alignment_mode,a.music_peak_weight,a.segment_weight,neighbors,segment_rng,forbidden,windows)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();step+=1
            losses.append(ctc_scalar);aux_losses.append(aux_scalar);peak_losses.append(peak_scalar);segment_losses.append(segment_scalar)
            if a.max_steps and step>=a.max_steps:break
        model.eval();edits=tokens=0;dv=[]
        with torch.inference_mode():
            for i in range(0,len(dev),a.batch):
                rs=dev[i:i+a.batch];x,y,il,tl,_,_=batch(rs,False);lp,_=model(x);dv.append(loss_for(lp,y,il,tl).item())
                for j,r in enumerate(rs):
                    truth=target[r['id']].tolist();pred=collapse(lp[j,:il[j]].argmax(-1).cpu().tolist());edits+=edit_distance(truth,pred);tokens+=len(truth)
        val=float(np.mean(dv));improved=val<best;best=min(best,val)
        ck={'model':model.state_dict(),'vocab':vocab,'hidden':a.hidden,'bins':64,'causal_layers':a.causal_layers,'dropout':.1,'epoch':epoch+1,'step':step,'best':best,
            'manifest_hash':sha,'feature_version':cache['feature_version'],'text_version':'espeak-case-v2','initialization':'own checkpoint' if a.warmstart else 'random','parent_checkpoint_sha256':parent_hash,'args':vars(a),'parameters':sum(p.numel() for p in model.parameters())}
        if a.adapter_hidden:ck.update(model_kind=MODEL_KIND,adapter_hidden=a.adapter_hidden)
        ck['negative_music_frames']=[a.negative_min_frames,a.negative_max_frames]
        ck['background_pool_size']=len(bg)
        ck['negative_music_pool_size']=len(negative_bg)
        ck['expanded_music_sha256']=hash_file(a.expanded_backgrounds)
        ck['music_peak_objective']='top5pct-blank-nll-v1';ck['music_peak_weight']=a.music_peak_weight
        ck['pronunciation_label_versions']=sorted({r.get('phone_pronunciation_version','espeak-v1') for r in train})
        ck['room_feature_sha256']=room_hash
        ck['timed_annotation_source_sha256']=annotation_source_sha256;ck['alignment_weight']=a.alignment_weight;ck['alignment_objective']='conditional-nonblank-timed-v1' if a.alignment_mode=='conditional' else 'unconditional-timed-v1'
        ck['timed_target_mapping']='canonical-one-to-one-substitution-r55-v1' if a.target_mapping=='canonical' else None
        ck['phone_target_delay_frames']=a.target_delay_frames
        ck['parent_training_recipe']=inherited
        ck['ctc_segment']={'weight':a.segment_weight,'margin':1.,'decoy_plan_sha256':hash_file(a.decoy_plan),'rng_seed':831,'timed_and_ctc_ground_truth_unchanged':True,'segment_lengths':[4,8,12],'coverage':segment_coverage,'edit_modes':['insert','delete','substitute'],'scope':'Original annotated train GT only; no queries, teacher, new data or prefix training'}
        ck['annotation_storage']='RAM only';ck['annotation_storage']='RAM only';ck['microbatch']=a.microbatch;ck['base_manifest_sha256']=base_sha;ck['new_train_rows']=len(rows)-len(base_rows)
        torch.save(ck,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        metrics={'epoch':epoch+1,'step':step,'train_ctc':float(np.mean(losses)),'train_timed_phone_nll':float(np.mean(aux_losses)),'alignment_weight':a.alignment_weight,'train_music_peak_blank_nll':float(np.mean(peak_losses)),'music_peak_weight':a.music_peak_weight,'dev_ctc':val,'dev_phone_error_rate':edits/tokens,'dev_unsupported':sum(r['split']=='dev' for r in rows)-len(dev),
                 'train_ctc_segment':float(np.mean(segment_losses)),'segment_weight':a.segment_weight,'wall_seconds':time.monotonic()-started,'lr':lr,'parameters':ck['parameters'],'device':a.device}
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if stop_after_epoch(step,a.max_steps,time.monotonic()-started,a.max_seconds):break

if __name__=='__main__':main()
