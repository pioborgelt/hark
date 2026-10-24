import argparse,hashlib,json,time
from pathlib import Path
import torch
from .audio import read_audio,RATE
from .model import load_model
from .text import encode,phones,PRONUNCIATION_VERSIONS,LEXICON_SHA256
from .detect import Detector
from .contracts import SCHEMA,VERSION,KERNEL_SHA256,PARAMETERS,sha,seal,validate,identity

def active():
    path=Path('artifacts/active_model.json')
    return json.loads(path.read_text()) if path.exists() else {}
def default_checkpoint():return active().get('checkpoint','artifacts/prototype/best.pt')
def default_calibration(checkpoint):
    settings=active()
    if settings.get('checkpoint') and Path(settings['checkpoint']).resolve()==Path(checkpoint).resolve() and settings.get('calibration'):return Path(settings['calibration'])
    return Path(checkpoint).parent/'calibration.json'

def main(argv=None):
    p=argparse.ArgumentParser(prog='hark');p.add_argument('--version',action='version',version='hark 0.2.0 / '+VERSION)
    p.add_argument('--threads',type=int,default=2);sub=p.add_subparsers(dest='command',required=True)
    enroll=sub.add_parser('enroll');enroll.add_argument('phrase');enroll.add_argument('--language',choices=['de','en'],required=True)
    enroll.add_argument('--checkpoint');enroll.add_argument('--calibration');enroll.add_argument('--out',default='artifacts/phrase.json')
    enroll.add_argument('--threshold',type=float);enroll.add_argument('--profile',choices=['strict','balanced'],default='strict')
    enroll.add_argument('--min-phone-probability',type=float,default=.003)
    enroll.add_argument('--mean-phone-probability',type=float,default=.1)
    enroll.add_argument('--phone-margin',type=float,default=.1)
    enroll.add_argument('--min-blank-probability',type=float,default=.1)
    enroll.add_argument('--pronunciation-version',choices=PRONUNCIATION_VERSIONS)
    detect=sub.add_parser('detect');detect.add_argument('wav');detect.add_argument('--realtime',action='store_true')
    mic=sub.add_parser('mic',help='Local input; requires a live-validated enrollment')
    mic.add_argument('--device');mic.add_argument('--rate',type=int);mic.add_argument('--seconds',type=float)
    for cmd in [detect,mic]:
        cmd.add_argument('--phrase',default='artifacts/phrase.json');cmd.add_argument('--checkpoint')
        cmd.add_argument('--chunk-ms',type=int,default=80)
    sub.add_parser('devices');a=p.parse_args(argv);torch.set_num_threads(a.threads)
    if a.command=='devices':
        from .microphone import list_devices
        try:print(json.dumps(list_devices(),ensure_ascii=False,indent=2))
        except Exception as exc:p.exit(2,f'hark devices: {exc}\n')
        return
    a.checkpoint=a.checkpoint or default_checkpoint();model,ck=load_model(a.checkpoint)
    if a.command=='enroll':
        calibration_path=None;calibration_sha=None
        pronunciation_version=a.pronunciation_version or 'espeak-v1'
        if a.threshold is None:
            calibration_path=Path(a.calibration) if a.calibration else default_calibration(a.checkpoint)
            if not calibration_path.exists():raise ValueError('No absolute-score calibration; supply a v5 calibration or an explicit research threshold')
            calibration=json.loads(calibration_path.read_text())
            if calibration.get('decoder_version')!=VERSION or calibration.get('schema_version')!=SCHEMA or calibration.get('kernel_sha256')!=KERNEL_SHA256:
                raise ValueError('Retired relative-score calibration; old strict/balanced thresholds are incompatible with '+VERSION)
            if calibration['checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('Calibration belongs to different model weights')
            calibrated_pronunciation=calibration.get('pronunciation_version','espeak-v1')
            if a.pronunciation_version and a.pronunciation_version!=calibrated_pronunciation:raise ValueError('Pronunciation differs from calibration')
            pronunciation_version=calibrated_pronunciation
            if pronunciation_version=='de-lexicon-v1' and calibration.get('pronunciation_lexicon_sha256')!=LEXICON_SHA256:raise ValueError('Pronunciation lexicon changed; recalibrate')
            settings=calibration['languages'][a.language]
            if a.profile!='strict':raise ValueError('No balanced operating point validated for absolute decoder')
            params={k:settings[k] for k in PARAMETERS};calibration_sha=sha(calibration_path)
        else:
            params={'threshold':a.threshold,'min_phone_probability':a.min_phone_probability,
                    'mean_phone_probability':a.mean_phone_probability,'phone_margin':a.phone_margin,
                    'min_blank_probability':a.min_blank_probability}
        # Validate scalar parameters without inference; do not turn a manual
        # threshold into a claim that a live microphone is validated.
        from .detect import Matcher
        tokens=encode(a.phrase,a.language,ck['vocab'],ck.get('text_version','legacy-lower-v1'),pronunciation_version)
        negative_evidence=None
        if a.threshold is None and calibration.get('negative_calibration'):
            from .calibrate import calibrate,load_negative_cache
            descriptor=calibration['negative_calibration']
            streams=load_negative_cache(descriptor,sha(a.checkpoint),calibration_path.resolve().parent)
            result=calibrate(tokens,streams,descriptor['seconds'],descriptor['fixed_parameters'],descriptor['thresholds'],descriptor['max_events_per_hour'])
            if not result['feasible']:raise ValueError('No text threshold meets the recorded negative-calibration constraint')
            params=result['parameters']
            negative_evidence={'version':descriptor['version'],'descriptor_sha256':identity(descriptor),
                               'negative_hours':result['negative_hours'],'trials':result['trials']}
        Matcher(tokens,**params)
        phrase=seal({'schema_version':SCHEMA,'decoder_version':VERSION,'kernel_sha256':KERNEL_SHA256,'text':a.phrase,'language':a.language,
            'phones':phones(a.phrase,a.language,ck.get('text_version','legacy-lower-v1'),pronunciation_version),'text_version':ck.get('text_version','legacy-lower-v1'),'pronunciation_version':pronunciation_version,
            'pronunciation_lexicon_sha256':LEXICON_SHA256 if pronunciation_version=='de-lexicon-v1' else None,'tokens':tokens,'decoder_parameters':params,
            'vocab_sha256':hashlib.sha256(json.dumps(ck['vocab'],sort_keys=True).encode()).hexdigest(),
            'checkpoint_sha256':sha(a.checkpoint),'calibration_path':str(calibration_path) if calibration_path else None,
            'calibration_sha256':calibration_sha,'live_validated':False,'validation_status':'offline_research_no_human_pai_validation'})
        if negative_evidence:phrase['negative_calibration']=negative_evidence;seal(phrase)
        out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(phrase,ensure_ascii=False,indent=2))
        print(json.dumps(phrase,ensure_ascii=False));return
    phrase=json.loads(Path(a.phrase).read_text());validate(phrase,a.checkpoint,ck['vocab'])
    if a.chunk_ms<1:raise ValueError('chunk-ms must be positive')
    if a.command=='mic':
        if not phrase.get('live_validated'):
            p.exit(2,'hark mic: live detection is disabled after the music false-alarm incident. This enrollment has offline research validation only; no microphone was opened.\n')
        if a.seconds is not None and a.seconds<=0:raise ValueError('seconds must be positive')
        from .microphone import listen
        try:listen(model,phrase,a.device,a.rate,a.chunk_ms,a.seconds,emit=lambda s:print(s,flush=True))
        except Exception as exc:p.exit(2,f'hark mic: {exc}\n')
        return
    audio=read_audio(a.wav);detector=Detector(model,phrase['tokens'],**phrase['decoder_parameters'])
    chunk=max(1,int(RATE*a.chunk_ms/1000));started=time.perf_counter();count=0
    for i in range(0,len(audio),chunk):
        for event in detector.push(audio[i:i+chunk]):count+=1;print(json.dumps({'phrase':phrase['text'],**event}),flush=True)
        if a.realtime:time.sleep(max(0,min((i+chunk)/RATE,len(audio)/RATE)-(time.perf_counter()-started)))
    elapsed=time.perf_counter()-started
    print(json.dumps({'events':count,'audio_seconds':len(audio)/RATE,'wall_seconds':elapsed,
                      'real_time_factor':elapsed/(len(audio)/RATE) if len(audio) else None}))
if __name__=='__main__':main()
