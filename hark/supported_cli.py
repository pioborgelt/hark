"""Explicit stored-WAV V6 research CLI. No input-device or live commands."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from .model import load_model
from .audio import read_audio,RATE
from .text import encode,LEXICON_SHA256
from .supported_detect import Detector,source_sha
from .supported_contracts import SCHEMA,VERSION,KERNEL_SHA256,sha,seal,check_settings,validate_parameters,validate

def main(argv=None):
    parser=argparse.ArgumentParser(prog='hark-v6-offline')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--version',action='version',version='hark research / '+VERSION)
    sub=parser.add_subparsers(dest='command',required=True)
    enroll=sub.add_parser('enroll');enroll.add_argument('phrase');enroll.add_argument('--language',choices=['de','en'],required=True)
    enroll.add_argument('--checkpoint',required=True);enroll.add_argument('--calibration',required=True);enroll.add_argument('--out',required=True)
    detect=sub.add_parser('detect');detect.add_argument('wav');detect.add_argument('--checkpoint',required=True);detect.add_argument('--phrase',required=True)
    detect.add_argument('--chunk-ms',type=int,default=80)
    args=parser.parse_args(argv);torch.set_num_threads(args.threads);model,ck=load_model(args.checkpoint)
    if args.command=='enroll':
        calibration_path=Path(args.calibration).resolve();calibration=json.loads(calibration_path.read_text());check_settings(calibration,args.checkpoint)
        tokens=encode(args.phrase,args.language,ck['vocab'],ck.get('text_version','legacy-lower-v1'),calibration['pronunciation_version'])
        parameters=calibration['languages'][args.language];validate_parameters(parameters,tokens)
        phrase={'schema_version':SCHEMA,'decoder_version':VERSION,'kernel_sha256':KERNEL_SHA256,'decoder_source_sha256':source_sha(),
            'text':args.phrase,'language':args.language,'tokens':tokens,'checkpoint_sha256':sha(args.checkpoint),
            'vocab_sha256':hashlib.sha256(json.dumps(ck['vocab'],sort_keys=True).encode()).hexdigest(),
            'decoder_parameters':parameters,'calibration_path':str(calibration_path),'calibration_sha256':sha(calibration_path),
            'pronunciation_version':calibration['pronunciation_version'],'pronunciation_lexicon_sha256':LEXICON_SHA256,'live_validated':False}
        target=Path(args.out)
        if target.exists():raise ValueError('Preserve existing enrollment')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(seal(phrase),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'enrollment':str(target),'decoder_version':VERSION,'live_validated':False},ensure_ascii=False));return
    phrase=json.loads(Path(args.phrase).read_text());validate(phrase,args.checkpoint,ck['vocab'])
    if args.chunk_ms<=0:raise ValueError('Positive chunk duration required')
    detector=Detector(model,phrase['tokens'],**phrase['decoder_parameters']);audio=read_audio(args.wav)
    chunk=max(1,int(RATE*args.chunk_ms/1000));started=time.perf_counter();count=0
    for i in range(0,len(audio),chunk):
        for event in detector.push(audio[i:i+chunk]):count+=1;print(json.dumps({'phrase':phrase['text'],**event}),flush=True)
    wall=time.perf_counter()-started
    print(json.dumps({'events':count,'audio_seconds':len(audio)/RATE,'wall_seconds':wall,
                      'real_time_factor':wall/(len(audio)/RATE) if len(audio) else None}))
if __name__=='__main__':main()
