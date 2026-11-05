#!/usr/bin/env python3
"""Model-free pinned tokenization fuzz corpus and symbolic MIDI edge fixtures."""
import json
from pathlib import Path
import random
import sys
import subprocess
import mido
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,'/refs/MuLaCover/src')
assert subprocess.check_output(['git','-C','/refs/MuLaCover','rev-parse','HEAD'],text=True).strip()=='e27fa7fa0da7c9817879cc6210cc5693f9327946'
from mulacover.symbolic import SymbolicCondition
from mulacover import MuLaCoverGenPipeline,MuLaCoverGenConfig,MuLaCoverConfig
root=Path('refdata/conditioning-text-score');root.mkdir(parents=True,exist_ok=True)
sources={n:json.loads(Path('models',n+'.json').read_text()) for n in ['generator','style']}
tokenizers={n:Tokenizer.from_file(str(Path(sources[s]['snapshot'])/'tokenizer.json')) for n,s in [('prompt','generator'),('style','style')]}
corpus=["DON'T I'M we'll 'ſ 'S 123456789 １２３４ Ⅷ²",' cafe\u0301 CAFÉ İSTANBUL ΟΣ ΟΣΑ Ο\u0345ΣΑ ',
        '\u1100\u1161\u11a8 한글 日本語 中文 😀👩\u200d🎸\r\n\t', '\x00\x1c\x1f\t\n\r\u0085\u00a0\u2003\u2028\u202f',
        '\U0001e4d0\U0001e4d1\U0001e4f0\U0001e4f1\U00011f02\U00011f36',
        '<|begin_of_text|><|end_of_text|><tool_call><think> <|endoftext|>']
random.seed(19287);alphabet=list("abCD'S 0123456789\n\r\t.,!?-")+list('é中ΩΣİşſ²Ⅳ😀')+['\u0301','\u200d','\u2003','\U0001e4d0','\U0001e4f0']
for _ in range(256):corpus.append(''.join(random.choice(alphabet) for _ in range(random.randrange(1,160))))
for i,text in enumerate(corpus):
    folder=root/'text'/str(i);folder.mkdir(parents=True,exist_ok=True);(folder/'text.txt').write_text(text,encoding='utf-8')
    for name,t in tokenizers.items():np.asarray(t.encode(text,add_special_tokens=name=='style').ids,np.int32).tofile(folder/(name+'.i32'))
    (folder/'lower.txt').write_text(text.lower(),encoding='utf-8');(folder/'strip.txt').write_text(text.strip(),encoding='utf-8')

score=root/'score';score.mkdir(exist_ok=True)
def write(name,tracks):
    f=mido.MidiFile(type=1,ticks_per_beat=480)
    for events in tracks:
        track=mido.MidiTrack();f.tracks.append(track)
        for msg in events:track.append(msg)
    f.save(score/(name+'.mid'))
on=lambda note,time=0,channel=0,velocity=100:mido.Message('note_on',note=note,time=time,channel=channel,velocity=velocity)
off=lambda note,time=0,channel=0:mido.Message('note_off',note=note,time=time,channel=channel)
write('melody',[[mido.MetaMessage('set_tempo',tempo=600000),on(60,60),on(60,0),on(61,60),off(60,60),on(61,60,velocity=0),off(60,60),on(70,0,channel=9),off(70,120,channel=9)],
                [mido.MetaMessage('set_tempo',tempo=300000,time=120),on(72,480,channel=1),off(72,120,channel=1)]])
write('chord',[[on(48),on(52),on(55),off(48,480),off(52),off(55),on(67,120),off(67,240)]])
write('drum',[[on(36,channel=9),off(36,120,channel=9),on(38,120,channel=2),off(38,240,channel=2)]])
c=SymbolicCondition.from_midi(score/'melody.mid',score/'chord.mid',score/'drum.mid')
for name,t in c.to_tensors().items():t.numpy().tofile(score/(name+('.u8' if name=='context_mask' else '.f32')))
write('bad-unmatched',[[off(60)]])
write('bad-unterminated',[[on(60)]])
write('bad-empty-melody',[[on(60,channel=9),off(60,120,channel=9)]])
write('bad-too-long',[[on(60,720120),off(60,120)]])
write('bad-zero-tempo',[[mido.MetaMessage('set_tempo',tempo=0),on(60),off(60,120)]])
valid=(score/'melody.mid').read_bytes()
for name,start,values in [('bad-type2',8,b'\x00\x02'),('bad-smpte',12,b'\xe7\x28'),('bad-zero-ppq',12,b'\x00\x00')]:
    data=bytearray(valid);data[start:start+2]=values;(score/(name+'.mid')).write_bytes(data)
(score/'bad-truncated.mid').write_bytes(valid[:-3])
invalid=[]
for path in sorted(score.glob('bad-*.mid')):
    try:SymbolicCondition.from_midi(path,score/'chord.mid',score/'drum.mid')
    except (ValueError,OSError,EOFError,KeyError):invalid.append(path.name)
    else:raise RuntimeError('Upstream accepted negative fixture '+path.name)
(score/'invalid.txt').write_text(''.join(name+'\n' for name in invalid))
manifest={'text_cases':len(corpus),'symbolic_frames':c.length,'bpm':c.bpm,'upstream_rejected_midi':invalid,'score_checks':['half-grid ties round up','overlapping same-pitch FIFO','note_on velocity zero','tracks and channels','melody/chord channel 9 excluded','explicit drum MIDI includes all channels','tempo changes are metadata']}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest,flush=True)
