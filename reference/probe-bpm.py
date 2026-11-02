#!/usr/bin/env python3
"""Prepare tempo-metadata and text-tag ablations through pinned upstream code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import mido

p=argparse.ArgumentParser(description=__doc__)
for name in ('upstream','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();project=Path(__file__).resolve().parents[1]
if a.output.exists():raise ValueError('Use a fresh BPM study directory')
frozen=json.loads((a.prepared/'manifest.json').read_text());a.output.mkdir(parents=True)
sys.path.insert(0,str(project/'tools'))
from midi_bridge import midi_notes
sources={k:Path(v['file']) for k,v in frozen['input_files'].items()}
report=dict(upstream_revision=frozen['upstream_revision'],requested_audio_frames=372,tempos=[80,160],cases=[])
for mode in ('metadata','tags'):
    for bpm in report['tempos']:
        case=a.output/f'{mode}-{bpm}';case.mkdir();request=dict(frozen['request'])
        note_checks={}
        for key,source in sources.items():
            if mode=='metadata':
                midi=mido.MidiFile(source);found=False
                for track in midi.tracks:
                    for i,msg in enumerate(track):
                        if msg.type=='set_tempo':track[i]=msg.copy(tempo=mido.bpm2tempo(bpm));found=True
                if not found:midi.tracks[0].insert(0,mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bpm)))
                target=case/(key+'.mid');midi.save(target)
                before=midi_notes(source)[1];after=midi_notes(target)[1]
                assert before==after
                note_checks[key]=dict(notes=len(before),beat_positions_unchanged=True)
                request[key]=target.name
            else:request[key]=str(source.resolve())
        if mode=='tags':request['tags']=request['tags']+f', {bpm} bpm'
        (case/'request.json').write_text(json.dumps(request,indent=2)+'\n')
        subprocess.run([sys.executable,str(project/'reference/prepare-conditioning.py'),'--upstream',str(a.upstream),
            '--request',str(case/'request.json'),'--output',str(case/'prepared'),'--frames','372'],check=True)
        prepared=json.loads((case/'prepared/manifest.json').read_text())
        changed=[k for k,v in frozen['preprocessing'].items() if any(v[f]!=prepared['preprocessing'][k][f] for f in ('sha256','shape','stored_dtype'))]
        if mode=='metadata' and changed:raise ValueError('Tempo-only edits unexpectedly changed model inputs')
        report['cases'].append(dict(mode=mode,bpm=bpm,prepared=str(case/'prepared'),changed_tensors=changed,
            note_checks=note_checks,tensor_hashes={k:v['sha256'] for k,v in prepared['preprocessing'].items()}))
        print('BPM ablation',mode,bpm,'changed',changed,flush=True)
report['metadata_only_all_11_tensors_identical']=all(not c['changed_tensors'] for c in report['cases'] if c['mode']=='metadata')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
