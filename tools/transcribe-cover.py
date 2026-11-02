#!/usr/bin/env python3
"""Run existing native SheetSage2, audit MIDI/events, and prepare an aligned excerpt."""
import argparse
import collections
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import wave
import numpy as np
from midi_bridge import audit_events,crop_midi,midi_notes


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audio','cli','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-audio-sha256');p.add_argument('--melody-role',choices=('combined','vocal','instrumental'),default='instrumental')
    p.add_argument('--sixteenths',type=int,default=64);p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--full-song',action='store_true',help='Keep all transcribed MIDI and the entire source reference audio')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    source_hash=digest(a.audio)
    if a.expected_audio_sha256 and source_hash!=a.expected_audio_sha256:raise ValueError('Source audio hash changed')
    if a.sixteenths<1 or a.sixteenths>5000:raise ValueError('Invalid excerpt length')
    if (a.output/'transcription.json').exists():raise ValueError('Use a fresh transcription directory')
    a.output.mkdir(parents=True,exist_ok=True);whole=a.output/'whole-midi'
    command=[str(a.cli.resolve()),'--model',str(a.model.resolve()),'--audio',str(a.audio.resolve()),
        '--output',str((a.output/'score.abc').resolve()),'--midi-dir',str(whole.resolve()),
        '--events',str((a.output/'events.json').resolve()),'--full','--device',a.device]
    spec=importlib.util.spec_from_file_location('native_parity',root/'tools/run-parity.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    with (a.output/'stdout.txt').open('w') as out,(a.output/'stderr.txt').open('w') as err:
        run,memory=helper.run_native(command,out,err,dict(os.environ,GGML_CUDA_TF32='0'),a.device=='cuda')
    if run.returncode:raise RuntimeError((a.output/'stderr.txt').read_text()[-3000:])
    events=json.loads((a.output/'events.json').read_text());checks={}
    names={'combined':'melody.mid','vocal':'melody_vocal.mid','instrumental':'melody_instrumental.mid'}
    for role,name in names.items():checks[role]=audit_events(whole/name,events['events'],role)
    if not all(c['beat_positions_pitches_durations_match'] for c in checks.values()):raise ValueError('Native MIDI differs from decoded score events')
    excerpt=a.output/'excerpt-midi';excerpt.mkdir()
    def select_midi(source,destination):
        if not a.full_song:return crop_midi(source,destination,a.sixteenths)
        shutil.copyfile(source,destination)
        midi,notes=midi_notes(destination)
        return dict(notes=len(notes),duration_seconds=midi.length,ticks_per_beat=midi.ticks_per_beat,
            full_song=True,last_note_step=max((n['onset_step']+n['duration_steps'] for n in notes),default=0))
    melody=select_midi(whole/names[a.melody_role],excerpt/'melody.mid')
    chords=select_midi(whole/'chords.mid',excerpt/'chords.mid')
    if melody['notes']==0 or chords['notes']==0:raise ValueError('Selected excerpt has no melody or chord notes')
    # Map the beat-space crop back to source timestamps for a reference listen.
    points={int(e['subbeat']):float(e['time_seconds']) for e in events['events']}
    keys=sorted(points);source_end=float(np.interp(a.sixteenths,keys,[points[k] for k in keys]))
    with wave.open(str(a.audio),'rb') as f:
        audio_info=dict(channels=f.getnchannels(),sample_rate=f.getframerate(),sample_width=f.getsampwidth(),frames=f.getnframes())
        if a.full_song:source_end=f.getnframes()/f.getframerate()
        data=f.readframes(min(f.getnframes(),round(source_end*f.getframerate())))
        with wave.open(str(a.output/'source-excerpt.wav'),'wb') as out:out.setparams(f.getparams());out.writeframes(data)
    binary_version=subprocess.check_output([str(a.cli.resolve()),'--version'],text=True).strip()
    linked_ggml=None;dll=a.cli.parent/'ggml-base.dll'
    if os.name=='nt' and dll.exists():
        with os.add_dll_directory(str(a.cli.parent.resolve())):
            lib=ctypes.CDLL(str(dll.resolve()));lib.ggml_commit.restype=ctypes.c_char_p;linked_ggml=lib.ggml_commit().decode()
    if digest(a.audio)!=source_hash:raise ValueError('Source audio changed during transcription')
    report=dict(source_path=str(a.audio.resolve()),source_sha256=source_hash,source_modified=False,source_audio=audio_info,
        command=command,transcriber_version=binary_version,transcriber_binary_sha256=digest(a.cli),transcriber_model_sha256=digest(a.model),
        linked_transcriber_ggml=linked_ggml,device_memory_sampling=memory,duration_seconds=events['duration_seconds'],
        event_count=len(events['events']),native_notes_by_track=dict(collections.Counter(n['track'] for e in events['events'] for n in e['notes'])),
        warnings=events.get('warnings',[]),midi_event_audits=checks,melody_role=a.melody_role,melody_excerpt=melody,chord_excerpt=chords,
        source_excerpt_start_seconds=0,source_excerpt_end_seconds=source_end,full_song=a.full_song,
        timing_note='MuLaCover consumes MIDI beat positions; MIDI render tempo and source timestamp interpolation are distinct, and generated tempo is not fixed',
        files={str(path.relative_to(a.output)):digest(path) for path in [a.output/'events.json',a.output/'score.abc',a.output/'source-excerpt.wav',*sorted(whole.glob('*.mid')),*sorted(excerpt.glob('*.mid'))]},
        execution_passed=True,musical_transcription_quality_accepted=False)
    (a.output/'transcription.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('command','files')},indent=2))


if __name__=='__main__':main()
