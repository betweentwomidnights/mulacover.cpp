#!/usr/bin/env python3
"""Render a MIDI lane as a simple fixed timbre to audition transcribed notes."""
import argparse
from collections import defaultdict,deque
import hashlib
import json
from pathlib import Path
import wave
import mido
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--midi',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing preview is preserved')
    midi=mido.MidiFile(a.midi)
    if midi.type not in (0,1) or midi.ticks_per_beat<=0:raise ValueError('Expected shared PPQ MIDI timeline')
    active=defaultdict(deque);notes=[];time=0.;tempo=500000
    for msg in mido.merge_tracks(midi.tracks):
        time+=mido.tick2second(msg.time,midi.ticks_per_beat,tempo)
        if msg.type=='set_tempo':tempo=msg.tempo
        elif msg.type=='note_on' and msg.velocity>0:active[(msg.channel,msg.note)].append(time)
        elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
            key=(msg.channel,msg.note)
            if not active[key]:raise ValueError('Unmatched note-off')
            start=active[key].popleft()
            if time<=start:raise ValueError('Nonpositive note duration')
            notes.append((start,time,msg.note))
    if any(active.values()):raise ValueError('Unterminated note')
    sr=24000;audio=np.zeros(int(np.ceil((time+.3)*sr)),np.float64)
    for start,end,pitch in notes:
        first=round(start*sr);count=max(1,round(end*sr)-first);t=np.arange(count)/sr
        hz=440*2**((pitch-69)/12);phase=2*np.pi*hz*t
        envelope=np.minimum(1,t/.008)*np.minimum(1,(count/sr-t)/.025)*np.exp(-1.5*t)
        audio[first:first+count]+=(np.sin(phase)+.22*np.sin(2*phase)+.08*np.sin(3*phase))*envelope
    peak=float(np.abs(audio).max(initial=0));gain=.65/peak if peak else 1
    pcm=np.rint(np.clip(audio*gain,-1,1)*32767).astype('<i2')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(a.output),'wb') as out:out.setparams((1,2,sr,0,'NONE','not compressed'));out.writeframes(pcm.tobytes())
    report=dict(midi_sha256=hashlib.sha256(a.midi.read_bytes()).hexdigest(),audio_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        note_count=len(notes),duration_seconds=len(audio)/sr,sample_rate=sr,preview_only=True,
        timbre='Fixed harmonic tone, equal note velocities; not recovered source instrumentation',
        timing='Exported MIDI tempo map, not source wall-clock alignment or MuLaCover generated timing',gain=gain)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
