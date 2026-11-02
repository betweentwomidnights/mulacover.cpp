"""Preserve native SheetSage2 MIDI beat positions and audit/crop an explicit role."""
from collections import Counter,defaultdict,deque
from pathlib import Path
import mido


def midi_notes(path):
    midi=mido.MidiFile(path)
    if midi.type not in (0,1) or midi.ticks_per_beat<=0:raise ValueError('Expected a shared PPQ MIDI timeline')
    notes=[]
    for track in midi.tracks:
        tick=0;active=defaultdict(deque)
        for msg in track:
            tick+=msg.time
            if msg.type=='note_on' and msg.velocity>0:active[(msg.channel,msg.note)].append(tick)
            elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
                key=(msg.channel,msg.note)
                if not active[key]:raise ValueError('Unmatched MIDI note-off')
                start=active[key].popleft()
                if tick<=start:raise ValueError('Nonpositive MIDI note duration')
                notes.append(dict(pitch=msg.note,start_tick=start,end_tick=tick,channel=msg.channel,
                    onset_step=start*4/midi.ticks_per_beat,duration_steps=(tick-start)*4/midi.ticks_per_beat))
        if any(active.values()):raise ValueError('Unterminated MIDI notes')
    return midi,notes


def crop_midi(source,destination,sixteenths):
    midi,_=midi_notes(source)
    cutoff=round(sixteenths*midi.ticks_per_beat/4)
    output=mido.MidiFile(type=midi.type,ticks_per_beat=midi.ticks_per_beat)
    for track in midi.tracks:
        result=mido.MidiTrack();output.tracks.append(result);tick=0;previous=0;active=Counter()
        for msg in track:
            tick+=msg.time
            if tick>=cutoff:break
            if msg.type=='end_of_track':continue
            result.append(msg.copy(time=tick-previous));previous=tick
            if msg.type=='note_on' and msg.velocity>0:active[(msg.channel,msg.note)]+=1
            elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):active[(msg.channel,msg.note)]-=1
        for (channel,note),count in active.items():
            for _ in range(count):
                result.append(mido.Message('note_off',channel=channel,note=note,velocity=0,time=cutoff-previous));previous=cutoff
        result.append(mido.MetaMessage('end_of_track',time=cutoff-previous))
    Path(destination).parent.mkdir(parents=True,exist_ok=True);output.save(destination)
    _,notes=midi_notes(destination)
    return dict(notes=len(notes),duration_seconds=output.length,ticks_per_beat=output.ticks_per_beat,
        sixteenths=sixteenths,last_note_step=max((n['onset_step']+n['duration_steps'] for n in notes),default=0))


def audit_events(path,events,role):
    _,notes=midi_notes(path)
    ids={'vocal':{0},'instrumental':{1},'combined':{0,1}}[role]
    expected=Counter((e['subbeat'],n['pitch'],n['duration_steps']) for e in events for n in e['notes'] if n['track'] in ids)
    actual=Counter((round(n['onset_step'],8),n['pitch'],round(n['duration_steps'],8)) for n in notes)
    return dict(role=role,native_event_notes=sum(expected.values()),midi_notes=len(notes),
        beat_positions_pitches_durations_match=expected==actual,
        missing_notes=sum((expected-actual).values()),extra_notes=sum((actual-expected).values()))
