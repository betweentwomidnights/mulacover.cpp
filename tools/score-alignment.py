#!/usr/bin/env python3
"""Estimate timed melody preservation from SheetSage2 output; not a listening verdict."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from midi_bridge import midi_notes


def match(reference,observed,tolerance):
    """Maximum ordered one-to-one pitch-class/onset matches, deterministic ties.

    Octave changes are allowed; transposition is not. No tempo fitting or time
    warping is used to improve the score. Times are in seconds.
    """
    n,m=len(reference),len(observed);dp=np.zeros((n+1,m+1),np.int32)
    for i in range(1,n+1):
        t,p=reference[i-1]
        for j in range(1,m+1):
            u,q=observed[j-1]
            same=p%12==q%12 and abs(t-u)<=tolerance
            dp[i,j]=max(dp[i-1,j],dp[i,j-1],dp[i-1,j-1]+int(same))
    pairs=[];i,j=n,m
    while i and j:
        t,p=reference[i-1];u,q=observed[j-1]
        if p%12==q%12 and abs(t-u)<=tolerance and dp[i,j]==dp[i-1,j-1]+1:
            pairs.append((i-1,j-1));i-=1;j-=1
        elif dp[i-1,j]>=dp[i,j-1]:i-=1
        else:j-=1
    return list(reversed(pairs))


def analyze(midi,events,bpm,duration,tolerance=.25,offset_beats=0):
    _,notes=midi_notes(midi);score=json.loads(Path(events).read_text())
    reference=sorted(((n['onset_step']/4-offset_beats)*60/bpm,n['pitch']) for n in notes)
    reference=[n for n in reference if 0<=n[0]<duration]
    observed=sorted((e['time_seconds'],n['pitch']) for e in score['events'] if e.get('has_timestamp') for n in e['notes'] if n['track']==1 and 0<=e['time_seconds']<duration)
    pairs=match(reference,observed,tolerance);count=len(pairs)
    precision=count/len(observed) if observed else 0;recall=count/len(reference) if reference else 0
    errors=[observed[j][0]-reference[i][0] for i,j in pairs]
    return dict(target_bpm=bpm,duration_seconds=duration,onset_tolerance_seconds=tolerance,offset_beats=offset_beats,
        reference_notes=len(reference),transcribed_instrumental_notes=len(observed),ordered_matches=count,
        precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
        matched_onset_error_median_seconds=float(np.median(errors)) if errors else None,
        matched_onset_error_absolute_median_seconds=float(np.median(np.abs(errors))) if errors else None,
        reference_midi_sha256=hashlib.sha256(Path(midi).read_bytes()).hexdigest(),events_sha256=hashlib.sha256(Path(events).read_bytes()).hexdigest(),
        limitation='SheetSage2 re-transcription can make mistakes and choose another melody. Octave-equivalent pitches allowed; no transposition, fitted tempo or time warping. This estimates on-clock note preservation, not musical quality or a tempo guarantee.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('midi','events','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--bpm',type=float,required=True);p.add_argument('--duration',type=float,required=True)
    p.add_argument('--tolerance',type=float,default=.25);p.add_argument('--offset-beats',type=float,default=0)
    a=p.parse_args()
    if a.bpm<=0 or a.duration<=0 or a.tolerance<0:raise ValueError('Invalid alignment parameters')
    if a.output.exists():raise ValueError('Use a fresh report path')
    result=analyze(a.midi,a.events,a.bpm,a.duration,a.tolerance,a.offset_beats)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
