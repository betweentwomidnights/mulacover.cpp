#!/usr/bin/env python3
"""Report onset-periodicity candidates and optional independent score-clock fits."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def read_wave(path):
    raw=Path(path).read_bytes()
    if raw[:4]!=b'RIFF' or raw[8:12]!=b'WAVE':raise ValueError('Expected little-endian RIFF WAV')
    fmt=data=None;offset=12
    while offset+8<=len(raw):
        key=raw[offset:offset+4];size=struct.unpack_from('<I',raw,offset+4)[0];payload=raw[offset+8:offset+8+size]
        if len(payload)!=size:raise ValueError('Truncated WAV')
        if key==b'fmt ':fmt=payload
        if key==b'data':data=payload
        offset+=8+size+(size%2)
    if fmt is None or data is None:raise ValueError('Missing WAV chunks')
    kind,channels,rate,_,_,bits=struct.unpack_from('<HHIIHH',fmt)
    if kind==1 and bits==16:x=np.frombuffer(data,'<i2').astype(np.float64)/32768
    elif kind==3 and bits==32:x=np.frombuffer(data,'<f4').astype(np.float64)
    else:raise ValueError('Tempo analysis supports PCM16 or float32 WAV')
    x=x.reshape(-1,channels).mean(axis=1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite audio')
    return rate,x


def onset_envelope(x,rate):
    hop=max(1,round(rate/100));window=2048;audio=np.pad(x,(window//2,window//2));frames=np.lib.stride_tricks.sliding_window_view(audio,window)[::hop]
    taper=np.hanning(window);freq=np.fft.rfftfreq(window,1/rate);band=(freq>=60)&(freq<=8000)
    envelope=[];previous=None
    for first in range(0,len(frames),256):
        spectrum=np.log1p(np.abs(np.fft.rfft(frames[first:first+256]*taper,axis=1))[:,band])
        diff=np.diff(spectrum,axis=0,prepend=(spectrum[:1] if previous is None else previous[None,:]))
        envelope.extend(np.maximum(diff,0).mean(axis=1));previous=spectrum[-1]
    return np.asarray(envelope),rate/hop


def tempo_candidates(envelope,fps):
    x=np.asarray(envelope,dtype=np.float64);x=x-x.mean()
    if np.dot(x,x)<=1e-20:return []
    lo=max(2,int(np.floor(60*fps/240)));hi=min(len(x)//3,int(np.ceil(60*fps/40)))
    if hi<=lo+2:return []
    scores={}
    for lag in range(lo-1,hi+2):
        a,b=x[:-lag],x[lag:];den=np.sqrt(np.dot(a,a)*np.dot(b,b));scores[lag]=float(np.dot(a,b)/den) if den else 0
    peaks=[]
    for lag in range(lo,hi+1):
        left,center,right=scores[lag-1],scores[lag],scores[lag+1]
        if center<=0 or center<left or center<right:continue
        curvature=left-2*center+right;shift=.5*(left-right)/curvature if curvature else 0
        bpm=60*fps/(lag+float(np.clip(shift,-.5,.5)))
        if 40<=bpm<=240:peaks.append(dict(bpm=bpm,correlation=center))
    return sorted(peaks,key=lambda p:p['correlation'],reverse=True)[:6]


def score_clock(path):
    score=json.loads(Path(path).read_text());events=[e for e in score['events'] if e.get('has_timestamp')]
    if len(events)<3:raise ValueError('Insufficient timestamped score events')
    t=np.array([e['time_seconds'] for e in events]);beats=np.array([e['subbeat']/4 for e in events]);fit=np.polyfit(t,beats,1)
    windows=[]
    for start in np.arange(0,max(1,t.max()-14),10):
        mask=(t>=start)&(t<start+15)
        if mask.sum()>=4:windows.append(dict(start_seconds=float(start),end_seconds=float(min(t.max(),start+15)),estimated_bpm=float(np.polyfit(t[mask],beats[mask],1)[0]*60)))
    return dict(estimated_bpm=float(fit[0]*60),clock_residual_beats_rms=float(np.sqrt(np.mean((beats-np.polyval(fit,t))**2))),timestamped_events=len(events),windows=windows,warnings=score.get('warnings',[]),
        method='Linear fit of SheetSage2 decoded musical beats against its predicted source timestamps; estimated, not ground-truth tempo')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audio',action='append',required=True,help='ID=PATH');p.add_argument('--events',action='append',default=[],help='ID=PATH');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Existing tempo report is preserved')
    events=dict(item.split('=',1) for item in a.events);cases={}
    for item in a.audio:
        name,path=item.split('=',1);rate,x=read_wave(path);env,fps=onset_envelope(x,rate);candidates=tempo_candidates(env,fps);windows=[]
        for start in range(0,max(1,int(len(x)/rate)-14),10):
            first=round(start*fps);last=round((start+15)*fps);windows.append(dict(start_seconds=start,candidates=tempo_candidates(env[first:last],fps)[:3]))
        cases[name]=dict(audio_path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),duration_seconds=len(x)/rate,onset_autocorrelation_candidates=candidates,windows=windows)
        if name in events:cases[name]['sheetsage2_score_clock']=score_clock(events[name])
        print(name,'onset candidates',[(round(c['bpm'],2),round(c['correlation'],3)) for c in candidates],flush=True)
    report=dict(numpy=np.__version__,method='Positive log-spectral flux, 2048-sample Hann window, approximately 100Hz envelope; normalized lag autocorrelation, local maxima between 40 and 240 BPM',
        limitation='Candidates include beat subdivisions and half/double-time aliases. Rank is not a musical beat annotation. Score-clock estimates are also model predictions. No hard adherence claim from either estimator alone.',cases=cases)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
