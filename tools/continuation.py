"""Select encoded audio guides and plan the 80-ms continuation timeline."""
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import hashlib
from pathlib import Path
import struct
import numpy as np

FRAME_SECONDS=Decimal('0.08')


def frames_for_seconds(value,allow_zero=False):
    try:seconds=Decimal(str(value))
    except InvalidOperation:raise ValueError('Duration must be a finite number of seconds') from None
    if not seconds.is_finite() or seconds<0:raise ValueError('Duration must be finite and nonnegative')
    frames=int((seconds/FRAME_SECONDS).to_integral_value(rounding=ROUND_FLOOR))
    if not allow_zero and frames<1:raise ValueError('Duration must include at least one complete 80-ms frame')
    return frames


def select_guide(path,seconds=None,start_seconds='0'):
    data=Path(path).read_bytes()
    if not data or len(data)%32:raise ValueError('Guide must contain complete int32[frames,8] RVQ frames')
    codes=np.frombuffer(data,dtype='<i4').reshape(-1,8)
    if np.any(codes<0) or np.any(codes>=8192):raise ValueError('Guide contains a reserved or invalid RVQ code')
    start=frames_for_seconds(start_seconds,allow_zero=True)
    count=frames_for_seconds(seconds) if seconds is not None else len(codes)-start
    if count<1 or start+count>len(codes):raise ValueError('Selected guide extends beyond available encoded audio')
    selected=np.ascontiguousarray(codes[start:start+count])
    metadata=dict(source_codes_sha256=hashlib.sha256(data).hexdigest(),source_frames=len(codes),
        requested_start_seconds=str(start_seconds),requested_guide_seconds=None if seconds is None else str(seconds),
        source_start_frame=start,source_start_seconds=start*.08,guide_frames=count,guide_seconds=count*.08,
        rounding='Start and duration independently rounded down to complete 80-ms frames')
    return selected,metadata


def plan_frames(prompt,prefix,frames=None,continuation_seconds=None,capacity=None):
    if frames is not None and continuation_seconds is not None:raise ValueError('Choose --frames or --continuation-seconds, not both')
    total=prefix+frames_for_seconds(continuation_seconds) if continuation_seconds is not None else (100 if frames is None else frames)
    required=prompt+total-1
    if prompt<1 or prefix<0 or total<=prefix or total>8192 or required>8192:raise ValueError('Guide and continuation must fit the 8192-position model context and leave new audio frames')
    actual_capacity=((required+255)//256)*256 if capacity is None else capacity
    if actual_capacity<required or actual_capacity>8192:raise ValueError(f'Insufficient/invalid cache capacity: need at least {required} positions')
    return total,actual_capacity


def write_continuation(source,destination,prefix_frames,total_frames):
    """Copy the generated suffix of the native float32 stereo WAV, unchanged."""
    size=(total_frames-prefix_frames)*3840*8
    if Path(source).resolve()==Path(destination).resolve():raise ValueError('Continuation destination must differ from the full WAV')
    if prefix_frames<1 or size<=0:raise ValueError('Continuation extraction needs a guide and new audio')
    with Path(source).open('rb') as src:
        header=bytearray(src.read(44))
        if len(header)!=44 or header[:4]!=b'RIFF' or header[8:16]!=b'WAVEfmt ' or header[36:40]!=b'data':raise ValueError('Expected native WAV header')
        if struct.unpack_from('<HHIIHH',header,20)!=(3,2,48000,384000,8,32):raise ValueError('Expected native 48kHz float32 stereo WAV')
        if Path(source).stat().st_size!=44+total_frames*3840*8:raise ValueError('Native WAV length mismatch')
        src.seek(44+prefix_frames*3840*8);struct.pack_into('<I',header,4,size+36);struct.pack_into('<I',header,40,size)
        with Path(destination).open('wb') as dst:
            dst.write(header)
            remaining=size
            while remaining:
                block=src.read(min(1024*1024,remaining))
                if not block:raise ValueError('Truncated continuation data')
                dst.write(block);remaining-=len(block)
